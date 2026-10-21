"""Shared Trainer assembly; adapters build models, not new training loops."""

from __future__ import annotations

import os
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from workflow.common import atomic_json, read_json, require, sha256


@dataclass
class TrainingPlan:
    module: Any
    datamodule: Any
    resume_key: str


def preflight_runtime(runtime: dict) -> None:
    allowed = {"accelerator", "devices", "strategy", "swanlab_mode"}
    require(not set(runtime) - allowed, "Unknown runtime option")
    # This runnable starter is CPU-only. A CUDA adapter must validate its own strategy.
    require(runtime.get("accelerator", "cpu") == "cpu", "Demo requires CPU")
    require(
        type(runtime.get("devices", 1)) is int and runtime.get("devices", 1) == 1,
        "Demo requires one CPU process",
    )
    require(runtime.get("strategy", "auto") == "auto", "Demo requires auto strategy")
    mode = runtime.get("swanlab_mode", "disabled")
    require(mode in {"disabled", "offline", "online"}, "Invalid SwanLab mode")
    if mode != "disabled":
        import importlib.metadata

        require(importlib.metadata.version("swanlab") == "0.9.8", "Install the tracking extra")
        importlib.metadata.version("lightning")  # SwanLab 0.9.8 uses this namespace.


def execute(plan: TrainingPlan, config: dict, run_dir: Path, resume: Path | None):
    import pytorch_lightning as L
    import torch
    from pytorch_lightning.callbacks import Callback

    runtime = read_json(run_dir / "RUN.json").get("execution", {})
    preflight_runtime(runtime)
    training = config["training"]
    ckpt_dir = run_dir / "ckpt"
    for directory in (ckpt_dir, run_dir / "view", run_dir / "log" / "swanlab"):
        directory.mkdir(parents=True, exist_ok=True)

    def save(trainer, path):
        temporary = path.with_suffix(".partial")
        try:
            trainer.save_checkpoint(temporary)
            os.replace(temporary, path)
            atomic_json(
                path.with_suffix(path.suffix + ".json"),
                {
                    "schema": 1,
                    "checkpoint_sha256": sha256(path),
                    "completed_steps": trainer.global_step,
                    "resume_key": plan.resume_key,
                    "torch": torch.__version__,
                    "lightning": L.__version__,
                },
            )
        finally:
            temporary.unlink(missing_ok=True)

    class Checkpoints(Callback):
        def __init__(self):
            self.last_step = -1
            self.paths = []

        def on_validation_end(self, trainer, module):
            atomic_json(
                run_dir / "view" / f"step-{trainer.global_step:06d}.json",
                {
                    "optimizer_step": trainer.global_step,
                    "metrics": {
                        k: float(v)
                        for k, v in trainer.callback_metrics.items()
                        if k.startswith("val/")
                    },
                },
            )

        def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx):
            step = trainer.global_step
            if step and step != self.last_step and step % training["checkpoint_every"] == 0:
                self.last_step = step
                path = ckpt_dir / f"step-{step:06d}.ckpt"
                save(trainer, path)
                self.paths.append(path)
                atomic_json(ckpt_dir / "latest.json", {"path": path.name})
                # Only remove generations created by this callback in this owned Run.
                while len(self.paths) > training["keep_checkpoints"]:
                    old = self.paths.pop(0)
                    old.unlink()
                    old.with_suffix(old.suffix + ".json").unlink()

    logger = False
    if runtime.get("swanlab_mode", "disabled") != "disabled":
        from swanlab.integration.pytorch_lightning import SwanLabLogger

        logger = SwanLabLogger(
            project=config["logging"]["project"],
            experiment_name=config["experiment"]["name"],
            tags=list(config["logging"]["tags"]),
            mode=runtime["swanlab_mode"],
            log_dir=str(run_dir / "log" / "swanlab"),
        )
    accumulation = config["trainer"]["accumulate_grad_batches"]
    trainer = L.Trainer(
        accelerator="cpu",
        devices=1,
        strategy="auto",
        max_steps=training["max_steps"],
        precision=config["trainer"]["precision"],
        accumulate_grad_batches=accumulation,
        gradient_clip_val=config["trainer"]["gradient_clip_val"],
        val_check_interval=training["eval_every"] * accumulation,
        check_val_every_n_epoch=None,
        num_sanity_val_steps=0,
        callbacks=[Checkpoints()],
        logger=logger,
        log_every_n_steps=1,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        deterministic=True,
        default_root_dir=run_dir,
    )
    failure = None
    try:
        if logger:
            logger.log_hyperparams(deepcopy(config))
        if training["eval_before_training"] and resume is None:
            trainer.validate(plan.module, datamodule=plan.datamodule, verbose=False)
        trainer.fit(plan.module, datamodule=plan.datamodule, ckpt_path=resume)
        if training["eval_at_end"]:
            trainer.validate(plan.module, datamodule=plan.datamodule, verbose=False)
        save(trainer, ckpt_dir / "last.ckpt")
        return trainer
    except BaseException as exc:
        failure = exc
        raise
    finally:
        if logger:
            import swanlab

            try:
                swanlab.finish(state="crashed" if failure else "success", async_log_timeout=30)
            except Exception:
                if failure is None:
                    raise

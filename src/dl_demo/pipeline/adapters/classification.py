"""Registered classification contract and assembly, with no training loop."""

from __future__ import annotations

import math
from copy import deepcopy

from dl_training.lightning import TrainingPlan, execute, preflight_runtime  # noqa: F401
from workflow.common import atomic_json, digest, read_json, require, sha256


def resume_key(config):
    compatible = deepcopy(config)
    compatible["training"].pop("max_steps")
    compatible["experiment"].pop("name")
    return digest({"contract": "lightning-classification-v1", "config": compatible})


def validate(config):
    fields = {
        "data": {"samples", "batch_size", "num_workers"},
        "model": {"hidden"},
        "optimizer": {"learning_rate"},
        "training": {
            "max_steps",
            "eval_every",
            "eval_before_training",
            "eval_at_end",
            "checkpoint_every",
            "keep_checkpoints",
        },
        "trainer": {"precision", "accumulate_grad_batches", "gradient_clip_val"},
        "experiment": {"name", "group"},
        "logging": {"project", "tags"},
    }
    require(
        set(config) == {"schema", "seed", "inputs", "expected_artifacts", *fields},
        "Unknown/missing config fields",
    )
    require(
        config["schema"] == 2 and not config["inputs"], "Demo requires schema 2 and generated data"
    )
    for section, keys in fields.items():
        require(
            isinstance(config[section], dict) and set(config[section]) == keys,
            f"Invalid {section} fields",
        )
    for section, keys in {
        "data": ["samples", "batch_size"],
        "model": ["hidden"],
        "trainer": ["accumulate_grad_batches"],
        "training": ["max_steps", "eval_every", "checkpoint_every", "keep_checkpoints"],
    }.items():
        for key in keys:
            require(
                type(config[section][key]) is int and config[section][key] > 0,
                f"Invalid {section}.{key}",
            )
    require(
        type(config["data"]["num_workers"]) is int and config["data"]["num_workers"] == 0,
        "Demo uses no loader workers",
    )
    # Identical complete batches make the demo sampler resumable without changing data order.
    require(config["data"]["samples"] == config["data"]["batch_size"], "Demo requires full batches")
    require(config["trainer"]["precision"] == "32-true", "Demo requires FP32")
    for section, key in [("optimizer", "learning_rate"), ("trainer", "gradient_clip_val")]:
        value = config[section][key]
        require(
            type(value) in (float, int) and math.isfinite(value) and value > 0,
            f"Invalid {section}.{key}",
        )
    for key in ["eval_before_training", "eval_at_end"]:
        require(config["training"][key] is True, f"Demo requires training.{key}")
    for section, key in [("logging", "project"), ("experiment", "name"), ("experiment", "group")]:
        require(
            type(config[section][key]) is str and bool(config[section][key].strip()),
            f"Invalid {section}.{key}",
        )
    require(
        isinstance(config["logging"]["tags"], list)
        and all(type(t) is str for t in config["logging"]["tags"]),
        "Invalid logging.tags",
    )


def preflight(config, bindings, resume_metadata):
    import pytorch_lightning as L
    import torch

    if resume_metadata:
        require(resume_metadata["resume_key"] == resume_key(config), "Incompatible resume config")
        step = resume_metadata["completed_steps"]
        require(
            type(step) is int and 0 < step < config["training"]["max_steps"],
            "Resume step must precede the new step budget",
        )
        require(
            resume_metadata["torch"] == torch.__version__
            and resume_metadata["lightning"] == L.__version__,
            "Resume requires the same torch/Lightning versions",
        )
    return {"torch": torch.__version__, "lightning": L.__version__, "resume": bool(resume_metadata)}


def build(config):
    from dl_demo.data.classification import ClassificationData
    from dl_demo.lightning.classification import ClassificationModule
    from dl_demo.model.classifier import Classifier

    return TrainingPlan(
        module=ClassificationModule(Classifier(config["model"]["hidden"]), config),
        datamodule=ClassificationData(
            config["data"],
            config["seed"],
            config["training"]["max_steps"] * config["trainer"]["accumulate_grad_batches"],
        ),
        resume_key=resume_key(config),
    )


def run(config, run_dir, bindings, resume):
    import pytorch_lightning as L
    import torch

    torch.set_num_threads(1)
    L.seed_everything(config["seed"], workers=True)
    plan = build(config)
    trainer = execute(plan, config, run_dir, resume)
    with torch.no_grad():
        x, y = plan.datamodule.train.tensors
        final_loss = float(torch.nn.functional.cross_entropy(plan.module.model(x), y))
    atomic_json(
        run_dir / "metrics.json",
        {
            "completed_steps": trainer.global_step,
            "initial_loss": plan.module.initial_loss,
            "final_loss": final_loss,
            "validation_accuracy": float(trainer.callback_metrics["val/accuracy"]),
            "resumed_from_step": plan.module.resumed_from_step,
            "validation_steps": plan.module.validation_steps,
        },
    )


def validate_outputs(config, run_dir):
    import torch

    metrics = read_json(run_dir / "metrics.json")
    path = run_dir / "ckpt/last.ckpt"
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    metadata = read_json(path.with_suffix(".ckpt.json"))
    require(
        checkpoint["global_step"]
        == metrics["completed_steps"]
        == metadata["completed_steps"]
        == config["training"]["max_steps"],
        "Step mismatch",
    )
    require(metadata["checkpoint_sha256"] == sha256(path), "Checkpoint hash mismatch")
    require(metadata["resume_key"] == resume_key(config), "Checkpoint identity mismatch")
    require(
        all(bool(torch.isfinite(v).all()) for v in checkpoint["state_dict"].values()),
        "Nonfinite model parameters",
    )
    require(
        all(
            math.isfinite(metrics[k]) for k in ("initial_loss", "final_loss", "validation_accuracy")
        ),
        "Nonfinite metrics",
    )
    require(
        metrics["final_loss"] < metrics["initial_loss"] and metrics["validation_accuracy"] >= 0.9,
        "Demo quality acceptance failed",
    )
    return metrics

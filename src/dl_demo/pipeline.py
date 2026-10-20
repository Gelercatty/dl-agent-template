"""Deterministic binary classification demonstrating checkpoint and resume contracts."""

from __future__ import annotations

import math
import os
from pathlib import Path

from workflow.common import atomic_json, digest, read_json, require, sha256


def resume_key(config: dict) -> str:
    # Changing the total step budget is supported. Every other config field must match.
    return digest(
        {
            "adapter_contract": "demo-v1",
            "config": {k: v for k, v in config.items() if k != "max_steps"},
        }
    )


def validate(config: dict) -> None:
    require(
        set(config) == {"schema", "seed", "max_steps", "inputs", "expected_artifacts", "task"},
        "Unknown/missing demo config fields",
    )
    task = config["task"]
    require(
        set(task) == {"samples", "hidden", "learning_rate", "checkpoint_every"},
        "Unknown/missing demo task fields",
    )
    for key in ("samples", "hidden", "checkpoint_every"):
        require(type(task[key]) is int and task[key] > 0, f"task.{key} must be positive")
    require(
        isinstance(task["learning_rate"], (float, int))
        and math.isfinite(task["learning_rate"])
        and 0 < task["learning_rate"] <= 1,
        "Invalid learning rate",
    )
    require(not config["inputs"], "Demo data is generated locally from the seed")


def preflight(config: dict, bindings: dict, resume_metadata: dict | None) -> dict:
    import torch

    if resume_metadata:
        require(resume_metadata["resume_key"] == resume_key(config), "Incompatible resume config")
        require(
            type(resume_metadata["completed_steps"]) is int
            and 0 < resume_metadata["completed_steps"] < config["max_steps"],
            "Resume step must precede the new step budget",
        )
        require(
            resume_metadata["torch"] == torch.__version__, "Resume requires the same torch version"
        )
    return {
        "torch": torch.__version__,
        "device": "cpu",
        "threads": 1,
        "data": "seeded synthetic train/validation split",
        "resume": bool(resume_metadata),
    }


def save_checkpoint(path: Path, state: dict, config: dict) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    try:
        torch.save(state, temporary)
        os.replace(temporary, path)
        atomic_json(
            path.with_suffix(path.suffix + ".json"),
            {
                "schema": 1,
                "checkpoint_sha256": sha256(path),
                "resume_key": resume_key(config),
                "completed_steps": state["step"],
                "torch": torch.__version__,
            },
        )
    finally:
        temporary.unlink(missing_ok=True)


def run(config: dict, run_dir: Path, bindings: dict, resume: Path | None) -> None:
    import torch
    from torch import nn

    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(config["seed"])
    task = config["task"]
    model = nn.Sequential(nn.Linear(2, task["hidden"]), nn.Tanh(), nn.Linear(task["hidden"], 2))
    optimizer = torch.optim.Adam(model.parameters(), lr=task["learning_rate"])
    generator = torch.Generator().manual_seed(config["seed"] + 1000)

    def dataset():
        y = torch.randint(0, 2, (task["samples"],), generator=generator)
        x = torch.randn(task["samples"], 2, generator=generator) * 0.35
        x += (y.float() * 2 - 1).unsqueeze(1)
        return x, y

    train_x, train_y = dataset()
    val_x, val_y = dataset()
    loss_fn = nn.CrossEntropyLoss()
    start = 0
    with torch.no_grad():
        initial_loss = float(loss_fn(model(train_x), train_y))
    if resume:
        checkpoint = torch.load(resume, map_location="cpu", weights_only=True)
        require(checkpoint["resume_key"] == resume_key(config), "Checkpoint config mismatch")
        metadata = read_json(resume.with_suffix(resume.suffix + ".json"))
        require(checkpoint["step"] == metadata["completed_steps"], "Checkpoint step mismatch")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        torch.set_rng_state(checkpoint["rng"])
        start, initial_loss = checkpoint["step"], checkpoint["initial_loss"]
    for step in range(start + 1, config["max_steps"] + 1):
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(model(train_x), train_y)
        require(bool(torch.isfinite(loss)), f"Non-finite loss at step {step}")
        loss.backward()
        optimizer.step()
        print(f"step={step} loss={float(loss.detach()):.8f}", flush=True)
        state = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "rng": torch.get_rng_state(),
            "step": step,
            "initial_loss": initial_loss,
            "resume_key": resume_key(config),
        }
        if step % task["checkpoint_every"] == 0:
            # Unique checkpoint generations survive interruption of a later write.
            checkpoint_path = run_dir / "ckpt" / f"step-{step:06d}.pt"
            save_checkpoint(checkpoint_path, state, config)
            atomic_json(run_dir / "ckpt" / "latest.json", {"path": checkpoint_path.name})
    save_checkpoint(run_dir / "ckpt" / "last.pt", state, config)
    with torch.no_grad():
        final_loss = float(loss_fn(model(train_x), train_y))
        accuracy = float((model(val_x).argmax(dim=1) == val_y).float().mean())
    atomic_json(
        run_dir / "metrics.json",
        {
            "completed_steps": config["max_steps"],
            "initial_loss": initial_loss,
            "final_loss": final_loss,
            "validation_accuracy": accuracy,
            "resumed_from_step": start,
        },
    )


def validate_outputs(config: dict, run_dir: Path) -> dict:
    import torch

    metrics = read_json(run_dir / "metrics.json")
    checkpoint_path = run_dir / "ckpt" / "last.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    metadata = read_json(checkpoint_path.with_suffix(".pt.json"))
    require(
        checkpoint["step"] == metrics["completed_steps"] == config["max_steps"], "Step mismatch"
    )
    require(metadata["checkpoint_sha256"] == sha256(checkpoint_path), "Checkpoint hash mismatch")
    require(
        checkpoint["resume_key"] == metadata["resume_key"] == resume_key(config),
        "Checkpoint identity mismatch",
    )
    require(
        all(bool(torch.isfinite(v).all()) for v in checkpoint["model"].values()),
        "Non-finite model parameters",
    )
    require(
        all(
            math.isfinite(metrics[k]) for k in ("initial_loss", "final_loss", "validation_accuracy")
        ),
        "Non-finite metrics",
    )
    require(metrics["final_loss"] < metrics["initial_loss"], "Demo loss did not improve")
    require(metrics["validation_accuracy"] >= 0.9, "Demo validation accuracy is below 90%")
    return metrics

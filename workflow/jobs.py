"""Prepare fixed job definitions and execute one controller per run."""

from __future__ import annotations

import contextlib
import importlib
import json
import os
import platform
import re
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path

from .common import (
    atomic_json,
    atomic_text,
    digest,
    label,
    now,
    read_json,
    regular_file,
    require,
    safe_relative,
    sha256,
)
from .release import verify_release


def config_contract(config: dict) -> None:
    require(config.get("schema") == 1, "Unsupported config schema")
    require(
        type(config.get("max_steps")) is int and config["max_steps"] > 0,
        "max_steps must be a positive integer",
    )
    require(type(config.get("seed")) is int, "seed must be an integer")
    require(isinstance(config.get("inputs"), dict), "inputs must be an object")
    artifacts = config.get("expected_artifacts")
    require(
        isinstance(artifacts, list) and artifacts and len(set(artifacts)) == len(artifacts),
        "expected_artifacts must list distinct output files",
    )
    reserved = {
        "JOB_SUCCESS.json",
        "JOB_FAILED.json",
        "RUN.json",
        "TRAIN_COMPLETE.json",
        "train.log",
    }
    for name in artifacts:
        safe_relative(name)
        require(name not in reserved, f"Controller-owned output: {name}")
    for name, spec in config["inputs"].items():
        label(name)
        require(
            isinstance(spec, dict)
            and isinstance(spec.get("id"), str)
            and bool(re.fullmatch(r"[0-9a-f]{64}", spec.get("sha256", ""))),
            f"Input {name} needs a logical id and SHA-256",
        )


def registration(release: Path, pipeline: str) -> tuple[dict, dict]:
    registry = read_json(regular_file(release, "configs/registry.json"))
    require(registry.get("schema") == 1, "Unsupported pipeline registry schema")
    require(pipeline in registry["pipelines"], f"Unregistered pipeline: {pipeline}")
    entry = registry["pipelines"][pipeline]
    require(
        bool(re.fullmatch(r"[a-zA-Z_]\w*(\.[a-zA-Z_]\w*)*", entry["adapter"])),
        "Adapter must be a Python module name",
    )
    config = read_json(regular_file(release, entry["config"]))
    config_contract(config)
    return entry, config


def prepare_job(
    release: Path,
    workspace: Path,
    name: str,
    pipeline: str,
    plan: str,
    bindings: Path | None = None,
    resume: Path | None = None,
) -> Path:
    release, workspace = release.resolve(), workspace.resolve()
    require(not workspace.is_relative_to(release), "Runtime workspace cannot be inside a release")
    require(
        not any((p / ".git").exists() for p in (workspace, *workspace.parents)),
        "Runtime workspace must be outside Git source trees",
    )
    manifest = verify_release(release)
    entry, config = registration(release, pipeline)
    plan_text = regular_file(release, plan).read_text(encoding="utf-8")
    require(len(plan_text.strip()) >= 40, "Provide a real pre-run plan")
    job_id = label(name)
    destination, run = workspace / "Jobs" / job_id, workspace / "Runs" / job_id
    require(not destination.exists() and not run.exists(), f"Job/run already exists: {job_id}")
    input_paths = read_json(bindings) if bindings else {}
    require(set(input_paths) == set(config["inputs"]), "Input bindings do not match config inputs")
    input_paths = {k: str(Path(v).expanduser().resolve()) for k, v in input_paths.items()}
    resume_spec = None
    if resume:
        resume = resume.resolve()
        resume_spec = {
            "path": str(resume),
            "sha256": sha256(resume),
            "metadata_sha256": sha256(resume.with_suffix(resume.suffix + ".json")),
        }
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".job-", dir=destination.parent))
    try:
        atomic_json(staging / "config.json", config)
        atomic_text(staging / "PLAN.md", plan_text)
        script = (
            "#!/usr/bin/env bash\nset -euo pipefail\n"
            ': "${PYTHON_BIN:=python3}"\n'
            f"cd {shlex.quote(str(release))}\n"
            "export PYTHONDONTWRITEBYTECODE=1\n"
            f"export PYTHONPATH={shlex.quote(str(release) + ':' + str(release / 'src'))}\n"
            'exec "$PYTHON_BIN" -B -m workflow execute --job '
            f'{shlex.quote(str(destination))} "$@"\n'
        )
        atomic_text(staging / "run.sh", script)
        (staging / "run.sh").chmod(0o555)
        spec = {
            "schema": 1,
            "job_id": job_id,
            "created_at": now(),
            "release": str(release),
            "release_id": manifest["release_id"],
            "pipeline": pipeline,
            "adapter": entry["adapter"],
            "run": str(run),
            "bindings": input_paths,
            "resume": resume_spec,
            "files": {p: sha256(staging / p) for p in ("config.json", "PLAN.md", "run.sh")},
        }
        atomic_json(staging / "JOB.json", {**spec, "definition_sha256": digest(spec)})
        atomic_text(
            staging / "TRAINING_RECORD.md",
            f"# Training record: {job_id}\n\n## Before execution\n\n{plan_text}\n\n"
            "## Execution evidence\n\nPrepared; not started.\n\n"
            "## Interpretation\n\nPending human/agent review against the plan.\n",
        )
        for p in ("config.json", "PLAN.md", "JOB.json"):
            (staging / p).chmod(0o444)
        staging.rename(destination)
        return destination
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def verify_job(job: Path) -> tuple[dict, dict]:
    spec = read_json(regular_file(job, "JOB.json"))
    identity = {k: v for k, v in spec.items() if k != "definition_sha256"}
    require(
        spec.get("schema") == 1 and digest(identity) == spec["definition_sha256"],
        "Job definition changed",
    )
    require(set(spec["files"]) == {"config.json", "PLAN.md", "run.sh"}, "Job file contract changed")
    for name, expected in spec["files"].items():
        require(sha256(regular_file(job, name)) == expected, f"Job file changed: {name}")
    release = Path(spec["release"])
    manifest = verify_release(release)
    require(manifest["release_id"] == spec["release_id"], "Wrong release for job")
    entry, config = registration(release, spec["pipeline"])
    require(entry["adapter"] == spec["adapter"], "Job adapter differs from registered pipeline")
    require(read_json(job / "config.json") == config, "Job config differs from frozen config")
    return spec, config


def frozen_command(job: Path, dry_run: bool = False) -> list[str]:
    spec, _ = verify_job(job)
    return ["bash", str(job / "run.sh"), *(["--dry-run"] if dry_run else [])]


def check_job(job: Path) -> None:
    subprocess.run(
        frozen_command(job, dry_run=True),
        check=True,
        env={**os.environ, "PYTHON_BIN": sys.executable},
    )


def preflight(job: Path) -> tuple[dict, dict, object, dict]:
    spec, config = verify_job(job)
    # Import project code only after checking the frozen source; avoid editable checkout leakage.
    release = Path(spec["release"])
    require(
        Path(__file__).resolve().parent.parent == release,
        "Execute through the generated run.sh to use the frozen controller",
    )
    require(
        os.environ.get("WORLD_SIZE", "1") == "1" and os.environ.get("RANK", "0") == "0",
        "Launch one controller; a distributed adapter must own torchrun internally",
    )
    require(not Path(spec["run"]).exists(), "Run already exists; use a new job for every attempt")
    require(set(spec["bindings"]) == set(config["inputs"]), "Input bindings changed")
    for name, data in config["inputs"].items():
        require(sha256(Path(spec["bindings"][name])) == data["sha256"], f"Input changed: {name}")
    resume_metadata = None
    if spec["resume"]:
        checkpoint = Path(spec["resume"]["path"])
        metadata = checkpoint.with_suffix(checkpoint.suffix + ".json")
        require(sha256(checkpoint) == spec["resume"]["sha256"], "Resume checkpoint changed")
        require(sha256(metadata) == spec["resume"]["metadata_sha256"], "Resume metadata changed")
        resume_metadata = read_json(metadata)
        require(
            resume_metadata.get("schema") == 1
            and resume_metadata["checkpoint_sha256"] == sha256(checkpoint),
            "Resume checkpoint/metadata mismatch",
        )
    sys.path.insert(0, str(release / "src"))
    adapter = importlib.import_module(spec["adapter"])
    require(
        Path(adapter.__file__).resolve().is_relative_to(release), "Adapter loaded outside release"
    )
    adapter.validate(config)
    runtime = adapter.preflight(config, spec["bindings"], resume_metadata)
    return spec, config, adapter, runtime


def append_evidence(job: Path, evidence: dict) -> None:
    path = job / "TRAINING_RECORD.md"
    atomic_text(
        path,
        path.read_text(encoding="utf-8")
        + "\n## Controller result\n\n```json\n"
        + json.dumps(evidence, indent=2, ensure_ascii=False)
        + "\n```\n\n"
        "Review task quality, limitations and next steps; controller success only verifies "
        "the pipeline's execution contract.\n",
    )


def execute(job: Path, dry_run: bool = False) -> None:
    spec, config, adapter, runtime = preflight(job)
    if dry_run:
        print(
            json.dumps(
                {
                    "status": "ready",
                    "job_id": spec["job_id"],
                    "release_id": spec["release_id"],
                    "run": spec["run"],
                    "runtime": runtime,
                },
                indent=2,
            )
        )
        return
    run = Path(spec["run"])
    run.parent.mkdir(parents=True, exist_ok=True)
    run.mkdir()  # Exclusive claim: a second launcher cannot write this run.
    context = {
        "schema": 1,
        "job_id": spec["job_id"],
        "release_id": spec["release_id"],
        "definition_sha256": spec["definition_sha256"],
        "started_at": now(),
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "runtime": runtime,
        "resume": spec["resume"],
    }
    previous_handlers = {}

    def interrupted(signum: int, _frame: object) -> None:
        raise InterruptedError(f"Received signal {signum}")

    try:
        for sig in (signal.SIGTERM, signal.SIGINT):
            previous_handlers[sig] = signal.signal(sig, interrupted)
        atomic_json(run / "RUN.json", context)
        print(f"Running {spec['job_id']}; log: {run / 'train.log'}", flush=True)
        with (run / "train.log").open("w", buffering=1, encoding="utf-8") as log:
            with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                try:
                    adapter.run(
                        config,
                        run,
                        spec["bindings"],
                        Path(spec["resume"]["path"]) if spec["resume"] else None,
                    )
                    result = adapter.validate_outputs(config, run)
                    require(
                        type(result.get("completed_steps")) is int
                        and result["completed_steps"] == config["max_steps"],
                        "Training did not reach the planned step",
                    )
                    artifacts = {}
                    for name in config["expected_artifacts"]:
                        path = regular_file(run, name)
                        require(path.stat().st_size > 0, f"Empty output: {name}")
                        artifacts[name] = {"sha256": sha256(path), "bytes": path.stat().st_size}
                except BaseException:
                    traceback.print_exc()
                    raise
        # Recheck definitions after training so an in-place change cannot claim reproducibility.
        verify_job(job)
        evidence = {
            **context,
            "finished_at": now(),
            "status": "success",
            "result": result,
            "artifacts": artifacts,
        }
        atomic_json(run / "TRAIN_COMPLETE.json", result)
        append_evidence(job, evidence)
        atomic_json(run / "JOB_SUCCESS.json", evidence)
        print(f"Success: {run}")
    except BaseException as exc:
        # Failure is recorded for exceptions and catchable signals; SIGKILL has no terminal marker.
        (run / "TRAIN_COMPLETE.json").unlink(missing_ok=True)
        (run / "JOB_SUCCESS.json").unlink(missing_ok=True)
        evidence = {
            **context,
            "finished_at": now(),
            "status": "interrupted"
            if isinstance(exc, (KeyboardInterrupt, InterruptedError))
            else "failed",
            "error": f"{type(exc).__name__}: {exc}",
        }
        atomic_json(run / "JOB_FAILED.json", evidence)
        append_evidence(job, evidence)
        raise
    finally:
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)


def status(job: Path) -> dict:
    spec, config = verify_job(job)
    run = Path(spec["run"])
    success, failed = run / "JOB_SUCCESS.json", run / "JOB_FAILED.json"
    require(not (success.exists() and failed.exists()), "Conflicting terminal markers")
    if success.exists():
        evidence = read_json(success)
        require(evidence["definition_sha256"] == spec["definition_sha256"], "Wrong job evidence")
        require(evidence["status"] == "success", "Invalid success status")
        require(
            evidence["result"]["completed_steps"] == config["max_steps"]
            and read_json(run / "TRAIN_COMPLETE.json") == evidence["result"],
            "Completion evidence mismatch",
        )
        require(
            set(evidence["artifacts"]) == set(config["expected_artifacts"]),
            "Missing artifact evidence",
        )
        for name, entry in evidence["artifacts"].items():
            require(sha256(regular_file(run, name)) == entry["sha256"], f"Output changed: {name}")
        return evidence
    if failed.exists():
        evidence = read_json(failed)
        require(evidence["definition_sha256"] == spec["definition_sha256"], "Wrong job evidence")
        return evidence
    return {
        "job_id": spec["job_id"],
        "status": "running_or_unfinished" if run.exists() else "prepared",
        "note": "No terminal marker is not evidence of success or a live process.",
    }

"""Exercise filesystem and process boundaries, including a real CPU training continuation."""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from workflow.common import atomic_json, read_json
from workflow.jobs import prepare_job, status, verify_job
from workflow.release import create_release, verify_release

SOURCE = Path(__file__).resolve().parents[1]


@pytest.fixture
def workspace(tmp_path):
    # Include a space to exercise launcher quoting on developer machines.
    code, runtime = tmp_path / "my code", tmp_path / "runtime"
    shutil.copytree(
        SOURCE,
        code,
        ignore=shutil.ignore_patterns(
            ".git",
            ".venv",
            "__pycache__",
            ".pytest_cache",
            ".ruff_cache",
            ".local",
            "scratch",
        ),
    )
    for command in (
        ["init", "-b", "main"],
        ["add", "."],
        [
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "-m",
            "Initial fixture",
        ],
    ):
        subprocess.run(["git", "-C", str(code), *command], check=True, capture_output=True)
    return code, runtime


def launch(job: Path, *args: str, check: bool = True):
    return subprocess.run(
        ["bash", str(job / "run.sh"), *args],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHON_BIN": sys.executable},
        check=check,
        timeout=60,
    )


def job_for(workspace, name="test", pipeline="demo", resume=None):
    code, runtime = workspace
    release = create_release(code, runtime)
    return prepare_job(release, runtime, name, pipeline, "plans/demo.md", resume=resume)


def fake_pipeline(code: Path, run_body: str, result: str = '{"completed_steps": 40}'):
    path = code / "src" / "dl_demo" / "fake.py"
    path.write_text(
        "def validate(config): pass\n"
        "def preflight(config, bindings, resume): return {}\n"
        "def run(config, run_dir, bindings, resume):\n"
        + "\n".join("    " + line for line in run_body.splitlines())
        + f"\ndef validate_outputs(config, run_dir): return {result}\n"
    )
    registry = read_json(code / "configs/registry.json")
    registry["pipelines"]["demo"]["adapter"] = "dl_demo.fake"
    atomic_json(code / "configs/registry.json", registry)


def test_freeze_captures_dirty_and_new_source_but_not_private_files(workspace):
    code, runtime = workspace
    (code / "src/new.py").write_text("ANSWER = 42\n")
    (code / ".local").mkdir()
    (code / ".local/secrets.json").write_text('{"token": "private-test-value"}')
    release = create_release(code, runtime)
    manifest = verify_release(release)
    assert manifest["git"]["dirty"] is True
    assert "src/new.py" in manifest["files"]
    assert not (release / ".local").exists()
    assert create_release(code, runtime) == release
    (code / "src/new.py").write_text("ANSWER = 43\n")
    assert create_release(code, runtime) != release
    assert (release / "src/new.py").read_text() == "ANSWER = 42\n"


def test_runtime_must_not_be_inside_source(workspace):
    code, runtime = workspace
    with pytest.raises(ValueError, match="outside source"):
        create_release(code, code / "runtime")
    release = create_release(code, runtime)
    with pytest.raises(ValueError, match="outside Git"):
        prepare_job(release, code / "runtime", "nested", "demo", "plans/demo.md")


@pytest.mark.parametrize("kind", ["secret", "symlink"])
def test_freeze_rejects_selected_credential_or_symlink(workspace, kind):
    code, runtime = workspace
    if kind == "secret":
        (code / "configs/.env").write_text("TOKEN=private-test-value\n")
        subprocess.run(["git", "-C", str(code), "add", "-f", "configs/.env"], check=True)
    else:
        (code / "src/leak.py").symlink_to(code / "README.md")
    with pytest.raises(ValueError, match="Credential|Symlink"):
        create_release(code, runtime)
    assert not list((runtime / "Releases").glob("src-*"))


@pytest.mark.parametrize("change", ["changed", "extra", "symlink"])
def test_release_verification_rejects_changed_or_extra_source(workspace, change):
    code, runtime = workspace
    release = create_release(code, runtime)
    if change == "changed":
        path = release / "src/dl_demo/pipeline.py"
        path.chmod(0o644)
        path.write_text("raise RuntimeError('tampered')\n")
    else:
        release.chmod(0o755)
        if change == "extra":
            (release / "extra.py").write_text("untracked code")
        else:
            (release / "linked-dir").symlink_to(code, target_is_directory=True)
    with pytest.raises(ValueError, match="changed|symlink"):
        verify_release(release)


def test_dry_run_does_not_create_run_or_change_job(workspace):
    job = job_for(workspace)
    before = {p.name: p.read_bytes() for p in job.iterdir()}
    result = launch(job, "--dry-run")
    assert json.loads(result.stdout)["status"] == "ready"
    assert not (workspace[1] / "Runs").exists()
    assert before == {p.name: p.read_bytes() for p in job.iterdir()}


def test_config_changes_and_unregistered_pipeline_are_rejected(workspace):
    job = job_for(workspace)
    config = job / "config.json"
    config.chmod(0o644)
    config.write_text("{}")
    with pytest.raises(ValueError, match="Job file changed"):
        verify_job(job)
    with pytest.raises(ValueError, match="Unregistered"):
        job_for(workspace, "unknown", pipeline="typo")


def test_real_cpu_training_frozen_source_and_exact_resume(workspace):
    import torch

    code, runtime = workspace
    release = create_release(code, runtime)
    first = prepare_job(release, runtime, "first", "demo", "plans/demo.md")
    (code / "src/dl_demo/pipeline.py").write_text("raise RuntimeError('mutable code used')\n")
    launch(first)
    evidence = status(first)
    assert evidence["status"] == "success"
    assert evidence["result"]["completed_steps"] == 40
    assert evidence["result"]["validation_accuracy"] >= 0.9
    checkpoint = runtime / "Runs/first/ckpt/last.pt"
    resumed = prepare_job(
        release, runtime, "resumed", "demo-long", "plans/resume-demo.md", resume=checkpoint
    )
    baseline = prepare_job(release, runtime, "baseline", "demo-long", "plans/resume-demo.md")
    launch(resumed)
    launch(baseline)
    assert status(resumed)["result"]["resumed_from_step"] == 40
    a = torch.load(runtime / "Runs/resumed/ckpt/last.pt", weights_only=True)
    b = torch.load(runtime / "Runs/baseline/ckpt/last.pt", weights_only=True)
    for name in a["model"]:
        assert torch.equal(a["model"][name], b["model"][name])
    assert torch.equal(a["rng"], b["rng"])
    for key, value in a["optimizer"]["state"].items():
        for name, tensor in value.items():
            assert torch.equal(tensor, b["optimizer"]["state"][key][name])
    assert not (runtime / "Runs/resumed/JOB_FAILED.json").exists()
    assert "Controller result" in (resumed / "TRAINING_RECORD.md").read_text()
    assert "Hypothesis" in (first / "PLAN.md").read_text()
    # Same checkpoint cannot resume a run whose budget has already been reached.
    invalid = prepare_job(release, runtime, "invalid", "demo", "plans/demo.md", resume=checkpoint)
    rejected = launch(invalid, "--dry-run", check=False)
    assert rejected.returncode != 0 and "Resume step" in rejected.stderr
    assert not (runtime / "Runs/invalid").exists()
    # A changed checkpoint is detected before any new run is created.
    other = prepare_job(
        release, runtime, "corrupt", "demo-long", "plans/resume-demo.md", resume=checkpoint
    )
    with checkpoint.open("ab") as stream:
        stream.write(b"corrupt")
    rejected = launch(other, "--dry-run", check=False)
    assert rejected.returncode != 0 and "checkpoint changed" in rejected.stderr
    assert not (runtime / "Runs/corrupt").exists()
    with pytest.raises(ValueError, match="Output changed"):
        status(first)


@pytest.mark.parametrize("mode", ["raises", "no_outputs", "short_run"])
def test_failures_cannot_claim_success(workspace, mode):
    code, runtime = workspace
    fake_pipeline(
        code,
        "raise RuntimeError('intentional failure')" if mode == "raises" else "pass",
        '{"completed_steps": 3}' if mode == "short_run" else '{"completed_steps": 40}',
    )
    job = job_for(workspace)
    result = launch(job, check=False)
    assert result.returncode != 0
    run = runtime / "Runs/test"
    assert read_json(run / "JOB_FAILED.json")["status"] == "failed"
    assert not (run / "JOB_SUCCESS.json").exists()
    assert not (run / "TRAIN_COMPLETE.json").exists()
    assert status(job)["status"] == "failed"
    assert "Controller result" in (job / "TRAINING_RECORD.md").read_text()
    # Re-launching this job cannot overwrite the failed attempt.
    failure = (run / "JOB_FAILED.json").read_bytes()
    assert launch(job, check=False).returncode != 0
    assert (run / "JOB_FAILED.json").read_bytes() == failure


@pytest.mark.parametrize("sig", [signal.SIGTERM, signal.SIGKILL])
def test_interruption_is_not_success_and_preserves_run(workspace, sig):
    code, runtime = workspace
    fake_pipeline(code, "import time\n(run_dir / 'ready').write_text('ready')\ntime.sleep(45)")
    job = job_for(workspace)
    proc = subprocess.Popen(
        ["bash", str(job / "run.sh")],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env={**os.environ, "PYTHON_BIN": sys.executable},
    )
    try:
        deadline = time.monotonic() + 15
        while not (runtime / "Runs/test/ready").exists():
            assert proc.poll() is None and time.monotonic() < deadline
            time.sleep(0.05)
        proc.send_signal(sig)
        assert proc.wait(timeout=10) != 0
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()
    result = status(job)
    assert result["status"] == ("interrupted" if sig == signal.SIGTERM else "running_or_unfinished")
    assert not (runtime / "Runs/test/JOB_SUCCESS.json").exists()


def test_preflight_checks_bound_input_content(workspace, tmp_path):
    code, runtime = workspace
    from workflow.common import sha256

    data = tmp_path / "data.txt"
    data.write_text("training input")
    config = read_json(code / "configs/pipelines/demo.json")
    config["inputs"] = {"train": {"id": "dataset-v1/train", "sha256": sha256(data)}}
    atomic_json(code / "configs/pipelines/demo.json", config)
    fake_pipeline(code, "pass")
    bindings = tmp_path / "bindings.json"
    atomic_json(bindings, {"train": str(data)})
    release = create_release(code, runtime)
    job = prepare_job(release, runtime, "data", "demo", "plans/demo.md", bindings=bindings)
    launch(job, "--dry-run")
    data.write_text("changed input")
    rejected = launch(job, "--dry-run", check=False)
    assert rejected.returncode != 0 and "Input changed" in rejected.stderr
    assert not (runtime / "Runs").exists()

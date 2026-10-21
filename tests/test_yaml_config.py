"""Configuration integrity across review, freeze, and execution."""

import json
import shlex
import sys

import pytest
from test_workflow import launch
from test_workflow import workspace as make_workspace

from workflow.common import atomic_json, digest, read_json
from workflow.config import read_config
from workflow.jobs import check_job, prepare_job, verify_job
from workflow.release import create_release


@pytest.fixture
def workspace(tmp_path):
    return make_workspace.__wrapped__(tmp_path)


@pytest.mark.parametrize(
    "text",
    [
        "a: 1\na: 2\n",
        "a: &a [1]\nb: *a\n",
        "a: !!python/object/apply:os.system ['false']\n",
        "a: .nan\n",
        "a: 2026-10-21\n",
        "[1, 2]\n",
        "1: value\n",
    ],
)
def test_ambiguous_or_unsafe_yaml_is_rejected(tmp_path, text):
    path = tmp_path / "bad.yaml"
    path.write_text(text)
    with pytest.raises(ValueError):
        read_config(path)


def test_yaml_freeze_preserves_reviewed_comments_and_detects_edit(workspace):
    code, runtime = workspace
    config = code / "configs/pipelines/demo.yaml"
    original = b"# Reviewed experiment\r\n" + config.read_bytes().replace(b"\n", b"\r\n")
    config.write_bytes(original)
    release = create_release(code, runtime)
    job = prepare_job(release, runtime, "yaml", "demo", "plans/demo.md")
    assert (job / "config.yaml").read_bytes() == original
    assert not (job / "config.json").exists()
    assert verify_job(job)[1]["optimizer"]["learning_rate"] == 0.03
    config.write_text("schema: 999\n")
    launch(job, "--dry-run")
    target = job / "config.yaml"
    target.chmod(0o644)
    target.write_bytes(original + b"# Unreviewed change\r\n")
    with pytest.raises(ValueError, match="Job file changed"):
        verify_job(job)


def test_legacy_json_job_without_config_file_field_still_verifies(workspace):
    code, runtime = workspace
    config = {
        "schema": 1,
        "seed": 7,
        "max_steps": 2,
        "inputs": {},
        "expected_artifacts": ["metrics.json"],
    }
    atomic_json(code / "configs/pipelines/old.json", config)
    registry = read_json(code / "configs/registry.json")
    registry["pipelines"]["old"] = {
        "adapter": "dl_demo.pipeline.adapters.classification",
        "config": "configs/pipelines/old.json",
    }
    atomic_json(code / "configs/registry.json", registry)
    release = create_release(code, runtime)
    job = prepare_job(release, runtime, "old", "old", "plans/demo.md")
    spec = read_json(job / "JOB.json")
    for key in ["config_file", "execution", "definition_sha256"]:
        spec.pop(key)
    spec["definition_sha256"] = digest(spec)
    (job / "JOB.json").chmod(0o644)
    atomic_json(job / "JOB.json", spec)
    assert verify_job(job)[1] == config
    assert json.loads((job / "config.json").read_text()) == config


def test_runtime_is_frozen_separately_and_rejected_before_run(workspace, tmp_path):
    code, runtime = workspace
    placement = tmp_path / "runtime.yaml"
    placement.write_text("accelerator: cpu\ndevices: 2\n")
    release = create_release(code, runtime)
    job = prepare_job(release, runtime, "bad-resource", "demo", "plans/demo.md", runtime=placement)
    assert "devices" not in read_config(job / "config.yaml")
    assert read_json(job / "JOB.json")["execution"]["devices"] == 2
    result = launch(job, "--dry-run", check=False)
    assert result.returncode != 0 and "one CPU process" in result.stderr
    assert not (runtime / "Runs").exists()


def test_swanlab_offline_lightning_run(workspace, tmp_path):
    pytest.importorskip("swanlab")
    code, runtime = workspace
    placement = tmp_path / "runtime.yaml"
    placement.write_text("swanlab_mode: offline\n")
    release = create_release(code, runtime)
    job = prepare_job(release, runtime, "offline", "demo", "plans/demo.md", runtime=placement)
    launch(job, "--dry-run")
    assert not (runtime / "Runs").exists()
    launch(job)
    assert (runtime / "Runs/offline/JOB_SUCCESS.json").exists()
    assert any((runtime / "Runs/offline/log/swanlab").rglob("*.log"))


def test_check_uses_explicit_runtime_wrapper(workspace, tmp_path, monkeypatch):
    code, runtime = workspace
    adapter = code / "src/dl_demo/pipeline/adapters/classification.py"
    adapter.write_text(
        adapter.read_text().replace(
            "def preflight(config, bindings, resume_metadata):",
            "def preflight(config, bindings, resume_metadata):\n"
            "    import os\n    assert os.environ.get('RUNTIME_WRAPPER_TEST') == 'yes'",
        )
    )
    wrapper = tmp_path / "python wrapper"
    wrapper.write_text(
        f'#!/bin/sh\nexport RUNTIME_WRAPPER_TEST=yes\nexec {shlex.quote(sys.executable)} "$@"\n'
    )
    wrapper.chmod(0o755)
    monkeypatch.setenv("PYTHON_BIN", str(wrapper))
    release = create_release(code, runtime)
    job = prepare_job(release, runtime, "wrapper", "demo", "plans/demo.md")
    check_job(job)
    assert not (runtime / "Runs").exists()

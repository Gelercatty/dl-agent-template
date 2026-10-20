"""Freeze Git-visible source without requiring a commit per experiment."""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import tomllib
from pathlib import Path

from .common import atomic_json, digest, file_entry, now, read_json, regular_file, require, seal


def git(repo: Path, *args: str) -> bytes:
    return subprocess.check_output(["git", "-C", str(repo), *args], stderr=subprocess.PIPE)


def candidates(repo: Path, include: list[str]) -> list[str]:
    names = git(repo, "ls-files", "--cached", "--others", "--exclude-standard", "-z")
    return sorted(
        {
            os.fsdecode(name)
            for name in names.split(b"\0")
            if name
            and any(
                os.fsdecode(name) == p or os.fsdecode(name).startswith(p + "/") for p in include
            )
            and ((repo / os.fsdecode(name)).exists() or (repo / os.fsdecode(name)).is_symlink())
        }
    )


def allowed(path: str, size: int, limit: int) -> None:
    p = Path(path)
    require(
        not any(part in {".git", ".local", ".venv", "__pycache__", "scratch"} for part in p.parts),
        f"Private/runtime file selected: {path}",
    )
    require(
        not p.name.startswith(".env")
        and p.name not in {"credentials", "id_rsa", "id_ed25519"}
        and p.suffix.lower() not in {".pem", ".key", ".pt", ".pth", ".ckpt", ".safetensors"},
        f"Credential/weight file selected: {path}",
    )
    require(size <= limit, f"Source file exceeds {limit} bytes: {path}")


def verify_release(root: Path) -> dict:
    root = root.resolve()
    manifest = read_json(root / "RELEASE.json")
    require(manifest["schema"] == 1, "Unsupported release schema")
    identity = {k: manifest[k] for k in ("schema", "git", "files")}
    require(manifest["release_id"] == "src-" + digest(identity)[:20], "Release identity mismatch")
    require(not any(p.is_symlink() for p in root.rglob("*")), "Release contains a symlink")
    actual = {p.relative_to(root).as_posix() for p in root.rglob("*") if not p.is_dir()}
    require(actual == set(manifest["files"]) | {"RELEASE.json"}, "Release file set changed")
    for name, entry in manifest["files"].items():
        require(file_entry(regular_file(root, name)) == entry, f"Release file changed: {name}")
    return manifest


def create_release(repo: Path, workspace: Path) -> Path:
    repo, workspace = repo.resolve(), workspace.resolve()
    require(not workspace.is_relative_to(repo), "Runtime workspace must be outside source")
    require(
        Path(os.fsdecode(git(repo, "rev-parse", "--show-toplevel")).strip()).resolve() == repo,
        "Freeze the Git repository root",
    )
    settings = tomllib.loads((repo / "workflow.toml").read_text())["release"]
    include = settings["include"]
    require(
        isinstance(include, list) and all(isinstance(p, str) for p in include),
        "release.include must be a list of paths",
    )
    limit = settings["max_file_bytes"]
    status = git(repo, "status", "--porcelain=v1", "--untracked-files=all", "-z")
    try:
        commit = git(repo, "rev-parse", "HEAD").decode().strip()
    except subprocess.CalledProcessError:
        commit = None
    files = candidates(repo, include)
    require(files, "No source files selected")
    parent = workspace / "Releases"
    parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".freeze-", dir=parent))
    try:
        entries = {}
        for name in files:
            source = regular_file(repo, name)
            entry = file_entry(source)
            allowed(name, entry["bytes"], limit)
            target = staging / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            target.chmod(0o755 if entry["executable"] else 0o644)
            require(file_entry(target) == entry, f"Source changed during copy: {name}")
            entries[name] = entry
        # Rehash every source: Git's dirty flag alone cannot detect edits to an already dirty file.
        require(files == candidates(repo, include), "Source file set changed during freeze")
        for name, entry in entries.items():
            require(file_entry(regular_file(repo, name)) == entry, f"Source changed: {name}")
        require(
            status == git(repo, "status", "--porcelain=v1", "--untracked-files=all", "-z"),
            "Git state changed during freeze",
        )
        manifest = {"schema": 1, "git": {"commit": commit, "dirty": bool(status)}, "files": entries}
        manifest["release_id"] = "src-" + digest(manifest)[:20]
        manifest["created_at"] = now()
        atomic_json(staging / "RELEASE.json", manifest)
        verify_release(staging)
        destination = parent / manifest["release_id"]
        if destination.exists():
            verify_release(destination)
        else:
            staging.rename(destination)
            seal(destination)
        return destination
    finally:
        if staging.exists():
            shutil.rmtree(staging)

"""File contracts shared by the controller and pipeline adapters."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from datetime import UTC, datetime
from pathlib import Path


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def now() -> str:
    return datetime.now(UTC).isoformat()


def digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"Expected a JSON object: {path}")
    return value


def atomic_json(path: Path, value: object) -> None:
    atomic_text(path, json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n")


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def safe_relative(value: str) -> Path:
    path = Path(value)
    require(
        bool(value)
        and not path.is_absolute()
        and ".." not in path.parts
        and path.as_posix() == value
        and value != ".",
        f"Expected a normalized relative path: {value}",
    )
    return path


def regular_file(root: Path, relative: str) -> Path:
    path = root / safe_relative(relative)
    require(not path.is_symlink(), f"Symlink is not allowed: {path}")
    require(path.resolve().is_relative_to(root.resolve()), f"Path escapes its root: {path}")
    require(path.is_file(), f"Missing regular file: {path}")
    # Reject symlinked parent directories, even if their target is inside root.
    for parent in path.relative_to(root).parents:
        require(not (root / parent).is_symlink(), f"Symlink is not allowed: {root / parent}")
    return path


def label(value: str) -> str:
    require(bool(re.fullmatch(r"[a-z0-9][a-z0-9-]{0,63}", value)), "Use a short lowercase job name")
    return value


def file_entry(path: Path) -> dict:
    return {
        "sha256": sha256(path),
        "bytes": path.stat().st_size,
        "executable": bool(path.stat().st_mode & 0o111),
    }


def seal(root: Path) -> None:
    for path in root.rglob("*"):
        path.chmod(0o555 if path.is_dir() or path.stat().st_mode & 0o111 else 0o444)
    root.chmod(0o555)

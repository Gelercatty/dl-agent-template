"""python -m workflow --help"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from .jobs import check_job, execute, prepare_job, status
from .release import create_release, verify_release


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze source, prepare jobs, verify execution.")
    sub = parser.add_subparsers(dest="command", required=True)
    release = sub.add_parser("release", help="Capture selected Git-visible working-tree source")
    release.add_argument("--repo", type=Path, default=Path.cwd())
    release.add_argument(
        "--root", type=Path, required=True, help="Runtime workspace outside source"
    )
    verify = sub.add_parser("verify", help="Verify a frozen release")
    verify.add_argument("release", type=Path)
    prepare = sub.add_parser("prepare", help="Prepare a job; does not start training")
    prepare.add_argument("--release", type=Path, required=True)
    prepare.add_argument("--root", type=Path, required=True)
    prepare.add_argument("--name", required=True)
    prepare.add_argument("--pipeline", required=True)
    prepare.add_argument("--plan", required=True, help="Plan path inside the frozen source")
    prepare.add_argument(
        "--bindings", type=Path, help="Private JSON mapping input names to local files"
    )
    prepare.add_argument("--resume", type=Path, help="Checkpoint with .json sidecar")
    for name in ("check", "execute", "status"):
        cmd = sub.add_parser(name)
        cmd.add_argument("--job", type=Path, required=True)
        if name == "execute":
            cmd.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.command == "release":
        print(create_release(args.repo, args.root))
    elif args.command == "verify":
        print(json.dumps(verify_release(args.release), indent=2))
    elif args.command == "prepare":
        print(
            prepare_job(
                args.release,
                args.root,
                args.name,
                args.pipeline,
                args.plan,
                args.bindings,
                args.resume,
            )
        )
    elif args.command == "check":
        check_job(args.job.resolve())
    elif args.command == "execute":
        execute(args.job.resolve(), args.dry_run)
    else:
        print(json.dumps(status(args.job.resolve()), indent=2))


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, KeyError, ImportError, subprocess.CalledProcessError) as exc:
        print(f"workflow: {exc}", file=sys.stderr)
        sys.exit(1)

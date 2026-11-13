"""Command line entry points for running and verifying a laboratory workspace."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .errors import LabError
from .service import LaboratoryService


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m replay_lab",
                                     description="Offline agent incident replay and containment laboratory")
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run", help="run a complete experiment from local JSON")
    run.add_argument("--fixture", required=True, type=Path)
    run.add_argument("--workspace", required=True, type=Path)
    verify = subparsers.add_parser("verify", help="verify artifacts and SQLite records")
    verify.add_argument("--workspace", required=True, type=Path)
    inspect = subparsers.add_parser("inspect", help="print an existing summary")
    inspect.add_argument("--workspace", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "run":
            result = LaboratoryService(args.workspace).run_fixture(args.fixture)
        elif args.command == "verify":
            result = LaboratoryService(args.workspace).verify()
        else:
            summary = args.workspace / "summary.json"
            if not summary.is_file():
                raise LabError("summary file does not exist", details={"path": str(summary)})
            result = json.loads(summary.read_text(encoding="utf-8"))
        print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
        return 0 if result.get("ok", True) else 2
    except LabError as exc:
        print(json.dumps({"ok": False, "error": exc.as_dict()}, ensure_ascii=False, sort_keys=True, indent=2),
              file=sys.stderr)
        return 2
    except (OSError, ValueError, TypeError) as exc:
        print(json.dumps({"ok": False, "error": {"code": "runtime_error", "message": str(exc)}},
                         ensure_ascii=False, sort_keys=True, indent=2), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: summarize-output.py <directory>", file=sys.stderr)
        return 2
    root = Path(sys.argv[1])
    files = sorted(path for path in root.rglob("*") if path.is_file())
    print(f"Output directory: {root}")
    print(f"Files: {len(files)}")
    for path in files:
        print(f"- {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

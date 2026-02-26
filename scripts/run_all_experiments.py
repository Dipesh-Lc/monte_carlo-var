from __future__ import annotations

import argparse
import glob
import subprocess
import sys
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Run all experiments/*.yml configs via python -m src.run_from_yaml"
    )
    ap.add_argument(
        "--pattern",
        default="experiments/exp0*.yml",
        help='Glob pattern for YAML files (default: "experiments/exp0*.yml")',
    )
    ap.add_argument(
        "--stop_on_error",
        action="store_true",
        help="Stop immediately if any experiment fails",
    )
    ap.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable to use (default: current interpreter)",
    )
    args = ap.parse_args()

    files = sorted(glob.glob(args.pattern))
    if not files:
        raise FileNotFoundError(f"No experiment YAML files match: {args.pattern}")

    print("Found experiment configs:")
    for f in files:
        print(" -", f)

    failures: list[tuple[str, int]] = []

    for cfg in files:
        print("\n" + "=" * 90)
        print(f"RUNNING: {cfg}")
        print("=" * 90)

        cmd = [args.python, "-m", "src.run_from_yaml", cfg]
        proc = subprocess.run(cmd)

        if proc.returncode != 0:
            failures.append((cfg, proc.returncode))
            print(f"\n FAILED: {cfg} (exit code {proc.returncode})")
            if args.stop_on_error:
                sys.exit(proc.returncode)
        else:
            print(f"\n OK: {cfg}")

    print("\n" + "=" * 90)
    if not failures:
        print(" All experiments completed successfully.")
        sys.exit(0)

    print(f" Completed with {len(failures)} failure(s):")
    for cfg, code in failures:
        print(f" - {cfg} (exit code {code})")
    sys.exit(1)


if __name__ == "__main__":
    main()
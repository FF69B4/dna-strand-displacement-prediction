from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiment_core import ReproducibilityRunner

from dissertation.config import CONFIG


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="One-shot dissertation reproducibility runner.")
    parser.add_argument("--work-dir", type=Path, default=None)
    parser.add_argument("--keep-work-dir", action="store_true")
    parser.add_argument("--render-graphs", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = ReproducibilityRunner(CONFIG.reproducibility).run_cli(
        work_dir=args.work_dir,
        keep_work_dir=args.keep_work_dir,
        render_graphs=args.render_graphs,
        temp_prefix="diss_oneshot",
    )
    print(json.dumps(payload, indent=2))
    if not payload["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

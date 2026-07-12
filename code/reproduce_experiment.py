from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reproduce and verify the selected dissertation experiment."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/full_reproduction"),
    )
    parser.add_argument("--render-graphs", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.chdir(ROOT)

    from dissertation.config import CONFIG
    from experiment_core import ReproducibilityRunner

    output_dir = (ROOT / args.output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing reproduction directory: {output_dir}"
        )

    report = ReproducibilityRunner(CONFIG.reproducibility).run_once(
        output_dir,
        render_graphs=args.render_graphs,
    )
    report_path = output_dir / "reproduction_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Report saved to {display_path(report_path)}")
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

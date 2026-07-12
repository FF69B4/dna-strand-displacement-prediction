from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reproduce all reported metrics and figures."
    )
    parser.add_argument("--retrain-selected", action="store_true")
    return parser.parse_args()


def run(*arguments: str) -> None:
    subprocess.run([sys.executable, *arguments], cwd=ROOT, check=True)


def main() -> None:
    args = parse_args()
    run("verify_reported_results.py")
    run(
        "reproduce_reported_figures.py",
        "--output-dir",
        "outputs/reported_figures_complete",
    )
    run(
        "reproduce_predictions.py",
        "--output",
        "outputs/reported_table3_predictions.csv",
    )
    if args.retrain_selected:
        run(
            "reproduce_experiment.py",
            "--output-dir",
            "outputs/reported_selected_retraining",
            "--render-graphs",
        )

    report = {
        "passed": True,
        "reported_metrics_recomputed": True,
        "reported_figures_regenerated": True,
        "table3_ranking_recomputed": True,
        "selected_model_retrained": args.retrain_selected,
    }
    path = ROOT / "outputs/reproduce_all_reported_report.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

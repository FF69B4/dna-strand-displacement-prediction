from __future__ import annotations

import json
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent

TABLE1_COLUMNS = ["Model", "R2", "RMSE", "MAE"]

MODELS = [
    {
        "label": "Embeddings only",
        "summary": "training/artifacts/frozen_regression/k1_embeddings_only.json",
        "kind": "summary",
    },
    {
        "label": "Embeddings + NN features",
        "summary": "training/artifacts/frozen_regression/k1_regression_head.json",
        "kind": "summary",
    },
    {
        "label": "Embeddings + NN + availability",
        "summary": "training/artifacts/frozen_regression/k1_regression_head_with_availability.json",
        "kind": "summary",
    },
    {
        "label": "Selected hybrid model",
        "summary": "training/artifacts/frozen_regression/k1_regression_head_with_availability_tuned1.json",
        "kind": "summary",
    },
    {
        "label": "Unoptimised DNABERT-2",
        "summary": "training/artifacts/frozen_regression/dnabert2_optimized_final_summary.json",
        "kind": "dnabert2_unoptimised",
    },
    {
        "label": "Optimised DNABERT-2",
        "summary": "training/artifacts/frozen_regression/dnabert2_optimized_final_summary.json",
        "kind": "dnabert2_optimised",
    },
]


def round_half_up(value: float, places: str = "0.001") -> float:
    return float(Decimal(str(value)).quantize(Decimal(places), rounding=ROUND_HALF_UP))


def raw_test_metrics(config: dict[str, str]) -> dict[str, float]:
    summary = json.loads((ROOT / config["summary"]).read_text(encoding="utf-8"))
    if config["kind"] == "summary":
        return summary["metrics"]["test"]["raw"]
    if config["kind"] == "dnabert2_unoptimised":
        return {
            "r2": summary["baseline"]["test_r2"],
            "rmse": summary["baseline"]["test_rmse"],
            "mae": summary["baseline"]["test_mae"],
        }
    if config["kind"] == "dnabert2_optimised":
        return {
            "r2": summary["best_optimized"]["test_r2"],
            "rmse": summary["best_optimized"]["test_rmse"],
            "mae": summary["best_optimized"]["test_mae"],
        }
    raise ValueError(f"Unknown model summary kind: {config['kind']}")


def table1_row(config: dict[str, str]) -> dict[str, object]:
    metrics = raw_test_metrics(config)
    return {
        "Model": config["label"],
        "R2": round_half_up(float(metrics["r2"])),
        "RMSE": round_half_up(float(metrics["rmse"])),
        "MAE": round_half_up(float(metrics["mae"])),
    }


def main() -> None:
    output_dir = ROOT / "outputs" / "reported_results"
    output_dir.mkdir(parents=True, exist_ok=True)

    reproduced = pd.DataFrame([table1_row(config) for config in MODELS])
    reproduced = reproduced.loc[:, TABLE1_COLUMNS]
    canonical_path = (
        ROOT
        / "training/artifacts/model_comparison/reported_table1_model_comparison.csv"
    )
    canonical = pd.read_csv(canonical_path).loc[:, TABLE1_COLUMNS]

    try:
        pd.testing.assert_frame_equal(
            reproduced,
            canonical,
            check_dtype=False,
            check_exact=True,
        )
        table1_matches = True
    except AssertionError:
        table1_matches = False

    reproduced_path = output_dir / "reported_table1_model_comparison_recomputed.csv"
    reproduced.to_csv(reproduced_path, index=False)

    explicit_results = {
        "selected_6mer_test_r2": float(
            reproduced.loc[
                reproduced["Model"] == "Selected hybrid model",
                "R2",
            ].iloc[0]
        ),
        "optimized_dnabert2_test_r2": float(
            reproduced.loc[
                reproduced["Model"] == "Optimised DNABERT-2",
                "R2",
            ].iloc[0]
        ),
    }
    explicit_results_match_paper = (
        explicit_results["selected_6mer_test_r2"] == 0.702
        and explicit_results["optimized_dnabert2_test_r2"] == 0.687
    )

    report = {
        "passed": table1_matches and explicit_results_match_paper,
        "reported_table1_matches": table1_matches,
        "comparison_rows": len(reproduced),
        "explicit_results_match_paper": explicit_results_match_paper,
        "explicit_results": explicit_results,
        "recomputed_comparison": str(reproduced_path.relative_to(ROOT)),
    }
    report_path = output_dir / "reported_results_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

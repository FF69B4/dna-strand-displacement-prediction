from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Recompute and verify the final Table 3 ranking.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/table_3_predictions_recomputed.csv"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.chdir(ROOT)

    from dissertation.config import CONFIG
    from ml_core import ModelArtifact, PredictionService

    output_path = (ROOT / args.output).resolve()
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    result = PredictionService(
        ModelArtifact.load(CONFIG.reproducibility.canonical_model_artifact),
        schema=CONFIG.app.schema,
    ).predict_table(
        table_path=CONFIG.reproducibility.prediction_table,
        embedding_path=CONFIG.reproducibility.prediction_embeddings,
        batch_size=256,
    )

    generated = result.frame.sort_values("predicted_rank").reset_index(drop=True)
    generated["No."] = generated["No."].astype(str)
    generated["predicted_k1"] = generated["predicted_k1"].astype(int)
    canonical = pd.read_csv(CONFIG.reproducibility.canonical_ranked_predictions)
    canonical = canonical.sort_values("predicted_rank").reset_index(drop=True)
    canonical["No."] = canonical["No."].astype(str)

    ids_match = generated["No."].tolist() == canonical["No."].tolist()
    ranks_match = bool(
        np.array_equal(
            generated["predicted_rank"].to_numpy(),
            canonical["predicted_rank"].to_numpy(),
        )
    )
    generated.to_csv(output_path, index=False)

    report = {
        "passed": ids_match and ranks_match,
        "rows": len(generated),
        "ordered_ids_match": ids_match,
        "rank_exact_match": ranks_match,
        "top_id": generated.iloc[0]["No."],
        "output": display_path(output_path),
    }
    report_path = output_path.with_suffix(".json")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

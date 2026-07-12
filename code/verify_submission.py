from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


REQUIRED_PATHS = (
    "config/selected_experiment.json",
    "data/extracted/clean/table_1.csv",
    "data/extracted/clean/table_3.csv",
    "data/extracted/clean/table_4.csv",
    "bert/table_1_embeddings.npz",
    "bert/table_3_embeddings.npz",
    "training/artifacts/frozen_regression/k1_regression_head_with_availability_tuned1.pt",
    "training/artifacts/frozen_regression/k1_regression_head_with_availability_tuned1.json",
    "training/artifacts/final_predictions/table_3_final_predictions_ranked.csv",
    "training/artifacts/akay_comparisons/akay_comparison_with_availability_tuned1.json",
)


def main() -> None:
    missing = [relative for relative in REQUIRED_PATHS if not (ROOT / relative).exists()]
    if missing:
        print("Submission verification failed. Missing files:")
        for relative in missing:
            print(f"  - {relative}")
        raise SystemExit(1)

    from ml_core import ModelArtifact

    selected = json.loads((ROOT / "config/selected_experiment.json").read_text(encoding="utf-8"))
    artifact = ModelArtifact.load(ROOT / selected["canonical_artifact"])
    model = artifact.build_model()

    print("Submission verification passed.")
    print(f"Experiment: {selected['name']}")
    print(f"Project root: {ROOT}")
    print(f"Target: {artifact.target_column} ({artifact.target_transform})")
    print(f"Feature count: {len(artifact.feature_recipe.feature_names)}")
    print(f"Model layers: {[layer.__class__.__name__ for layer in model.network]}")
    print(f"Canonical artifact: {artifact.path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

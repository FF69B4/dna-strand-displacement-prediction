from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot Akay rank agreement for the frozen, fine-tuned, and blended models."
    )
    parser.add_argument(
        "--comparison-csv",
        type=Path,
        default=Path("training/artifacts/akay_comparisons/akay_comparison.csv"),
        help="CSV produced by the dissertation Akay comparison step.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("training/artifacts/akay_comparisons/akay_rank_comparison.png"),
        help="Destination image path.",
    )
    parser.add_argument(
        "--blend-weight",
        type=float,
        default=0.6,
        help=(
            "Weight for the frozen model in the blended score. "
            "The fine-tuned model gets (1 - blend_weight)."
        ),
    )
    return parser.parse_args()


def rank_descending(values: pd.Series | np.ndarray) -> np.ndarray:
    return pd.Series(values).rank(method="min", ascending=False).to_numpy(dtype=int)


def top_overlap(frame: pd.DataFrame, score_col: str, k: int) -> float:
    akay_top = set(frame.nlargest(k, "Predicted k1")["No."])
    model_top = set(frame.nlargest(k, score_col)["No."])
    return len(akay_top & model_top) / k


def main() -> None:
    args = parse_args()
    frame = pd.read_csv(args.comparison_csv)

    frozen_col = "k1_regression_head_predicted_k1"
    finetuned_col = "k1_finetuned_dnabert_tuned3_predicted_k1"
    if frozen_col not in frame.columns or finetuned_col not in frame.columns:
        raise ValueError(
            f"Expected columns {frozen_col!r} and {finetuned_col!r} in {args.comparison_csv}"
        )

    frame = frame.copy()
    frame["hybrid_predicted_k1"] = (
        args.blend_weight * frame[frozen_col]
        + (1.0 - args.blend_weight) * frame[finetuned_col]
    )
    frame["akay_rank"] = rank_descending(frame["Predicted k1"])
    frame["frozen_rank"] = rank_descending(frame[frozen_col])
    frame["finetuned_rank"] = rank_descending(frame[finetuned_col])
    frame["hybrid_rank"] = rank_descending(frame["hybrid_predicted_k1"])

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax = axes[0]
    ax.scatter(frame["akay_rank"], frame["frozen_rank"], s=10, alpha=0.35, label="Frozen")
    ax.scatter(frame["akay_rank"], frame["finetuned_rank"], s=10, alpha=0.35, label="Fine-tuned")
    ax.scatter(frame["akay_rank"], frame["hybrid_rank"], s=10, alpha=0.35, label="Hybrid")
    diagonal = np.arange(1, len(frame) + 1)
    ax.plot(diagonal, diagonal, color="black", linewidth=1, linestyle="--", label="Perfect match")
    ax.set_xlabel("Akay Rank")
    ax.set_ylabel("Model Rank")
    ax.legend(frameon=False)

    ks = [10, 20, 50, 100, 200, 300]
    overlap_frame = pd.DataFrame(
        {
            "k": ks,
            "Frozen": [top_overlap(frame, frozen_col, k) for k in ks],
            "Fine-tuned": [top_overlap(frame, finetuned_col, k) for k in ks],
            "Hybrid": [top_overlap(frame, "hybrid_predicted_k1", k) for k in ks],
        }
    )
    ax = axes[1]
    for label in ["Frozen", "Fine-tuned", "Hybrid"]:
        ax.plot(overlap_frame["k"], overlap_frame[label], marker="o", label=label)
    ax.set_xlabel("k")
    ax.set_ylabel("Overlap Fraction")
    ax.set_ylim(0, 1)
    ax.legend(frameon=False)
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved plot to {args.output}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import csv
import os
from collections import Counter, defaultdict
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "data" / "extracted" / "clean" / "table_3.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "training" / "artifacts" / "embedding_context_experiment"

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman", "CMU Serif", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        "axes.unicode_minus": False,
    }
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Scratch experiment: extract real DNA-BERT token-level hidden states for repeated 6-mers "
            "and plot their context-dependent positions in 3D PCA space. Writes only to the output folder."
        )
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model-name", default="zhihan1996/DNA_bert_6")
    parser.add_argument("--target-kmer", default=None, help="Optional 6-mer to inspect. Defaults to a frequent 6-mer.")
    parser.add_argument("--max-occurrences", type=int, default=120)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    return parser.parse_args()


def normalize_sequence(sequence: str) -> str:
    return "".join(str(sequence).upper().split())


def kmers(sequence: str, k: int = 6) -> list[str]:
    return [sequence[index : index + k] for index in range(len(sequence) - k + 1)]


def choose_target_kmer(sequences: list[str], requested: str | None) -> str:
    if requested:
        requested = normalize_sequence(requested)
        if len(requested) != 6:
            raise ValueError("--target-kmer must be exactly 6 bases")
        return requested

    counts: Counter[str] = Counter()
    sequence_counts: defaultdict[str, int] = defaultdict(int)
    for sequence in sequences:
        seen = set(kmers(sequence))
        counts.update(kmers(sequence))
        for kmer in seen:
            sequence_counts[kmer] += 1

    candidates = [
        (kmer, count, sequence_counts[kmer])
        for kmer, count in counts.items()
        if sequence_counts[kmer] >= 8 and "N" not in kmer
    ]
    if not candidates:
        raise ValueError("No repeated 6-mer was found often enough for a useful context plot.")
    return max(candidates, key=lambda item: (item[2], item[1]))[0]


def collect_occurrences(frame: pd.DataFrame, target_kmer: str, max_occurrences: int) -> list[dict[str, object]]:
    occurrences: list[dict[str, object]] = []
    for _, row in frame.iterrows():
        record_id = str(row["No."])
        sequence = normalize_sequence(row["Sequence"])
        sequence_kmers = kmers(sequence)
        for position, kmer in enumerate(sequence_kmers):
            if kmer != target_kmer:
                continue
            left_base = sequence[position - 1] if position > 0 else "^"
            right_index = position + len(target_kmer)
            right_base = sequence[right_index] if right_index < len(sequence) else "$"
            occurrences.append(
                {
                    "record_id": record_id,
                    "sequence": sequence,
                    "kmer": kmer,
                    "token_position": position,
                    "left_base": left_base,
                    "right_base": right_base,
                    "context_label": f"{left_base}_{right_base}",
                    "kmer_sequence": " ".join(sequence_kmers),
                }
            )
            if len(occurrences) >= max_occurrences:
                return occurrences
    return occurrences


def extract_token_vectors(
    occurrences: list[dict[str, object]],
    model_name: str,
    device: str,
) -> np.ndarray:
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    model = AutoModel.from_pretrained(model_name, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.sep_token or tokenizer.unk_token
    model.to(device)
    model.eval()

    vectors: list[np.ndarray] = []
    for occurrence in occurrences:
        encoded = tokenizer(
            str(occurrence["kmer_sequence"]),
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        token_index = int(occurrence["token_position"]) + 1
        import torch

        with torch.inference_mode():
            output = model(**encoded)
        vectors.append(output.last_hidden_state[0, token_index, :].detach().cpu().numpy())
    return np.vstack(vectors).astype(np.float32)


def project_pca_3d(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    u, singular_values, _ = np.linalg.svd(centered, full_matrices=False)
    coordinates = u[:, :3] * singular_values[:3]
    explained_variance = (singular_values**2) / max(matrix.shape[0] - 1, 1)
    explained_ratio = explained_variance / explained_variance.sum()
    return coordinates, explained_ratio


def style_3d_axes(ax) -> None:
    ax.set_xlabel("PC1", labelpad=5)
    ax.set_ylabel("PC2", labelpad=5)
    ax.set_zlabel("PC3", labelpad=5)
    ax.view_init(elev=23, azim=-47)
    ax.grid(True, color="0.82", linewidth=0.6)
    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        axis.pane.set_facecolor((1.0, 1.0, 1.0, 0.0))
        axis.pane.set_edgecolor("0.75")


def draw_boxed_strand(ax, sequence: str) -> None:
    ax.axis("off")
    ax.set_xlim(0, len(sequence))
    ax.set_ylim(0, 1)
    y = 0.22
    height = 0.58
    rect = plt.Rectangle(
        (0, y),
        len(sequence),
        height,
        facecolor="white",
        edgecolor="black",
        linewidth=0.9,
    )
    ax.add_patch(rect)
    for index, base in enumerate(sequence):
        if index > 0:
            ax.plot([index, index], [y, y + height], color="black", linewidth=0.75)
        ax.text(index + 0.5, y + height / 2, base, ha="center", va="center", fontsize=8)


def plot_context_pca(output_dir: Path, table: pd.DataFrame, variance: np.ndarray, target_kmer: str) -> None:
    context_counts = table["context_label"].value_counts()
    common_contexts = context_counts.head(5).index.tolist()
    markers = ["o", "s", "^", "D", "*"]

    fig = plt.figure(figsize=(5.6, 4.7))
    ax = fig.add_axes([0.02, 0.04, 0.96, 0.92], projection="3d")
    for marker, context in zip(markers, common_contexts, strict=False):
        group = table.loc[table["context_label"] == context]
        ax.scatter(
            group["pc1"],
            group["pc2"],
            group["pc3"],
            marker=marker,
            s=38 if marker != "*" else 62,
            facecolor="white",
            edgecolor="black",
            linewidth=0.55,
            alpha=0.78,
            label=f"{context} context",
        )

    other = table.loc[~table["context_label"].isin(common_contexts)]
    if not other.empty:
        ax.scatter(
            other["pc1"],
            other["pc2"],
            other["pc3"],
            marker="x",
            s=24,
            color="0.35",
            linewidth=0.55,
            alpha=0.55,
            label="other contexts",
        )

    style_3d_axes(ax)
    ax.legend(frameon=False, fontsize=7, loc="upper left")
    fig.savefig(output_dir / "same_6mer_token_context_pca_3d.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_readme(output_dir: Path, target_kmer: str, occurrence_count: int, variance: np.ndarray) -> None:
    (output_dir / "README.md").write_text(
        "\n".join(
            [
                "Scratch token-level DNA-BERT context experiment.",
                "",
                "This folder is intentionally separate from the main paper graph/data artifacts.",
                "It reads existing table 3 sequences and a locally cached DNA-BERT model, then writes experiment-only outputs here.",
                "",
                f"- Target 6-mer: `{target_kmer}`",
                f"- Token occurrences plotted: `{occurrence_count}`",
                f"- PCA variance in first three components: `{variance[:3].sum():.4f}`",
                "- `same_6mer_token_context_pca_3d.png`: 3D PCA plot of token hidden states grouped by immediate left/right sequence context.",
                "- `same_6mer_token_context_vectors.csv`: extracted token metadata and PCA coordinates.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    frame = pd.read_csv(args.input)
    frame["Sequence"] = frame["Sequence"].map(normalize_sequence)
    target_kmer = choose_target_kmer(frame["Sequence"].tolist(), args.target_kmer)
    occurrences = collect_occurrences(frame, target_kmer, args.max_occurrences)
    if len(occurrences) < 3:
        raise ValueError(f"Only found {len(occurrences)} occurrences of {target_kmer}; need at least 3 for PCA.")

    vectors = extract_token_vectors(occurrences, args.model_name, args.device)
    coordinates, variance = project_pca_3d(vectors)

    output_table = pd.DataFrame(occurrences)
    output_table["pc1"] = coordinates[:, 0]
    output_table["pc2"] = coordinates[:, 1]
    output_table["pc3"] = coordinates[:, 2]
    output_table = output_table.drop(columns=["kmer_sequence"])
    output_table.to_csv(output_dir / "same_6mer_token_context_vectors.csv", index=False, quoting=csv.QUOTE_MINIMAL)

    plot_context_pca(output_dir, output_table, variance, target_kmer)
    write_readme(output_dir, target_kmer, len(output_table), variance)
    print(f"Wrote token-context experiment outputs to {output_dir}")


if __name__ == "__main__":
    main()

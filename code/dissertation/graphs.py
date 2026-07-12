from __future__ import annotations

import ast
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from viz_core import FigureOutput, monochrome_serif_theme

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = PROJECT_ROOT / "training" / "artifacts"
COMPARISONS = ARTIFACTS / "akay_comparisons"
FINAL_PREDICTIONS = ARTIFACTS / "final_predictions"
FROZEN = ARTIFACTS / "frozen_regression"
MODEL_COMPARISON = ARTIFACTS / "model_comparison"
OUTPUT = ARTIFACTS / "paper_graphs"
EMBEDDINGS_ONLY_SUMMARY = FROZEN / "k1_embeddings_only.json"
EMBEDDINGS_ONLY_PREDICTIONS = FROZEN / "k1_embeddings_only_predictions.csv"
THEME = monochrome_serif_theme()
THEME.apply()
FIGURES = FigureOutput(OUTPUT, THEME)
BAR_HATCHES = THEME.bar_hatches
LINE_STYLES = THEME.line_styles
MARKERS = THEME.markers


MODEL_LABELS = {
    "Frozen embeddings+NN+availability tuned": "Frozen emb. + NN + availability\n(tuned)",
    "Frozen embeddings+NN+availability domain-weighted": "Domain-weighted\navailability model",
    "Frozen embeddings+NN+availability": "Frozen emb. + NN + availability",
    "Frozen embeddings+NN": "Frozen emb. + NN",
    "Fine-tuned bert+NN+availability retry": "Fine-tuned BERT + NN + availability",
    "Fine-tuned bert+NN best": "Fine-tuned BERT + NN",
}


def style_axes(ax):
    THEME.style_axes(ax)


def save(fig, filename: str) -> None:
    FIGURES.save(fig, filename)


def project_pca_3d(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    u, singular_values, _ = np.linalg.svd(centered, full_matrices=False)
    coordinates = u[:, :3] * singular_values[:3]
    explained_variance = (singular_values**2) / max(matrix.shape[0] - 1, 1)
    explained_ratio = explained_variance / explained_variance.sum()
    return coordinates, explained_ratio


def apply_bar_patterns(bars, hatches: list[str] | None = None) -> None:
    THEME.apply_bar_patterns(bars, hatches)


def plot_model_r2(comparison: pd.DataFrame) -> None:
    selected = comparison[
        comparison["Model"].isin(
            [
                "Frozen embeddings+NN",
                "Frozen embeddings+NN+availability",
                "Frozen embeddings+NN+availability tuned",
                "Fine-tuned bert+NN best",
                "Fine-tuned bert+NN+availability retry",
            ]
        )
    ].copy()
    selected["label"] = selected["Model"].map(MODEL_LABELS)
    selected = selected.sort_values("Measured table_1 test R2")

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    bars = ax.barh(selected["label"], selected["Measured table_1 test R2"])
    apply_bar_patterns(bars)
    ax.set_xlabel("Measured test R$^2$")
    ax.set_xlim(0, 0.8)
    style_axes(ax)
    for bar in bars:
        width = bar.get_width()
        ax.text(width + 0.015, bar.get_y() + bar.get_height() / 2, f"{width:.3f}", va="center")
    save(fig, "model_test_r2_comparison.png")


def plot_external_metrics(comparison: pd.DataFrame) -> None:
    selected = comparison[
        comparison["Model"].isin(
            [
                "Frozen embeddings+NN",
                "Frozen embeddings+NN+availability",
                "Frozen embeddings+NN+availability tuned",
                "Frozen embeddings+NN+availability domain-weighted",
            ]
        )
    ].copy()
    selected["label"] = selected["Model"].map(MODEL_LABELS)

    metrics = [
        ("Akay spearman_log10", "Spearman $\\rho$"),
        ("Akay within 2x", "Within 2x"),
        ("Akay top-100 overlap", "Top-100 overlap"),
    ]
    x = np.arange(len(metrics))
    width = 0.19
    fig, ax = plt.subplots(figsize=(11.2, 5.0))
    for offset, (_, row) in enumerate(selected.iterrows()):
        values = [row[column] for column, _ in metrics]
        bars = ax.bar(
            x + (offset - 1.5) * width,
            values,
            width=width,
            label=row["label"],
        )
        apply_bar_patterns(bars, [BAR_HATCHES[offset]])

    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label in metrics])
    ax.set_ylim(0, 0.82)
    ax.set_ylabel("Score")
    ax.legend(
        fontsize=8,
        frameon=False,
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        borderaxespad=0.0,
    )
    style_axes(ax)
    save(fig, "external_ranking_metrics.png")


def plot_feature_composition() -> None:
    labels = [
        "DNA-BERT\nembedding",
        "Positional\nNN",
        "Frequency\nNN",
        "Availability",
    ]
    values = np.array([768, 464, 16, 30])
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    bars = ax.bar(labels, values)
    apply_bar_patterns(bars)
    ax.set_ylabel("Feature dimensions")
    style_axes(ax)
    for bar, value in zip(bars, values, strict=True):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 12, str(value), ha="center")
    ax.text(0.5, 0.91, "768 + 464 + 16 + 30 = 1278", transform=ax.transAxes, ha="center")
    save(fig, "feature_concatenation_dimensions.png")


def plot_rank_agreement() -> None:
    frame = pd.read_csv(COMPARISONS / "akay_comparison_with_availability_tuned1.csv")
    x = frame["akay_rank"].to_numpy()
    y = frame["k1_regression_head_with_availability_tuned1_rank"].to_numpy()

    fig, ax = plt.subplots(figsize=(6.0, 5.5))
    ax.scatter(x, y, s=14, alpha=0.45, facecolor="white", edgecolor="black", linewidth=0.45)
    limit = max(x.max(), y.max())
    ax.plot([1, limit], [1, limit], color="black", linewidth=1.2, linestyle="--")
    ax.invert_xaxis()
    ax.invert_yaxis()
    ax.set_xlabel("Reference rank (Akay)")
    ax.set_ylabel("Model rank")
    ax.text(0.03, 0.95, "Spearman $\\rho$ = 0.745", transform=ax.transAxes, va="top")
    style_axes(ax)
    save(fig, "akay_model_rank_agreement.png")


def plot_top_candidates() -> None:
    frame = pd.read_csv(FINAL_PREDICTIONS / "table_3_final_predictions_ranked.csv").head(10).copy()
    frame["label"] = frame["No."].astype(str)
    frame = frame.iloc[::-1]

    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    bars = ax.barh(frame["label"], frame["predicted_k1"] / 1_000_000)
    apply_bar_patterns(bars)
    ax.set_xlabel("Predicted k1 (millions)")
    ax.set_ylabel("Candidate ID")
    style_axes(ax)
    for bar, value in zip(bars, frame["predicted_k1"] / 1_000_000, strict=True):
        ax.text(value + 0.05, bar.get_y() + bar.get_height() / 2, f"{value:.2f}", va="center", fontsize=8)
    save(fig, "top10_predicted_table3_candidates.png")


def plot_availability_heatmap() -> None:
    frame = pd.read_csv(FINAL_PREDICTIONS / "table_3_final_predictions_ranked.csv").head(20).copy()
    matrix = np.vstack([np.asarray(ast.literal_eval(value), dtype=float) for value in frame["Nucleotide Availability"]])

    fig, ax = plt.subplots(figsize=(10.2, 5.4))
    x_edges = np.arange(matrix.shape[1] + 1)
    y_edges = np.arange(matrix.shape[0] + 1)
    ax.pcolormesh(
        x_edges,
        y_edges,
        np.zeros_like(matrix),
        facecolor="none",
        edgecolors="black",
        linewidth=0.18,
    )

    bins = np.digitize(matrix, [0.25, 0.5, 0.75], right=False)
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            level = bins[row, col]
            x = col + 0.5
            y = row + 0.5
            if level == 1:
                ax.plot(x, y, marker=".", color="black", markersize=2.4, alpha=0.45, linestyle="None")
            elif level == 2:
                ax.plot([x - 0.28, x + 0.28], [y + 0.28, y - 0.28], color="black", alpha=0.45, linewidth=0.65)
            elif level == 3:
                ax.plot([x - 0.28, x + 0.28], [y + 0.28, y - 0.28], color="black", alpha=0.45, linewidth=0.65)
                ax.plot([x - 0.28, x + 0.28], [y - 0.28, y + 0.28], color="black", alpha=0.45, linewidth=0.65)

    ax.set_xlabel("Nucleotide position")
    ax.set_ylabel("Top-ranked candidate")
    ax.set_xlim(0, matrix.shape[1])
    ax.set_ylim(matrix.shape[0], 0)
    ax.set_xticks(np.arange(0.5, matrix.shape[1], 5))
    ax.set_xticklabels(np.arange(1, matrix.shape[1] + 1, 5))
    ax.set_yticks(np.arange(0.5, matrix.shape[0], 1))
    ax.set_yticklabels(frame["No."].astype(str), fontsize=7)

    blank = plt.Line2D([], [], color="black", marker="s", markersize=6, markerfacecolor="white", linestyle="None")
    dot = plt.Line2D([], [], color="black", marker=".", markersize=7, linestyle="None")
    slash = plt.Line2D([], [], color="black", linewidth=1.0)
    cross = plt.Line2D([], [], color="black", marker="x", markersize=6, linestyle="None")
    hatch_labels = ["0.00-0.25", "0.25-0.50", "0.50-0.75", "0.75-1.00"]
    handles = [
        blank,
        dot,
        slash,
        cross,
    ]
    ax.legend(
        handles,
        hatch_labels,
        title="Availability",
        frameon=False,
        fontsize=8,
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
    )
    save(fig, "top20_availability_heatmap.png")


def plot_predicted_vs_actual() -> None:
    frame = pd.read_csv(FROZEN / "k1_regression_head_with_availability_tuned1_predictions.csv")
    test = frame.loc[frame["split"] == "test"].copy()

    x = np.log10(test["k1_actual"].to_numpy(dtype=float))
    y = np.log10(test["k1_predicted"].to_numpy(dtype=float))
    lower = min(x.min(), y.min())
    upper = max(x.max(), y.max())

    fig, ax = plt.subplots(figsize=(6.2, 5.6))
    ax.scatter(x, y, s=18, alpha=0.5, facecolor="white", edgecolor="black", linewidth=0.45)
    ax.plot([lower, upper], [lower, upper], color="black", linestyle="--", linewidth=1.2)
    ax.set_xlabel("Actual log$_{10}$(k$_1$)")
    ax.set_ylabel("Predicted log$_{10}$(k$_1$)")
    ax.text(0.04, 0.95, "Test R$^2$ = 0.702", transform=ax.transAxes, va="top")
    style_axes(ax)
    save(fig, "predicted_vs_actual_log10_test.png")


def plot_residuals() -> None:
    frame = pd.read_csv(FROZEN / "k1_regression_head_with_availability_tuned1_predictions.csv")
    test = frame.loc[frame["split"] == "test"].copy()
    predicted = np.log10(test["k1_predicted"].to_numpy(dtype=float))
    residual = test["k1_model_predicted"].to_numpy(dtype=float) - test["k1_model_actual"].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.scatter(predicted, residual, s=16, alpha=0.5, facecolor="white", edgecolor="black", linewidth=0.45)
    ax.axhline(0, color="black", linestyle="--", linewidth=1.1)
    ax.set_xlabel("Predicted log$_{10}$(k$_1$)")
    ax.set_ylabel("Residual on log10 scale")
    style_axes(ax)
    save(fig, "residuals_vs_predicted_log10_test.png")


def load_embeddings_only_summary() -> dict:
    return json.loads(EMBEDDINGS_ONLY_SUMMARY.read_text(encoding="utf-8"))


def load_embeddings_only_predictions() -> pd.DataFrame:
    frame = pd.read_csv(EMBEDDINGS_ONLY_PREDICTIONS)
    required_columns = {
        "split",
        "k1_actual",
        "k1_predicted",
        "k1_model_actual",
        "k1_model_predicted",
    }
    missing = required_columns - set(frame.columns)
    if missing:
        raise ValueError(f"Missing columns in {EMBEDDINGS_ONLY_PREDICTIONS}: {sorted(missing)}")
    return frame


def plot_embeddings_only_predicted_vs_actual() -> None:
    summary = load_embeddings_only_summary()
    frame = load_embeddings_only_predictions()
    test = frame.loc[frame["split"] == "test"].copy()
    x = test["k1_model_actual"].to_numpy(dtype=float)
    y = test["k1_model_predicted"].to_numpy(dtype=float)
    lower = min(x.min(), y.min())
    upper = max(x.max(), y.max())
    metrics = summary["metrics"]["test"]["raw"]

    fig, ax = plt.subplots(figsize=(6.2, 5.6))
    ax.scatter(x, y, s=18, alpha=0.5, facecolor="white", edgecolor="black", linewidth=0.45)
    ax.plot([lower, upper], [lower, upper], color="black", linestyle="--", linewidth=1.2)
    ax.set_xlabel(r"Actual $\log_{10}(k_1)$")
    ax.set_ylabel(r"Predicted $\log_{10}(k_1)$")
    ax.text(
        0.04,
        0.95,
        "\n".join(
            [
                rf"Test $R^2$ = {metrics['r2']:.3f}",
                f"RMSE = {metrics['rmse']:,.0f}",
                f"MAE = {metrics['mae']:,.0f}",
            ]
        ),
        transform=ax.transAxes,
        va="top",
    )
    style_axes(ax)
    save(fig, "embeddings_only_predicted_vs_actual_log10_test.png")


def plot_embeddings_only_residuals() -> None:
    frame = load_embeddings_only_predictions()
    test = frame.loc[frame["split"] == "test"].copy()
    predicted = test["k1_model_predicted"].to_numpy(dtype=float)
    residual = predicted - test["k1_model_actual"].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.scatter(predicted, residual, s=16, alpha=0.5, facecolor="white", edgecolor="black", linewidth=0.45)
    ax.axhline(0, color="black", linestyle="--", linewidth=1.1)
    ax.set_xlabel(r"Predicted $\log_{10}(k_1)$")
    ax.set_ylabel(r"Residual on $\log_{10}$ scale")
    style_axes(ax)
    save(fig, "embeddings_only_residuals_log10_test.png")


def plot_embeddings_only_split_r2() -> None:
    summary = load_embeddings_only_summary()
    splits = ["train", "val", "test"]
    raw_scores = [summary["metrics"][split]["raw"]["r2"] for split in splits]
    model_scale_scores = [summary["metrics"][split]["model_scale"]["r2"] for split in splits]
    x = np.arange(len(splits))
    width = 0.34

    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    raw_bars = ax.bar(x - width / 2, raw_scores, width=width, label=r"Raw $k_1$")
    transformed_bars = ax.bar(x + width / 2, model_scale_scores, width=width, label=r"$\log_{10}(k_1)$")
    apply_bar_patterns(raw_bars)
    apply_bar_patterns(transformed_bars, ["///"])
    ax.set_xticks(x)
    ax.set_xticklabels(["Train", "Validation", "Test"])
    ax.set_ylabel(r"$R^2$")
    ax.set_ylim(0, 1.0)
    ax.legend(frameon=False)
    style_axes(ax)
    for bars in [raw_bars, transformed_bars]:
        for bar in bars:
            value = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, value + 0.018, f"{value:.3f}", ha="center", fontsize=8)
    save(fig, "embeddings_only_split_r2.png")


def plot_embeddings_only_ablation_context(comparison: pd.DataFrame) -> None:
    summary = load_embeddings_only_summary()
    scores = {
        "Embeddings\nonly": summary["metrics"]["test"]["raw"]["r2"],
        "Emb. + NN": float(
            comparison.loc[
                comparison["Model"] == "Frozen embeddings+NN",
                "Measured table_1 test R2",
            ].iloc[0]
        ),
        "Emb. + NN\n+ availability": float(
            comparison.loc[
                comparison["Model"] == "Frozen embeddings+NN+availability",
                "Measured table_1 test R2",
            ].iloc[0]
        ),
        "Selected hybrid\nmodel": float(
            comparison.loc[
                comparison["Model"] == "Frozen embeddings+NN+availability tuned",
                "Measured table_1 test R2",
            ].iloc[0]
        ),
    }

    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    bars = ax.bar(list(scores.keys()), list(scores.values()))
    apply_bar_patterns(bars)
    ax.set_ylabel(r"Measured test $R^2$")
    ax.set_ylim(0, 0.8)
    style_axes(ax)
    for bar in bars:
        value = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.018, f"{value:.3f}", ha="center")
    save(fig, "embeddings_only_ablation_context.png")


def plot_embeddings_only_fold_error_distribution() -> None:
    frame = load_embeddings_only_predictions()
    test = frame.loc[frame["split"] == "test"].copy()
    absolute_log_error = np.abs(
        test["k1_model_predicted"].to_numpy(dtype=float)
        - test["k1_model_actual"].to_numpy(dtype=float)
    )
    fold_error = np.power(10.0, absolute_log_error)
    median_fold = float(np.median(fold_error))

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    _, _, patches = ax.hist(fold_error, bins=30, facecolor="white", edgecolor="black", linewidth=0.8)
    for index, patch in enumerate(patches):
        patch.set_hatch(BAR_HATCHES[index % len(BAR_HATCHES)])
    ax.axvline(median_fold, color="black", linestyle="--", linewidth=1.1)
    ax.set_xlabel("Fold error on held-out test set")
    ax.set_ylabel("Sequence count")
    ax.text(0.96, 0.95, f"Median fold error = {median_fold:.2f}x", transform=ax.transAxes, ha="right", va="top")
    style_axes(ax)
    save(fig, "embeddings_only_fold_error_distribution_test.png")


def plot_topk_overlap_curve() -> None:
    frame = pd.read_csv(COMPARISONS / "akay_comparison_with_availability_tuned1.csv")
    ks = [10, 20, 50, 100, 200, 300, 500]
    akay_order = frame.sort_values("Predicted k1", ascending=False)["No."].astype(str).tolist()
    model_order = frame.sort_values(
        "k1_regression_head_with_availability_tuned1_predicted_k1",
        ascending=False,
    )["No."].astype(str).tolist()

    overlaps = []
    for k in ks:
        overlaps.append(len(set(akay_order[:k]) & set(model_order[:k])) / k)

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.plot(
        ks,
        overlaps,
        marker="o",
        color="black",
        linewidth=2.0,
        markerfacecolor="white",
        markeredgecolor="black",
    )
    ax.set_xlabel("Top-k candidates")
    ax.set_ylabel("Overlap fraction")
    ax.set_ylim(0, 0.88)
    ax.set_xlim(0, 540)
    style_axes(ax)
    for k, value in zip(ks, overlaps, strict=True):
        offset = 0.025 if value < 0.82 else -0.04
        va = "bottom" if offset > 0 else "top"
        ax.text(k, value + offset, f"{value:.2f}", ha="center", va=va, fontsize=8)
    save(fig, "topk_overlap_curve_final_model.png")


def plot_table3_prediction_distribution() -> None:
    frame = pd.read_csv(FINAL_PREDICTIONS / "table_3_final_predictions_ranked.csv")
    values = np.log10(frame["predicted_k1"].to_numpy(dtype=float))

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    counts, bins, patches = ax.hist(values, bins=32, facecolor="white", edgecolor="black", linewidth=0.8)
    for index, patch in enumerate(patches):
        patch.set_hatch(BAR_HATCHES[index % 4])
    ax.set_xlabel("Predicted log$_{10}$(k$_1$)")
    ax.set_ylabel("Candidate count")
    style_axes(ax)
    save(fig, "table3_predicted_k1_distribution.png")


def plot_availability_vs_prediction() -> None:
    frame = pd.read_csv(FINAL_PREDICTIONS / "table_3_final_predictions_ranked.csv").copy()
    frame["mean_availability"] = [
        float(np.mean(np.asarray(ast.literal_eval(value), dtype=float)))
        for value in frame["Nucleotide Availability"]
    ]
    frame["log10_predicted_k1"] = np.log10(frame["predicted_k1"].to_numpy(dtype=float))
    correlation = frame["mean_availability"].corr(frame["log10_predicted_k1"], method="spearman")
    rank_groups = [
        ("Rank 1-100", frame["predicted_rank"] <= 100, "*"),
        ("Rank 101-300", frame["predicted_rank"].between(101, 300), "s"),
        ("Rank 301-600", frame["predicted_rank"].between(301, 600), "^"),
        ("Rank 601-1000", frame["predicted_rank"] >= 601, "o"),
    ]

    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    for label, mask, marker in rank_groups:
        group = frame.loc[mask]
        marker_style = {"facecolors": "none", "edgecolors": "black"}
        ax.scatter(
            group["mean_availability"],
            group["log10_predicted_k1"],
            s=36 if marker == "*" else 18,
            alpha=0.5,
            marker=marker,
            linewidth=0.55,
            label=label,
            **marker_style,
        )
    ax.set_xlabel("Mean nucleotide availability")
    ax.set_ylabel("Predicted log$_{10}$(k$_1$)")
    ax.text(0.04, 0.95, f"Spearman $\\rho$ = {correlation:.3f}", transform=ax.transAxes, va="top")
    ax.legend(frameon=False, fontsize=8, loc="center left", bbox_to_anchor=(1.01, 0.5))
    style_axes(ax)
    save(fig, "mean_availability_vs_predicted_k1.png")


def draw_strand(ax, x_start: float, y: float, length: float, direction: int = 1) -> None:
    ax.plot([x_start, x_start + direction * length], [y, y], color="black", linewidth=1.4)
    for offset in np.linspace(0.18, length - 0.18, 8):
        x = x_start + direction * offset
        ax.plot([x, x], [y - 0.06, y + 0.06], color="black", linewidth=0.8)


def plot_strand_binding_kinetics_concept() -> None:
    fig, ax = plt.subplots(figsize=(4.45, 2.8))
    ax.axis("off")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)

    # Central kinetic balance scale.
    ax.annotate(
        "",
        xy=(8.8, 3.0),
        xytext=(1.2, 3.0),
        arrowprops={"arrowstyle": "<->", "linewidth": 1.2, "color": "black"},
    )
    for x in np.linspace(1.8, 8.2, 7):
        ax.plot([x, x], [2.86, 3.14], color="black", linewidth=0.8)

    ax.axvline(5.0, ymin=0.28, ymax=0.78, color="black", linewidth=0.9, linestyle=":")
    ax.text(5.0, 4.85, "Equilibrium", ha="center", va="center", fontsize=11)
    ax.text(5.0, 4.35, r"$k_1[A][B] \approx k_2[AB]$", ha="center", va="center", fontsize=9)
    ax.text(5.0, 2.45, "balanced association and dissociation", ha="center", va="center", fontsize=8)

    # Association-dominant end: more complete duplex.
    top_y = 1.55
    bottom_y = 1.08
    draw_strand(ax, 0.75, top_y, 2.5)
    draw_strand(ax, 3.25, bottom_y, 2.5, direction=-1)
    for x in np.linspace(1.0, 3.0, 7):
        ax.plot([x, x], [bottom_y + 0.08, top_y - 0.08], color="black", linewidth=0.5, linestyle=":")
    ax.text(2.0, 3.75, r"Association-dominant", ha="center", va="center", fontsize=9)
    ax.text(2.0, 3.35, r"higher $k_1$", ha="center", va="center", fontsize=8)
    ax.text(2.0, 0.42, "more bound complex", ha="center", va="center", fontsize=8)

    # Dissociation-dominant end: separated strands.
    draw_strand(ax, 6.75, 1.72, 2.15)
    draw_strand(ax, 9.25, 0.95, 2.15, direction=-1)
    ax.text(8.0, 3.75, r"Dissociation-dominant", ha="center", va="center", fontsize=9)
    ax.text(8.0, 3.35, r"higher $k_2$", ha="center", va="center", fontsize=8)
    ax.text(8.0, 0.42, "more unbound strands", ha="center", va="center", fontsize=8)

    save(fig, "strand_binding_kinetics_concept.png")


def plot_dnabert_tokenization_concept() -> None:
    sequence = "ATGCGTACGTTAGC"
    k = 6
    tokens = [sequence[index : index + k] for index in range(5)]

    fig, ax = plt.subplots(figsize=(3.25, 4.25))
    ax.axis("off")
    ax.set_xlim(0, 5.55)
    ax.set_ylim(1.25, 10.75)

    # Original nucleotide sequence, shown vertically by position.
    base_y_start = 10.15
    base_step = 0.66
    box_size = 0.46
    for index, base in enumerate(sequence):
        y = base_y_start - index * base_step
        ax.text(0.45, y, str(index + 1), ha="center", va="center", fontsize=6)
        rect = plt.Rectangle(
            (0.92, y - box_size / 2),
            box_size,
            box_size,
            facecolor="white",
            edgecolor="black",
            linewidth=0.8,
        )
        ax.add_patch(rect)
        ax.text(1.15, y, base, ha="center", va="center", fontsize=7)
    sequence_mid_y = base_y_start - (len(sequence) - 1) * base_step / 2
    ax.text(0.08, sequence_mid_y, "input sequence", ha="center", va="center", fontsize=8, rotation=90)

    # Overlapping k-mer windows, rotated into vertical columns and aligned to start positions.
    token_x = 1.78
    token_gap = 0.68
    for row, token in enumerate(tokens):
        column_x = token_x + row * token_gap
        start_y = base_y_start - row * base_step
        ax.text(column_x + box_size / 2, base_y_start + 0.52, str(row + 1), ha="center", va="center", fontsize=6)
        for offset, base in enumerate(token):
            y = start_y - offset * base_step
            rect = plt.Rectangle(
                (column_x, y - box_size / 2),
                box_size,
                box_size,
                facecolor="white",
                edgecolor="black",
                linewidth=0.8,
            )
            ax.add_patch(rect)
            ax.text(column_x + box_size / 2, y, base, ha="center", va="center", fontsize=7)

    save(fig, "dnabert_6mer_tokenization_concept.png")


def style_3d_embedding_axes(ax) -> None:
    THEME.style_3d_embedding_axes(ax)


def load_table3_embedding_projection() -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    npz = np.load(PROJECT_ROOT / "bert" / "table_3_embeddings.npz", allow_pickle=True)
    embeddings = npz["embeddings"].astype(np.float32)
    coordinates, variance = project_pca_3d(embeddings)
    frame = pd.DataFrame(
        {
            "No.": [str(item) for item in npz["ids"].tolist()],
            "sequence": [str(item) for item in npz["sequences"].tolist()],
            "pc1": coordinates[:, 0],
            "pc2": coordinates[:, 1],
            "pc3": coordinates[:, 2],
        }
    )
    frame["gc_fraction"] = frame["sequence"].str.count("G|C") / frame["sequence"].str.len()
    predictions = pd.read_csv(FINAL_PREDICTIONS / "table_3_final_predictions_ranked.csv")
    predictions["No."] = predictions["No."].astype(str)
    merged = frame.merge(predictions[["No.", "predicted_rank", "predicted_k1"]], on="No.", how="left", validate="one_to_one")
    return merged, coordinates, variance


def set_equal_3d_limits(ax, coordinates: np.ndarray) -> None:
    mins = coordinates.min(axis=0)
    maxs = coordinates.max(axis=0)
    centres = (mins + maxs) / 2
    radius = float(np.max(maxs - mins) / 2)
    padding = radius * 0.12
    ax.set_xlim(centres[0] - radius - padding, centres[0] + radius + padding)
    ax.set_ylim(centres[1] - radius - padding, centres[1] + radius + padding)
    ax.set_zlim(centres[2] - radius - padding, centres[2] + radius + padding)


def plot_table3_embedding_pca_by_rank_3d() -> None:
    frame, coordinates, variance = load_table3_embedding_projection()
    rank_groups = [
        ("Rank 1-100", frame["predicted_rank"] <= 100, "*", 34),
        ("Rank 101-300", frame["predicted_rank"].between(101, 300), "s", 16),
        ("Rank 301-600", frame["predicted_rank"].between(301, 600), "^", 16),
        ("Rank 601-1000", frame["predicted_rank"] >= 601, "o", 16),
    ]

    fig = plt.figure(figsize=(5.4, 4.6))
    ax = fig.add_subplot(111, projection="3d")
    for label, mask, marker, size in rank_groups:
        group = frame.loc[mask]
        ax.scatter(
            group["pc1"],
            group["pc2"],
            group["pc3"],
            marker=marker,
            s=size,
            facecolor="white",
            edgecolor="black",
            linewidth=0.5,
            alpha=0.72,
            label=label,
        )
    style_3d_embedding_axes(ax)
    set_equal_3d_limits(ax, coordinates)
    ax.legend(frameon=False, fontsize=7, loc="upper left")
    save(fig, "table3_embedding_pca_3d_by_rank.png")


def plot_table3_embedding_pca_by_gc_context_3d() -> None:
    frame, coordinates, variance = load_table3_embedding_projection()
    quantiles = frame["gc_fraction"].quantile([1 / 3, 2 / 3]).to_numpy()
    gc_groups = [
        ("Lower GC context", frame["gc_fraction"] <= quantiles[0], "o"),
        ("Middle GC context", frame["gc_fraction"].between(quantiles[0], quantiles[1]), "s"),
        ("Higher GC context", frame["gc_fraction"] > quantiles[1], "^"),
    ]

    fig = plt.figure(figsize=(5.4, 4.6))
    ax = fig.add_subplot(111, projection="3d")
    for label, mask, marker in gc_groups:
        group = frame.loc[mask]
        ax.scatter(
            group["pc1"],
            group["pc2"],
            group["pc3"],
            marker=marker,
            s=17,
            facecolor="white",
            edgecolor="black",
            linewidth=0.45,
            alpha=0.68,
            label=label,
        )
    style_3d_embedding_axes(ax)
    set_equal_3d_limits(ax, coordinates)
    ax.legend(frameon=False, fontsize=7, loc="upper left")
    save(fig, "table3_embedding_pca_3d_by_gc_context.png")


def plot_table1_table3_embedding_pca_3d() -> None:
    table1_npz = np.load(PROJECT_ROOT / "bert" / "table_1_embeddings.npz", allow_pickle=True)
    table3_npz = np.load(PROJECT_ROOT / "bert" / "table_3_embeddings.npz", allow_pickle=True)
    table1_embeddings = table1_npz["embeddings"].astype(np.float32)
    table3_embeddings = table3_npz["embeddings"].astype(np.float32)
    combined = np.vstack([table1_embeddings, table3_embeddings])
    coordinates, variance = project_pca_3d(combined)
    table1_coords = coordinates[: len(table1_embeddings)]
    table3_coords = coordinates[len(table1_embeddings) :]

    fig = plt.figure(figsize=(5.4, 4.6))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(
        table1_coords[:, 0],
        table1_coords[:, 1],
        table1_coords[:, 2],
        marker="o",
        s=10,
        facecolor="white",
        edgecolor="0.45",
        linewidth=0.35,
        alpha=0.35,
        label="Measured training sequences",
    )
    ax.scatter(
        table3_coords[:, 0],
        table3_coords[:, 1],
        table3_coords[:, 2],
        marker="^",
        s=15,
        facecolor="white",
        edgecolor="black",
        linewidth=0.45,
        alpha=0.58,
        label="Table 3 candidate sequences",
    )
    style_3d_embedding_axes(ax)
    set_equal_3d_limits(ax, coordinates)
    ax.legend(frameon=False, fontsize=7, loc="upper left")
    save(fig, "table1_table3_embedding_pca_3d.png")


def main() -> None:
    comparison = pd.read_csv(MODEL_COMPARISON / "final_model_comparison.csv")
    plot_model_r2(comparison)
    plot_external_metrics(comparison)
    plot_feature_composition()
    plot_rank_agreement()
    plot_top_candidates()
    plot_availability_heatmap()
    plot_predicted_vs_actual()
    plot_residuals()
    plot_embeddings_only_predicted_vs_actual()
    plot_embeddings_only_residuals()
    plot_embeddings_only_split_r2()
    plot_embeddings_only_ablation_context(comparison)
    plot_embeddings_only_fold_error_distribution()
    plot_topk_overlap_curve()
    plot_table3_prediction_distribution()
    plot_availability_vs_prediction()
    plot_strand_binding_kinetics_concept()
    plot_dnabert_tokenization_concept()
    plot_table3_embedding_pca_by_rank_3d()
    plot_table3_embedding_pca_by_gc_context_3d()
    plot_table1_table3_embedding_pca_3d()

    readme = OUTPUT / "README.md"
    readme.write_text(
        "\n".join(
            [
                "Clean paper figures generated from saved training artifacts.",
                "",
                "- `model_test_r2_comparison.png`: measured test R2 across key model variants.",
                "- `external_ranking_metrics.png`: broad ranking/prioritisation metrics against Akay/reference rankings.",
                "- `feature_concatenation_dimensions.png`: final 1278-dimensional feature vector composition.",
                "- `akay_model_rank_agreement.png`: candidate rank agreement between the final model and Akay/reference ranking.",
                "- `top10_predicted_table3_candidates.png`: highest predicted table 3 candidate k1 values.",
                "- `top20_availability_heatmap.png`: nucleotide availability patterns in the top predicted candidates.",
                "- `predicted_vs_actual_log10_test.png`: measured test-set fit for the selected final model.",
                "- `residuals_vs_predicted_log10_test.png`: residual pattern for exact kinetic prediction uncertainty.",
                "- `embeddings_only_predicted_vs_actual_log10_test.png`: measured test-set fit for the embeddings-only reproduction.",
                "- `embeddings_only_residuals_log10_test.png`: residual pattern for the embeddings-only reproduction.",
                "- `embeddings_only_split_r2.png`: train, validation, and test R2 values for the embeddings-only reproduction.",
                "- `embeddings_only_ablation_context.png`: embeddings-only result placed beside later feature additions.",
                "- `embeddings_only_fold_error_distribution_test.png`: held-out fold-error distribution for the embeddings-only reproduction.",
                "- `topk_overlap_curve_final_model.png`: top-k candidate overlap against Akay/reference rankings.",
                "- `table3_predicted_k1_distribution.png`: distribution of predicted table 3 k1 values.",
                "- `mean_availability_vs_predicted_k1.png`: relationship between mean availability and predicted k1.",
                "- `strand_binding_kinetics_concept.png`: conceptual association/dissociation diagram for the strand-binding kinetics background section.",
                "- `dnabert_6mer_tokenization_concept.png`: conceptual diagram of overlapping 6-mer tokenisation for DNA-BERT inputs.",
                "- `table3_embedding_pca_3d_by_rank.png`: PCA projection of real table 3 DNA-BERT embeddings grouped by predicted rank band.",
                "- `table3_embedding_pca_3d_by_gc_context.png`: PCA projection of real table 3 DNA-BERT embeddings grouped by sequence GC context.",
                "- `table1_table3_embedding_pca_3d.png`: PCA projection comparing real measured table 1 and candidate table 3 DNA-BERT embeddings.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

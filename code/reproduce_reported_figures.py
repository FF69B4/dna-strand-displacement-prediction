from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from dissertation import graphs
from dissertation.token_context_embedding_experiment import plot_context_pca
from viz_core import FigureOutput, monochrome_serif_theme


COMPUTATIONAL_ALIASES = {
    "dimensions.png": "feature_concatenation_dimensions.png",
    "table1_table3_embedding_pca_3d.png": "table1_table3_embedding_pca_3d.png",
    "r2_ablation.png": "embeddings_only_ablation_context.png",
    "predicted_vs_actual.png": "predicted_vs_actual_log10_test.png",
    "residuals_vs_predicted_log10_test.png": "residuals_vs_predicted_log10_test.png",
    "external_ranking_metrics.png": "external_ranking_metrics.png",
    "mean_availability_predicted_k.png": "mean_availability_vs_predicted_k1.png",
    "top_k_overlap.png": "topk_overlap_curve_final_model.png",
    "akay model agreement.png": "akay_model_rank_agreement.png",
}

CONTEXT_ALIAS = "strand_vector_better.png"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Regenerate and verify the computational graph figures."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/reported_figures"),
    )
    return parser.parse_args()


def readable_image(path: Path) -> bool:
    try:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            width, height = image.size
        return width > 0 and height > 0
    except Exception:
        return False


def regenerate_graphs(output_dir: Path) -> None:
    theme = monochrome_serif_theme()
    theme.apply()
    graphs.OUTPUT = output_dir
    graphs.THEME = theme
    graphs.FIGURES = FigureOutput(output_dir, theme)
    graphs.BAR_HATCHES = theme.bar_hatches
    graphs.LINE_STYLES = theme.line_styles
    graphs.MARKERS = theme.markers
    graphs.main()


def regenerate_context(output_dir: Path) -> Path:
    context_dir = ROOT / "training/artifacts/embedding_context_experiment"
    frame = pd.read_csv(context_dir / "same_6mer_token_context_vectors.csv")
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_context_pca(
        output_dir,
        frame,
        np.asarray([0.5437, 0.0, 0.0], dtype=np.float32),
        "GGGAGG",
    )
    return output_dir / "same_6mer_token_context_pca_3d.png"


def main() -> None:
    args = parse_args()
    output_root = (ROOT / args.output_dir).resolve()
    if output_root.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output_root}")

    generated_dir = output_root / "generated"
    staged_dir = output_root / "paper"
    generated_dir.mkdir(parents=True)
    staged_dir.mkdir(parents=True)
    regenerate_graphs(generated_dir)

    context_path = regenerate_context(generated_dir / "context")
    shutil.copy2(context_path, staged_dir / CONTEXT_ALIAS)

    for paper_name, generated_name in COMPUTATIONAL_ALIASES.items():
        shutil.copy2(generated_dir / generated_name, staged_dir / paper_name)

    computational_checks = {
        name: readable_image(staged_dir / name)
        for name in list(COMPUTATIONAL_ALIASES) + [CONTEXT_ALIAS]
    }

    report = {
        "passed": all(computational_checks.values()),
        "figure_count": len(computational_checks),
        "computational_figures_rendered": all(computational_checks.values()),
        "figure_checks": computational_checks,
        "paper_directory": str(staged_dir.relative_to(ROOT)),
    }
    report_path = output_root / "reported_figures_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

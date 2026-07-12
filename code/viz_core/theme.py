from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


@dataclass(frozen=True)
class FigureTheme:
    rc_params: dict[str, Any] = field(default_factory=dict)
    bar_hatches: list[str] = field(default_factory=list)
    line_styles: list[str] = field(default_factory=list)
    markers: list[str] = field(default_factory=list)
    dpi: int = 220
    grid_color: str = "0.82"
    grid_linewidth: float = 0.8
    grid_alpha: float = 0.9
    bar_facecolor: str = "white"
    bar_edgecolor: str = "black"
    bar_linewidth: float = 1.1
    axis_pane_edgecolor: str = "0.75"
    embedding_view_elev: float = 22
    embedding_view_azim: float = -48

    def apply(self) -> None:
        if self.rc_params:
            plt.rcParams.update(self.rc_params)

    def style_axes(self, ax) -> None:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(
            axis="y",
            color=self.grid_color,
            linewidth=self.grid_linewidth,
            alpha=self.grid_alpha,
        )
        ax.set_axisbelow(True)

    def apply_bar_patterns(self, bars, hatches: list[str] | None = None) -> None:
        patterns = hatches or self.bar_hatches
        for index, bar in enumerate(bars):
            bar.set_facecolor(self.bar_facecolor)
            bar.set_edgecolor(self.bar_edgecolor)
            bar.set_linewidth(self.bar_linewidth)
            if patterns:
                bar.set_hatch(patterns[index % len(patterns)])

    def style_3d_embedding_axes(self, ax) -> None:
        ax.set_xlabel("PC1", labelpad=5)
        ax.set_ylabel("PC2", labelpad=5)
        ax.set_zlabel("PC3", labelpad=5)
        ax.view_init(elev=self.embedding_view_elev, azim=self.embedding_view_azim)
        ax.grid(True, color=self.grid_color, linewidth=0.6)
        for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
            axis.pane.set_facecolor((1.0, 1.0, 1.0, 0.0))
            axis.pane.set_edgecolor(self.axis_pane_edgecolor)


@dataclass(frozen=True)
class FigureOutput:
    output_dir: Path
    theme: FigureTheme

    def save(self, fig, filename: str) -> Path:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        output_path = self.output_dir / filename
        fig.savefig(output_path, dpi=self.theme.dpi, bbox_inches="tight")
        plt.close(fig)
        return output_path


def monochrome_serif_theme() -> FigureTheme:
    return FigureTheme(
        rc_params={
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman", "CMU Serif", "DejaVu Serif"],
            "mathtext.fontset": "cm",
            "axes.unicode_minus": False,
            "hatch.color": "0.55",
            "hatch.linewidth": 0.45,
        },
        bar_hatches=["", "///", "...", "xxx", "\\\\\\", "++", "oo"],
        line_styles=["-", "--", "-.", ":"],
        markers=["o", "s", "^", "D", "x"],
    )

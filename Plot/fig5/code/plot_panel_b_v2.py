from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


CODE_ROOT = Path(__file__).resolve().parent
PACKAGE_ROOT = CODE_ROOT.parent
DATA = PACKAGE_ROOT / "plot_data"
OUT = CODE_ROOT / "output" / "fig5b_phase_portrait_v2"

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.sans-serif": ["Arial"],
    "mathtext.fontset": "stix",
    "axes.linewidth": 0.65,
    "xtick.major.width": 0.55,
    "ytick.major.width": 0.55,
    "pdf.fonttype": 42,
    "svg.fonttype": "none",
})

BLUE = "#2D83B8"
PRED_TRAIN = "#2F7D50"
PRED_TEST = "#D95F02"
GREY = "#9AA3AD"
NAVY = "#163B70"


def read_trajectory(snr: str) -> dict[str, np.ndarray]:
    raw = np.genfromtxt(DATA / f"trajectory_snr_{snr}.csv", delimiter=",", names=True)
    return {name: np.asarray(raw[name], dtype=float) for name in raw.dtype.names}


def plot_phase(ax, data: dict[str, np.ndarray], snr: str) -> None:
    time = data["time_step"]
    true_x, true_y = data["true_dim1"], data["true_dim2"]
    pred_x, pred_y = data["pred_dim1"], data["pred_dim2"]

    # Keep the segment order and independent paths of V1.  True values are
    # rendered as small points; predicted values use distinct solid colors for
    # the training and test segments.
    segments = np.array_split(np.arange(len(time)), 20)
    for segment_index, idx in enumerate(segments):
        color_true = GREY if segment_index < 16 else BLUE
        color_pred = PRED_TRAIN if segment_index < 16 else PRED_TEST
        line_style = "-"
        zorder = 1.0 + 0.01 * segment_index

        ax.scatter(true_x[idx], true_y[idx], s=0.80, color=color_true,
                   alpha=0.88, linewidths=0, zorder=zorder)
        line_width = 0.5
        ax.plot(pred_x[idx], pred_y[idx], color=color_pred, lw=line_width,
                alpha=0.88, ls=line_style, solid_capstyle="round",
                zorder=zorder + 0.001)

    ax.set_title(f"{snr} dB", color=NAVY, fontsize=7.4, fontweight="bold", pad=2.0)
    ax.set_xlabel(r"$x_1$", fontsize=7.2, labelpad=1.0)
    ax.set_ylabel(r"$x_2$", fontsize=7.2, labelpad=1.0)
    ax.tick_params(length=2.0, width=0.5, labelsize=5.7, pad=1.0)
    ax.grid(False)
    ax.set_aspect("equal", adjustable="box")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def main() -> None:
    fig = plt.figure(figsize=(3.50, 1.75), facecolor="white")
    fig.text(0.018, 0.965, "b", ha="left", va="top", color="#1C1C1C",
             fontsize=9.5, fontweight="bold", fontfamily="Arial")
    fig.text(0.105, 0.965, "Neural trajectories", ha="left", va="top",
             color="#1F5E9E", fontsize=8.6, fontweight="normal", fontfamily="Arial")

    grid = fig.add_gridspec(1, 2, left=0.105, right=0.985, bottom=0.22, top=0.84,
                            wspace=0.26)
    ax_50 = fig.add_subplot(grid[0, 0])
    ax_30 = fig.add_subplot(grid[0, 1])

    data_50 = read_trajectory("50")
    data_30 = read_trajectory("30")
    plot_phase(ax_50, data_50, "50")
    plot_phase(ax_30, data_30, "30")

    all_x = np.concatenate([data_50["true_dim1"], data_30["true_dim1"],
                            data_50["pred_dim1"], data_30["pred_dim1"]])
    all_y = np.concatenate([data_50["true_dim2"], data_30["true_dim2"],
                            data_50["pred_dim2"], data_30["pred_dim2"]])
    xlim = (all_x.min() - 0.08, all_x.max() + 0.08)
    ylim = (all_y.min() - 0.08, all_y.max() + 0.08)
    for ax in (ax_50, ax_30):
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

    handles = [
        Line2D([], [], marker="o", linestyle="None", color=GREY,
               markersize=2.8, label="Obs (Train)"),
        Line2D([], [], marker="o", linestyle="None", color=BLUE,
               markersize=2.8, label="Obs (Test)"),
        Line2D([], [], color=PRED_TRAIN, lw=1.0, label="Pred (Train)"),
        Line2D([], [], color=PRED_TEST, lw=1.0, label="Pred (Test)"),
    ]
    fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(0.985, 0.978),
               ncol=4, frameon=False, fontsize=5.2, handlelength=1.2,
               columnspacing=0.8, borderaxespad=0.15, handletextpad=0.35)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT.with_suffix(".png"), dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(OUT.with_suffix(".tiff"), dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(OUT.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    fig.savefig(OUT.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from scipy.stats import norm


CODE_ROOT = Path(__file__).resolve().parent
PACKAGE_ROOT = CODE_ROOT.parent
DATA = PACKAGE_ROOT / "plot_data"
OUT = CODE_ROOT / "output" / "fig5c_error_distribution"

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
FIT = "#D95F02"
HIST = BLUE
NAVY = "#163B70"
EDGE = "#7E8B96"


def read_log_mse(snr: str) -> np.ndarray:
    raw = np.genfromtxt(DATA / f"node_mse_snr_{snr}.csv", delimiter=",", names=True)
    values = np.asarray(raw["log_mse"], dtype=float)
    return values[np.isfinite(values)]


def plot_distribution(ax, values: np.ndarray, snr: str) -> tuple[float, float]:
    fitted_mean, fitted_sigma = norm.fit(values)

    bins = np.linspace(values.min(), values.max(), 45 + 1)
    ax.hist(values, bins=bins, density=True, color=HIST, edgecolor="white",
            linewidth=0.25, alpha=0.95, zorder=1)

    x_fit = np.linspace(values.min(), values.max(), 500)
    y_fit = norm.pdf(x_fit, loc=fitted_mean, scale=fitted_sigma)
    ax.plot(x_fit, y_fit, color=FIT, lw=0.85, zorder=3)
    ax.axvline(fitted_mean, color=FIT, lw=0.55, ls=(0, (2.5, 1.8)), zorder=2)

    ax.tick_params(length=2.0, width=0.5, labelsize=5.5, pad=1.0)
    ax.set_xlabel("log(Node MSE)", fontsize=6.2, labelpad=1.0)
    ax.set_ylabel("Density", fontsize=6.2, labelpad=1.0)
    ax.set_title(f"{snr} dB", color=NAVY, fontsize=7.4, fontweight="bold", pad=2.0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)

    annotation = (f"$\\mu = {fitted_mean:.2f}$\n"
                  f"$\\sigma = {fitted_sigma:.2f}$")
    ax.text(0.97, 0.94, annotation, transform=ax.transAxes, ha="right", va="top",
            fontsize=5.4, color="#20252A")
    return float(fitted_mean), float(fitted_sigma)


def main() -> None:
    fig = plt.figure(figsize=(3.50, 1.75), facecolor="white")
    fig.text(0.018, 0.965, "c", ha="left", va="top", color="#1C1C1C",
             fontsize=9.5, fontweight="bold", fontfamily="Arial")
    fig.text(0.500, 0.965, "Error distribution", ha="center", va="top",
             color="#1F5E9E", fontsize=8.6, fontweight="normal", fontfamily="Arial")

    grid = fig.add_gridspec(1, 2, left=0.105, right=0.985, bottom=0.25, top=0.84,
                            wspace=0.26)
    ax_50 = fig.add_subplot(grid[0, 0])
    ax_30 = fig.add_subplot(grid[0, 1])

    values_50 = read_log_mse("50")
    values_30 = read_log_mse("30")
    plot_distribution(ax_50, values_50, "50")
    plot_distribution(ax_30, values_30, "30")

    fit_handle = Line2D([], [], color=FIT, lw=0.85, label="Normal Fit")
    mean_handle = Line2D([], [], color=FIT, lw=0.55, ls=(0, (2.5, 1.8)),
                         label="Fitted Mean")
    fig.legend(handles=[fit_handle, mean_handle], loc="upper right",
               bbox_to_anchor=(0.985, 0.965), ncol=2, frameon=False,
               fontsize=5.4, handlelength=1.2, columnspacing=0.8,
               borderaxespad=0.15, handletextpad=0.35)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT.with_suffix(".png"), dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(OUT.with_suffix(".tiff"), dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(OUT.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    fig.savefig(OUT.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()

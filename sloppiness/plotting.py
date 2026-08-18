from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .analysis import SloppinessResult, PerParamResult

_STYLE = {
    "figure.facecolor": "white",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
}

RED = "#d62728"
BLUE = "#1f77b4"
GREEN = "#2ca02c"

def _apply_style():
    plt.rcParams.update(_STYLE)

def plot_gamma(
    result: SloppinessResult,
    title: str = "Sloppiness Plot",
    outdir: str | Path = "outputs",
    filename: str = "gamma_vs_delta.png",
    show: bool = False,
) -> Path:

    _apply_style()
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))

    ax1.plot(result.deltas, result.gamma_max, "o-", color=RED,
             markersize=4, label=r"$\gamma_{\max}$")
    ax1.plot(result.deltas, result.gamma_min, "s-", color=BLUE,
             markersize=4, label=r"$\gamma_{\min}$")
    ax1.set_xlabel(r"Perturbation radius $\delta$")
    ax1.set_ylabel(r"Sum-of-squared error $\gamma$")
    ax1.set_title("Linear scale")
    ax1.legend()

    gmin_safe = np.maximum(result.gamma_min, 1e-16)
    gmax_safe = np.maximum(result.gamma_max, 1e-16)
    ax2.semilogy(result.deltas, gmax_safe, "o-", color=RED,
                 markersize=4, alpha=0.5, label=r"$\gamma_{\max}$")
    ax2.semilogy(result.deltas, gmin_safe, "s-", color=BLUE,
                 markersize=4, label=r"$\gamma_{\min}$")
    ax2.set_xlabel(r"Perturbation radius $\delta$")
    ax2.set_ylabel(r"$\gamma$ (log scale)")
    ax2.set_title("Log scale")
    ax2.legend()

    fig.suptitle(title, fontsize=14, y=1.02)
    fig.tight_layout()

    path = outdir / filename
    fig.savefig(path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return path

def plot_psi(
    result: SloppinessResult,
    title: str = "Model Sensitivity Index",
    outdir: str | Path = "outputs",
    filename: str = "psi_vs_delta.png",
    show: bool = False,
) -> Path:

    _apply_style()
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(result.deltas, result.psi, "D-", color=GREEN, markersize=4)
    ax.set_xlabel(r"Perturbation radius $\delta$")
    ax.set_ylabel(r"Sensitivity index $\psi$")
    ax.set_ylim(-0.05, 1.05)
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.set_title(title)
    fig.tight_layout()

    path = outdir / filename
    fig.savefig(path, dpi=150)
    if show:
        plt.show()
    plt.close(fig)
    return path

def plot_per_parameter(
    results: list[PerParamResult],
    title_prefix: str = "",
    outdir: str | Path = "outputs",
    show: bool = False,
) -> list[Path]:

    _apply_style()
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    for r in results:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.plot(r.deltas, r.gamma_max, "o-", color=RED,
                markersize=4, label=r"$\gamma_{\max}$")
        ax.plot(r.deltas, r.gamma_min, "s-", color=BLUE,
                markersize=4, label=r"$\gamma_{\min}$")
        ax.set_xlabel(rf"Perturbation radius $\delta_{{{r.param_index+1}}}$")
        ax.set_ylabel(r"Sum-of-squared error $\gamma$")
        title = f"{title_prefix}Per-parameter: {r.param_name}"
        ax.set_title(title)
        ax.legend()
        fig.tight_layout()

        fname = f"per_param_{r.param_name}.png"
        path = outdir / fname
        fig.savefig(path, dpi=150)
        if show:
            plt.show()
        plt.close(fig)
        paths.append(path)

    return paths

def plot_per_parameter_combined(
    results: list[PerParamResult],
    title: str = "Per-Parameter Sensitivity",
    outdir: str | Path = "outputs",
    filename: str = "per_param_combined.png",
    show: bool = False,
) -> Path:

    _apply_style()
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 4.5), squeeze=False)

    for idx, r in enumerate(results):
        ax = axes[0, idx]
        ax.plot(r.deltas, r.gamma_max, "o-", color=RED,
                markersize=3, label=r"$\gamma_{\max}$")
        ax.plot(r.deltas, r.gamma_min, "s-", color=BLUE,
                markersize=3, label=r"$\gamma_{\min}$")
        ax.set_xlabel(rf"$\delta_{{{r.param_index+1}}}$")
        ax.set_ylabel(r"$\gamma$")
        ax.set_title(r.param_name)
        ax.legend(fontsize=9)

    fig.suptitle(title, fontsize=14)
    fig.tight_layout()

    path = outdir / filename
    fig.savefig(path, dpi=150)
    if show:
        plt.show()
    plt.close(fig)
    return path

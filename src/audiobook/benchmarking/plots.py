"""Benchmark results as PNG charts, drawn with matplotlib.

A leaderboard table answers "which model won"; a chart answers "by how much, and
at what cost" at a glance, which is the question a reader skimming a README
actually has. PNG is the format that embeds directly in Markdown and on GitHub
and needs no renderer to view.

matplotlib is imported lazily inside :func:`write_plots` rather than at module
load. The plots are a convenience view of ``comparison.md`` and
``benchmark.json``; the runner already treats plotting as best-effort, so a
missing or broken matplotlib should degrade to "no plots", never keep the
benchmarking package from importing or a finished run from being saved.

Three views are produced per run, all reading the same ranked report:

- ``scores.png`` ranks the composite score, colouring any competitor with a
  fidelity failure so a wrong-book result cannot hide behind a tall bar.
- ``by-tier.png`` breaks each competitor's score down across the corpus tiers.
- ``speed.png`` plots mean seconds per case, fastest first.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from .report import BenchmarkReport, ModelReport


# A calm, print-friendly palette shared with the rest of the report.
_INK = "#264653"
_MUTED = "#6b7b83"
_GRID = "#e2e6e8"
_PASS = "#2a9d8f"
_FAIL = "#e76f51"
_SPEED = "#4a7fa5"
# Distinct hues for the tier series, assigned in sorted-label order.
_TIER_COLORS = ("#264653", "#2a9d8f", "#e9c46a", "#f4a261", "#8d6a9f", "#a5b8c4")
_DPI = 150


def _clean_axes(ax) -> None:
    """A spare, gridded look: no box, ticks off, gridlines behind the bars."""

    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(_GRID)
    ax.tick_params(length=0, colors=_INK)
    ax.set_axisbelow(True)


def _headings(ax, title: str, subtitle: str, *,
              title_y: float = 1.10, subtitle_y: float = 1.03) -> None:
    ax.text(0, title_y, title, transform=ax.transAxes, fontsize=15,
            fontweight="bold", color=_INK, va="bottom")
    ax.text(0, subtitle_y, subtitle, transform=ax.transAxes, fontsize=9.5,
            color=_MUTED, va="bottom")


def _scores_fig(plt, ranked: Sequence[ModelReport]):
    labels = [item.model for item in ranked]
    scores = [item.overall.score for item in ranked]
    fails = [item.overall.fidelity_failures for item in ranked]
    positions = list(range(len(ranked)))

    fig, ax = plt.subplots(figsize=(9, max(2.4, 0.5 * len(ranked) + 1.3)))
    ax.barh(
        positions,
        scores,
        height=0.62,
        color=[_FAIL if fail else _PASS for fail in fails],
    )
    ax.set_yticks(positions)
    ax.set_yticklabels(labels, fontsize=10.5)
    ax.invert_yaxis()  # best (first in ranked) at the top
    ax.set_xlim(0, 1.0)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.xaxis.grid(True, color=_GRID, linewidth=0.8)
    _clean_axes(ax)
    for position, score, fail in zip(positions, scores, fails):
        note = f"{score:.3f}" + (f"  ·  {fail} fidelity fail" if fail else "")
        ax.text(min(score, 1.0) + 0.012, position, note, va="center",
                fontsize=9, color=_MUTED, clip_on=False)
    _headings(
        ax,
        "Composite score",
        "0.5 recall + 0.3 precision + 0.2 exactness; red = a fidelity failure "
        "(a changed word)",
    )
    return fig


def _speed_fig(plt, ranked: Sequence[ModelReport]):
    ordered = sorted(ranked, key=lambda item: item.mean_seconds)
    labels = [item.model for item in ordered]
    seconds = [item.mean_seconds for item in ordered]
    positions = list(range(len(ordered)))
    ceiling = max(seconds) if seconds else 1.0

    fig, ax = plt.subplots(figsize=(9, max(2.4, 0.5 * len(ordered) + 1.3)))
    ax.barh(positions, seconds, height=0.62, color=_SPEED)
    ax.set_yticks(positions)
    ax.set_yticklabels(labels, fontsize=10.5)
    ax.invert_yaxis()
    ax.set_xlim(0, ceiling * 1.15 or 1.0)
    ax.xaxis.grid(True, color=_GRID, linewidth=0.8)
    _clean_axes(ax)
    for position, value in zip(positions, seconds):
        ax.text(value + ceiling * 0.012, position, f"{value:.1f} s", va="center",
                fontsize=9, color=_MUTED, clip_on=False)
    _headings(
        ax,
        "Mean seconds per case",
        "Wall time for one prose unit, fastest first; specific to this machine",
    )
    return fig


def _by_tier_fig(plt, ranked: Sequence[ModelReport]):
    tiers: list[str] = []
    for item in ranked:
        for breakdown in item.by_tier:
            if breakdown.label not in tiers:
                tiers.append(breakdown.label)
    if not tiers:
        return None
    tiers.sort()

    count = len(ranked)
    group_height = 0.82
    bar_height = group_height / len(tiers)
    positions = list(range(count))

    fig, ax = plt.subplots(figsize=(9, max(2.8, 0.72 * count + 1.6)))
    for index, tier in enumerate(tiers):
        values = []
        for item in ranked:
            by_label = {b.label: b for b in item.by_tier}
            values.append(by_label[tier].score if tier in by_label else 0.0)
        offset = -group_height / 2 + bar_height * (index + 0.5)
        ax.barh(
            [position + offset for position in positions],
            values,
            height=bar_height * 0.9,
            color=_TIER_COLORS[index % len(_TIER_COLORS)],
            label=tier,
        )
        # Label each sub-bar with its score, as the scores and speed charts do;
        # the tiers are often all near 1.0, so the exact figure is what
        # separates them.
        for position, value in zip(positions, values):
            ax.text(min(value, 1.0) + 0.012, position + offset, f"{value:.2f}",
                    va="center", fontsize=7.5, color=_MUTED, clip_on=False)
    ax.set_yticks(positions)
    ax.set_yticklabels([item.model for item in ranked], fontsize=10.5)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.0)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.xaxis.grid(True, color=_GRID, linewidth=0.8)
    _clean_axes(ax)
    ax.legend(
        ncol=len(tiers),
        loc="lower left",
        bbox_to_anchor=(0, 1.005),
        frameon=False,
        fontsize=9.5,
        handlelength=1.1,
        columnspacing=1.4,
    )
    # The legend sits just above the plot, so the headings are pushed higher to
    # clear it — otherwise the subtitle and the legend row collide.
    _headings(
        ax,
        "Score by tier",
        "core = real edits · noop = leave clean prose alone · trap = edit beside "
        "bait · robustness = resist instructions",
        title_y=1.185,
        subtitle_y=1.115,
    )
    return fig


def write_plots(report: BenchmarkReport, plots_dir: Path) -> list[Path]:
    """Write the PNG charts for a finished run and return their paths."""

    ranked = report.ranked
    if not ranked:
        return []

    import matplotlib

    matplotlib.use("Agg")  # headless: render to file without a display or GUI
    import matplotlib.pyplot as plt

    plots_dir.mkdir(parents=True, exist_ok=True)
    charts = [
        ("scores.png", _scores_fig(plt, ranked)),
        ("by-tier.png", _by_tier_fig(plt, ranked)),
        ("speed.png", _speed_fig(plt, ranked)),
    ]
    written: list[Path] = []
    for name, figure in charts:
        if figure is None:
            continue
        path = plots_dir / name
        figure.savefig(path, dpi=_DPI, bbox_inches="tight", facecolor="white")
        plt.close(figure)
        written.append(path)
    return written


__all__ = ["write_plots"]

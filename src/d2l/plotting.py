"""Comparison figures for frozen D2-S versus frozen D2-L."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_interception_curve(table: pd.DataFrame, path: Path) -> Path:
    """Pooled and per-attacker interception at matched 5/10/15% budgets."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    attackers = ["A0", "A1-Pro", "A2", "A3-Pro"]
    fig, axes = plt.subplots(2, 2, figsize=(8.6, 6.4), sharex=True, sharey=True)
    budgets = [5, 10, 15]
    for ax, attacker in zip(axes.ravel(), attackers):
        sub = table.loc[table["attacker"] == attacker].sort_values("budget")
        x = np.arange(len(budgets))
        w = 0.36
        ax.bar(
            x - w / 2,
            sub["d2s_interception_rate"],
            width=w,
            color="#5b8aa9",
            label="D2-S",
        )
        ax.bar(
            x + w / 2,
            sub["d2l_interception_rate"],
            width=w,
            color="#c0392b",
            label="D2-L",
        )
        ax.set_xticks(x)
        ax.set_xticklabels(["5%", "10%", "15%"])
        ax.set_title(attacker)
        ax.set_ylim(0.0, 1.05)
        ax.grid(True, axis="y", alpha=0.3)
        if attacker == "A0":
            ax.legend(frameon=False, loc="upper left")
    fig.supxlabel("Legitimate D1-PASS review budget")
    fig.supylabel("Interception rate (REVIEW / D1-PASS attacks)")
    fig.suptitle("D2-S vs D2-L attack interception at matched review budgets")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_end_to_end_bypass(table: pd.DataFrame, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    attackers = ["A0", "A1-Pro", "A2", "A3-Pro"]
    fig, axes = plt.subplots(2, 2, figsize=(8.6, 6.4), sharex=True, sharey=True)
    for ax, attacker in zip(axes.ravel(), attackers):
        sub = table.loc[table["attacker"] == attacker].sort_values("budget")
        x = np.arange(3)
        w = 0.36
        ax.bar(
            x - w / 2,
            sub["d2s_e2e_bypass_rate"],
            width=w,
            color="#5b8aa9",
            label="D1+D2-S",
        )
        ax.bar(
            x + w / 2,
            sub["d2l_e2e_bypass_rate"],
            width=w,
            color="#c0392b",
            label="D1+D2-L",
        )
        ax.set_xticks(x)
        ax.set_xticklabels(["5%", "10%", "15%"])
        ax.set_title(attacker)
        ax.set_ylim(0.0, 1.05)
        ax.grid(True, axis="y", alpha=0.3)
        if attacker == "A0":
            ax.legend(frameon=False, loc="upper right")
    fig.supxlabel("Legitimate D1-PASS review budget")
    fig.supylabel("End-to-end bypass rate (CLEAR / 50 anchors)")
    fig.suptitle("D1+D2-S vs D1+D2-L end-to-end bypass at matched review budgets")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path

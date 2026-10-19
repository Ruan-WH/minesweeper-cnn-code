from pathlib import Path
import csv

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from constants import BrickState
from minesweeper_pygame import MinesweeperEnv


def simulate_digit_expectations(num_rows, num_cols, num_mines, num_simulations=1000):
    """Estimate the percentage of all board cells containing digits 0--8."""
    digit_values = np.arange(BrickState.EMPTY, BrickState.EIGHT + 1, dtype=np.int32)
    total_counts = np.zeros_like(digit_values, dtype=np.int64)
    env = MinesweeperEnv(num_rows, num_cols, num_mines, render_mode="rgb_array")

    with tqdm(
        total=num_simulations,
        desc=f"{num_rows}x{num_cols}, mines={num_mines}",
        unit="board",
    ) as pbar:
        for _ in range(num_simulations):
            env.reset()
            # Mines are excluded from digit counts; percentages still use all
            # 81 cells as the denominator below.
            non_mine_cells = env.board[env.board >= BrickState.EMPTY]
            total_counts += np.bincount(non_mine_cells, minlength=9)[:9]
            pbar.update(1)
    env.close()

    total_cells = num_simulations * num_rows * num_cols
    expected_percentages = total_counts / total_cells * 100.0
    return digit_values, expected_percentages, total_counts


def plot_digit_distributions(all_results, output_stem):
    """Draw the four digit-distribution curves with larger publication fonts."""
    colors = ["#0066FF", "#00A878", "#FF9800", "#E6005C"]

    plt.rcParams.update(
        {
            "font.family": "SimSun",
            "font.size": 14,
            "axes.labelsize": 16,
            "axes.linewidth": 0.8,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "legend.fontsize": 13,
            "axes.unicode_minus": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )

    fig, ax = plt.subplots(figsize=(7.2, 4.4), constrained_layout=True)
    category_positions = np.arange(6)
    category_labels = ["0", "1", "2", "3", "4", "≥5"]

    for result, color in zip(all_results, colors):
        _, _, mines, _, percentages, _ = result
        grouped_percentages = np.r_[percentages[:5], percentages[5:].sum()]
        ax.plot(
            category_positions,
            grouped_percentages,
            color=color,
            linestyle="-",
            linewidth=1.8,
            marker="o",
            markersize=6,
            markerfacecolor=color,
            markeredgecolor="white",
            markeredgewidth=0.8,
            zorder=3,
            label=f"{mines} mines",
        )

    ax.set_xlabel("Cell Digit", fontname="Times New Roman")
    ax.set_ylabel("Proportion of All Grid Cells (%)", fontname="Times New Roman")
    ax.set_xticks(category_positions, category_labels)
    plt.setp(ax.get_xticklabels(), fontname="Times New Roman")
    plt.setp(ax.get_yticklabels(), fontname="Times New Roman")
    ax.set_xlim(-0.2, 5.2)
    ax.set_ylim(0, 38)
    ax.set_yticks(np.arange(0, 36, 5))
    ax.grid(axis="y", which="major", color="#D9D9D9", linewidth=0.65)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(
        axis="both",
        which="both",
        direction="out",
        length=3.5,
        width=0.8,
        labelbottom=True,
        labelleft=True,
    )
    ax.legend(loc="upper right", frameon=True, fancybox=False, shadow=True,
              facecolor="white", edgecolor="#888888", framealpha=1,
              prop={"family": "Times New Roman", "size": 13},
              borderpad=0.7, handlelength=2.4, labelspacing=0.4)

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_stem.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".svg"), bbox_inches="tight")
    return fig


if __name__ == "__main__":
    configs = [(9, 9, mines) for mines in (10, 14, 18, 22)]
    num_simulations = 10_000
    random_seed = 2026
    output_stem = Path(__file__).parent / "output/digit_distribution/digit_distribution_9x9"

    np.random.seed(random_seed)
    all_results = []
    for rows, cols, mines in configs:
        print(f"\n=== Configuration: {rows}x{cols}, mines={mines} ===")
        digits, expected_pct, total_counts = simulate_digit_expectations(
            rows, cols, mines, num_simulations=num_simulations
        )
        all_results.append(
            (rows, cols, mines, digits, expected_pct, total_counts)
        )

        print(f"{'Digit':<8}{'Percentage (%)':>18}{'Total count':>18}")
        for digit, percentage, count in zip(digits, expected_pct, total_counts):
            print(f"{digit:<8}{percentage:>17.4f}%{count:>18,}")

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    with output_stem.with_suffix(".csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow(["rows", "cols", "mines", "boards", "seed", "digit", "count", "percentage_of_all_cells"])
        for rows, cols, mines, digits, percentages, counts in all_results:
            for digit, count, percentage in zip(digits, counts, percentages):
                writer.writerow([rows, cols, mines, num_simulations, random_seed,
                                 int(digit), int(count), float(percentage)])
    plot_digit_distributions(all_results, output_stem)
    print(f"\nSaved figure to {output_stem.with_suffix('.png')}")
    print(f"Saved figure to {output_stem.with_suffix('.pdf')}")
    print(f"Saved figure to {output_stem.with_suffix('.svg')}")
    plt.show()

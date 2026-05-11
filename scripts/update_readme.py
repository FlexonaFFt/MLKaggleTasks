#!/usr/bin/env python3
import csv
import os
import tempfile
from datetime import datetime
from pathlib import Path

_CACHE_DIR = Path(tempfile.gettempdir()) / "mlkaggle-cache"
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE_DIR / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE_DIR))

try:
    import matplotlib
except ModuleNotFoundError as exc:
    raise SystemExit("matplotlib is required. Install with: python3 -m pip install matplotlib") from exc

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "scripts" / "competitions.csv"
SVG_PATH = ROOT / "assets" / "progress.svg"
README_PATH = ROOT / "README.md"


def clamp(value, lo, hi):
    return max(lo, min(hi, value))


def load_rows():
    rows = []
    if not DATA_PATH.exists():
        return rows
    with DATA_PATH.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = (row.get("competition") or "").strip()
            if not name:
                continue
            try:
                rank = int(row.get("rank", "").strip())
                total = int(row.get("total", "").strip())
            except ValueError:
                continue
            if total <= 0:
                continue
            percent = (total - rank) / total * 100.0
            percent = clamp(percent, 0.0, 100.0)
            rows.append((name, rank, total, percent))
    return rows


def color_for(percent):
    if percent >= 70:
        return "#4CAF50"
    if percent >= 40:
        return "#F4B942"
    return "#E55C5C"


def build_svg(rows):
    fig, ax = plt.subplots(figsize=(10, 4.8), dpi=100)
    ax.set_facecolor("white")

    if rows:
        names = [row[0] for row in rows]
        ranks = [row[1] for row in rows]
        totals = [row[2] for row in rows]
        percents = [row[3] for row in rows]
        colors = [color_for(p) for p in percents]

        indices = list(range(1, len(rows) + 1))
        bars = ax.bar(indices, percents, color=colors, edgecolor="#1f2933", linewidth=0.5)
        ax.set_ylim(0, 100)
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.set_ylabel("% participants beaten")
        ax.set_title("Kaggle Progress: % of Participants Beaten")
        ax.grid(axis="y", linestyle="--", linewidth=0.6, color="#e5e7eb")

        ax.set_xticks(indices)
        ax.set_xticklabels([str(i) for i in indices])

        for bar, percent in zip(bars, percents):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1.5,
                f"{percent:.1f}%",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )
    else:
        ax.set_axis_off()
        ax.text(0.5, 0.5, "No data yet. Add rows to scripts/competitions.csv", ha="center", va="center")

    fig.tight_layout()
    fig.savefig(SVG_PATH, format="svg", bbox_inches="tight")
    plt.close(fig)
    return SVG_PATH


def build_table(rows):
    lines = []
    lines.append("| # | Competition | Rank | Total | % beaten |")
    lines.append("| - | - | - | - | - |")
    for idx, (name, rank, total, percent) in enumerate(rows, start=1):
        lines.append(f"| {idx} | {name} | {rank} | {total} | {percent:.1f}% |")
    return "\n".join(lines)


def update_readme(rows):
    if README_PATH.exists():
        content = README_PATH.read_text(encoding="utf-8")
    else:
        content = ""

    chart_block = """
<!-- PROGRESS-CHART:START -->
<img src="assets/progress.svg" alt="Kaggle progress chart" width="100%" />
<p><sub>Last updated: {updated_at}</sub></p>
<!-- PROGRESS-CHART:END -->
""".strip()

    table_block = """
<!-- COMPETITION-TABLE:START -->
{table}
<!-- COMPETITION-TABLE:END -->
""".strip()

    updated_at = datetime.now().strftime("%Y-%m-%d %H:%M")
    chart_block = chart_block.format(updated_at=updated_at)

    if "<!-- PROGRESS-CHART:START -->" in content:
        before = content.split("<!-- PROGRESS-CHART:START -->")[0]
        after = content.split("<!-- PROGRESS-CHART:END -->")[-1]
        content = before + chart_block + after
    else:
        if content and not content.endswith("\n"):
            content += "\n"
        content += "\n" + chart_block + "\n"

    table = build_table(rows) if rows else "| # | Competition | Rank | Total | % beaten |\n| - | - | - | - | - |"
    table_block = table_block.format(table=table)

    if "<!-- COMPETITION-TABLE:START -->" in content:
        before = content.split("<!-- COMPETITION-TABLE:START -->")[0]
        after = content.split("<!-- COMPETITION-TABLE:END -->")[-1]
        content = before + table_block + after
    else:
        if content and not content.endswith("\n"):
            content += "\n"
        content += "\n" + table_block + "\n"

    README_PATH.write_text(content, encoding="utf-8")


def main():
    rows = load_rows()
    build_svg(rows)
    update_readme(rows)


if __name__ == "__main__":
    main()

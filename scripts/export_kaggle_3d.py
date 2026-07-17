#!/usr/bin/env python3
"""Export Kaggle submission history and an interactive 3D chart."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from fetch_kaggle_stats import (
    KaggleError,
    SCORE_COLUMNS,
    fetch_leaderboard,
    first_value,
    normalize_slug,
    parse_float,
)

ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "scripts" / "kaggle_submissions.csv"
HTML_PATH = ROOT / "assets" / "kaggle_3d.html"
PNG_PATH = ROOT / "assets" / "kaggle_3d.png"
PAGES_CHART_PATH = ROOT / "docs" / "kaggle_3d.html"

FIELDNAMES = [
    "competition_slug",
    "competition_title",
    "competition_start",
    "competition_deadline",
    "competition_duration_days",
    "submission_id",
    "file_name",
    "submitted_at",
    "description",
    "status",
    "public_score",
    "private_score",
    "attempt",
    "attempt_percent",
    "time_percent",
    "score_percent",
    "leaderboard_rank",
    "leaderboard_team_count",
    "leaderboard_best_score",
    "leaderboard_worst_score",
]


def parse_date(value: str | datetime) -> datetime | None:
    if isinstance(value, datetime):
        return value.replace(tzinfo=None)
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).replace(tzinfo=None)
    except (AttributeError, ValueError):
        return None


def format_percent(value: float | None) -> str:
    return "" if value is None else f"{value:.4f}"


def leaderboard_scores(rows: list[dict[str, str]]) -> list[tuple[int, float]]:
    ranked = []
    for row in rows:
        rank = parse_float(row.get("Rank") or row.get("rank") or "")
        score = parse_float(first_value(row, SCORE_COLUMNS))
        if rank is not None and score is not None:
            ranked.append((int(rank), score))
    ranked.sort()
    return ranked


def leaderboard_percent(score: str, leaderboard: list[tuple[int, float]]) -> tuple[float | None, int | None]:
    value = parse_float(score)
    if value is None or not leaderboard:
        return None, None
    best_score, worst_score = leaderboard[0][1], leaderboard[-1][1]
    higher_is_better = best_score > worst_score
    better = sum(
        other_score > value if higher_is_better else other_score < value
        for _, other_score in leaderboard
    )
    total = len(leaderboard)
    rank = min(better + 1, total)
    percent = 100.0 if total == 1 else (total - rank) / (total - 1) * 100.0
    return percent, rank


def authenticated_api():
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Kaggle package is required. Run this with .venv/bin/python after installing kaggle."
        ) from exc

    api = KaggleApi()
    api.authenticate()
    return api


def entered_competitions(api):
    page_token = ""
    competitions = []
    while True:
        response = api.competitions_list(
            group="entered", page=-1, page_size=200, page_token=page_token
        )
        competitions.extend(response.competitions or [])
        page_token = response.next_page_token or ""
        if not page_token:
            return competitions


def submissions(api, slug: str) -> list[dict[str, str]]:
    from kagglesdk.competitions.types.competition_api_service import ApiListSubmissionsRequest

    page_token = ""
    rows = []
    while True:
        request = ApiListSubmissionsRequest()
        request.competition_name = slug
        request.page = -1
        request.page_size = 200
        request.page_token = page_token
        with api.build_kaggle_client() as client:
            response = client.competitions.competition_api_client.list_submissions(request)
        rows.extend(item.__class__.to_dict(item) for item in response.submissions or [])
        page_token = response.next_page_token or ""
        if not page_token:
            return rows


def build_rows() -> list[dict[str, str]]:
    rows = []
    api = authenticated_api()
    for competition in entered_competitions(api):
        slug = normalize_slug(competition.ref)
        start = parse_date(competition.enabled_date)
        deadline = parse_date(competition.deadline)
        try:
            competition_submissions = submissions(api, slug)
            leaderboard = leaderboard_scores(fetch_leaderboard(slug))
        except KaggleError as exc:
            print(f"{slug}: skipped ({exc})", flush=True)
            continue

        dated = []
        for submission in competition_submissions:
            submitted_at = parse_date(str(submission.get("date", "")))
            if submitted_at:
                dated.append((submitted_at, submission))
        dated.sort(key=lambda item: item[0])
        total_attempts = len(dated)
        duration_days = (deadline - start).total_seconds() / 86400 if start and deadline else None

        for attempt, (submitted_at, submission) in enumerate(dated, start=1):
            if start and deadline and deadline > start:
                time_value = (submitted_at - start).total_seconds() / (deadline - start).total_seconds() * 100
                time_percent = max(0.0, min(100.0, time_value))
            else:
                time_percent = None
            public_score = str(submission.get("publicScore", "")).strip()
            score_value, rank = leaderboard_percent(public_score, leaderboard)
            rows.append(
                {
                    "competition_slug": slug,
                    "competition_title": competition.title,
                    "competition_start": start.isoformat(sep=" ") if start else "",
                    "competition_deadline": deadline.isoformat(sep=" ") if deadline else "",
                    "competition_duration_days": format_percent(duration_days),
                    "submission_id": str(submission.get("ref", "")).strip(),
                    "file_name": str(submission.get("fileName", "")).strip(),
                    "submitted_at": submitted_at.isoformat(sep=" "),
                    "description": str(submission.get("description", "")).strip(),
                    "status": str(submission.get("status", "")).strip(),
                    "public_score": public_score,
                    "private_score": str(submission.get("privateScore", "")).strip(),
                    "attempt": str(attempt),
                    "attempt_percent": format_percent(attempt / total_attempts * 100),
                    "time_percent": format_percent(time_percent),
                    "score_percent": format_percent(score_value),
                    "leaderboard_rank": str(rank or ""),
                    "leaderboard_team_count": str(len(leaderboard)),
                    "leaderboard_best_score": format_percent(leaderboard[0][1] if leaderboard else None),
                    "leaderboard_worst_score": format_percent(leaderboard[-1][1] if leaderboard else None),
                }
            )
        print(f"{slug}: {total_attempts} submissions", flush=True)
    return rows


def write_csv(rows: list[dict[str, str]]) -> None:
    with CSV_PATH.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDNAMES, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def chart_groups(rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if all(row[name] for name in ("attempt_percent", "time_percent", "score_percent")):
            groups[row["competition_title"]].append(row)
    return {title: points for title, points in groups.items() if len(points) >= 2}


def write_chart(rows: list[dict[str, str]]) -> None:
    # ponytail: use Plotly CDN; bundle it only when offline viewing is required.
    groups = chart_groups(rows)

    traces = []
    for title, points in sorted(groups.items()):
        points.sort(key=lambda row: int(row["attempt"]))
        traces.append(
            {
                "type": "scatter3d",
                "mode": "lines+markers",
                "name": title,
                "x": [float(row["attempt_percent"]) for row in points],
                "y": [float(row["time_percent"]) for row in points],
                "z": [float(row["score_percent"]) for row in points],
                "text": [
                    f"<b>{row['competition_title']}</b><br>"
                    f"Attempt {row['attempt']} ({float(row['attempt_percent']):.1f}%)<br>"
                    f"Submitted: {row['submitted_at']}<br>"
                    f"Public score: {row['public_score']}<br>"
                    f"Estimated leaderboard rank: {row['leaderboard_rank']} / {row['leaderboard_team_count']}<br>"
                    f"Teams beaten: {float(row['score_percent']):.1f}%"
                    for row in points
                ],
                "hovertemplate": "%{text}<extra></extra>",
                "marker": {"size": 3},
                "line": {"width": 4},
            }
        )

    html = f"""<!doctype html>
<html lang=\"en\"><head><meta charset=\"utf-8\"><title>Kaggle 3D submissions</title>
<script src=\"https://cdn.plot.ly/plotly-3.1.0.min.js\"></script>
<style>html,body,#chart{{height:100%;margin:0;font-family:system-ui,sans-serif}}</style></head>
<body><div id=\"chart\"></div><script>
const traces = {json.dumps(traces, ensure_ascii=False)};
Plotly.newPlot('chart', traces, {{
  title: 'Kaggle submission trajectories',
  scene: {{
    xaxis: {{title: 'Attempts used (%)', range: [0, 100]}},
    yaxis: {{title: 'Competition time used (%)', range: [0, 100]}},
    zaxis: {{title: 'Teams beaten (%)', range: [0, 100]}},
    aspectmode: 'cube'
  }},
  updatemenus: [{{
    type: 'buttons', direction: 'right', x: 0.01, y: 1.12,
    buttons: [
      {{label: 'Full Z (0–100%)', method: 'relayout', args: [{{'scene.zaxis.range': [0, 100]}}]}},
      {{label: 'Focus Z (70–100%)', method: 'relayout', args: [{{'scene.zaxis.range': [70, 100]}}]}}
    ]
  }}],
  legend: {{itemsizing: 'constant'}}, margin: {{l: 0, r: 0, b: 0, t: 70}}
}}, {{responsive: true}});
</script></body></html>"""
    HTML_PATH.write_text(html, encoding="utf-8")
    PAGES_CHART_PATH.parent.mkdir(exist_ok=True)
    PAGES_CHART_PATH.write_text(html, encoding="utf-8")


def write_readme_chart(rows: list[dict[str, str]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure = plt.figure(figsize=(12, 7), dpi=160)
    axis = figure.add_subplot(projection="3d")
    for points in chart_groups(rows).values():
        points.sort(key=lambda row: int(row["attempt"]))
        axis.plot(
            [float(row["attempt_percent"]) for row in points],
            [float(row["time_percent"]) for row in points],
            [float(row["score_percent"]) for row in points],
            marker="o",
            markersize=3,
            linewidth=2,
        )
    axis.set(
        xlim=(0, 100), ylim=(0, 100), zlim=(0, 100),
        xlabel="Attempts used (%)", ylabel="Competition time used (%)", zlabel="Teams beaten (%)",
    )
    axis.view_init(elev=24, azim=-62)
    figure.tight_layout()
    figure.savefig(PNG_PATH, bbox_inches="tight", pad_inches=0.2)
    plt.close(figure)


def self_check() -> None:
    assert leaderboard_percent("8", [(1, 10), (2, 5), (3, 0)]) == (50.0, 2)
    assert leaderboard_percent("2", [(1, 0), (2, 5), (3, 10)]) == (50.0, 2)
    assert leaderboard_percent("12", [(1, 10), (2, 5), (3, 0)]) == (100.0, 1)


def main() -> int:
    parser = argparse.ArgumentParser(description="Export Kaggle submissions and a 3D chart.")
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        self_check()
        return 0
    rows = build_rows()
    write_csv(rows)
    write_chart(rows)
    write_readme_chart(rows)
    print(f"wrote {len(rows)} rows to {CSV_PATH}")
    print(f"wrote 3D chart to {HTML_PATH}")
    print(f"wrote GitHub Pages chart to {PAGES_CHART_PATH}")
    print(f"wrote README chart to {PNG_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

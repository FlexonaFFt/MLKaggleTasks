#!/usr/bin/env python3
"""Update competition stats from the Kaggle CLI.

Requirements:
  - Install the official Kaggle CLI: python3 -m pip install kaggle
  - Configure ~/.kaggle/kaggle.json
  - Optionally fill the `slug` column in scripts/competitions.csv

If a slug is missing, the script searches Kaggle by title and fills it when the
match is confident. It keeps manual values when Kaggle data is unavailable.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from difflib import SequenceMatcher
from datetime import datetime
from pathlib import Path
from typing import Iterable
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "scripts" / "competitions.csv"
UPDATE_README = ROOT / "scripts" / "update_readme.py"
KAGGLE_TIMEOUT_SECONDS = int(os.environ.get("KAGGLE_TIMEOUT_SECONDS", "45"))

FIELDNAMES = [
    "title",
    "slug",
    "project_path",
    "rank",
    "total",
    "best_score",
    "metric",
    "score_order",
    "source",
    "last_synced",
]

SCORE_COLUMNS = ("publicScore", "public_score", "score", "Score")
RANK_COLUMNS = ("rank", "Rank", "position", "Position")
TEAM_COLUMNS = ("teamName", "TeamName", "team_name", "Team", "team")
TITLE_COLUMNS = ("title", "Title", "competitionTitle", "CompetitionTitle")
SLUG_COLUMNS = ("ref", "Ref", "slug", "Slug", "competition", "Competition")
TOTAL_COLUMNS = ("teamCount", "TeamCount", "totalTeams", "TotalTeams", "teams", "Teams")
METRIC_COLUMNS = ("evaluationMetric", "EvaluationMetric", "metric", "Metric")


class KaggleError(RuntimeError):
    pass


def read_rows() -> list[dict[str, str]]:
    if not DATA_PATH.exists():
        raise KaggleError(f"Missing data file: {DATA_PATH}")
    with DATA_PATH.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = []
        for row in reader:
            normalized = {name: (row.get(name) or "").strip() for name in FIELDNAMES}
            if not normalized["title"]:
                normalized["title"] = (row.get("competition") or "").strip()
            rows.append(normalized)
        return rows


def write_rows(rows: Iterable[dict[str, str]]) -> None:
    with DATA_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in FIELDNAMES})


def kaggle_command() -> list[str] | None:
    for base in (Path(sys.executable).parent, Path(sys.prefix) / "bin"):
        venv_kaggle = base / "kaggle"
        if venv_kaggle.exists():
            return [str(venv_kaggle)]
    kaggle = shutil.which("kaggle")
    if kaggle:
        return [kaggle]
    return None


def row_key(row: dict[str, str]) -> str:
    slug = (row.get("slug") or "").strip()
    if slug:
        return f"slug:{slug}"
    return f"title:{normalize_text(row.get('title', ''))}"


def run_kaggle(args: list[str], cwd: Path | None = None) -> str:
    command = kaggle_command()
    if command is None:
        raise KaggleError(
            "Kaggle CLI is not installed. Install it with: python3 -m pip install kaggle"
        )
    try:
        proc = subprocess.run(
            [*command, *args],
            cwd=cwd,
            check=False,
            text=True,
            capture_output=True,
            timeout=KAGGLE_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as exc:
        raise KaggleError(
            f"kaggle {' '.join(args)} timed out after {KAGGLE_TIMEOUT_SECONDS}s"
        ) from exc
    if proc.returncode != 0:
        message = (proc.stderr or proc.stdout).strip()
        raise KaggleError(message or f"kaggle {' '.join(args)} failed")
    return proc.stdout


def read_csv_text(text: str) -> list[dict[str, str]]:
    lines = [line for line in text.splitlines() if line.strip()]
    if not lines:
        return []
    return list(csv.DictReader(lines))


def normalize_text(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


def similarity(left: str, right: str) -> float:
    left_norm = normalize_text(left)
    right_norm = normalize_text(right)
    if not left_norm or not right_norm:
        return 0.0
    if left_norm == right_norm:
        return 1.0
    if left_norm in right_norm or right_norm in left_norm:
        return 0.92
    return SequenceMatcher(None, left_norm, right_norm).ratio()


def first_value(row: dict[str, str], candidates: Iterable[str]) -> str:
    for key in candidates:
        value = (row.get(key) or "").strip()
        if value:
            return value
    return ""


def parse_profile_name(profile: str) -> str:
    profile = profile.strip()
    if not profile:
        return ""
    if "://" in profile:
        parsed = urlparse(profile)
        return parsed.path.strip("/").split("/")[0]
    return profile.rstrip("/").split("/")[-1]


def normalize_slug(value: str) -> str:
    value = value.strip()
    if not value:
        return ""
    if "://" in value:
        parsed = urlparse(value)
        parts = parsed.path.strip("/").split("/")
        if "competitions" in parts:
            idx = parts.index("competitions")
            if idx + 1 < len(parts):
                return parts[idx + 1]
        return parts[-1] if parts else ""
    return value.strip("/")


def title_from_slug(slug: str) -> str:
    return normalize_slug(slug).replace("-", " ").title()


def list_competition_candidates(title: str) -> list[dict[str, str]]:
    text = run_kaggle(["competitions", "list", "-s", title, "-v"])
    return read_csv_text(text)


def list_entered_competitions(max_pages: int) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for page in range(1, max_pages + 1):
        text = run_kaggle(["competitions", "list", "--group", "entered", "-p", str(page), "-v"])
        page_rows = read_csv_text(text)
        if not page_rows:
            break
        rows.extend(page_rows)
    return rows


def row_from_competition(candidate: dict[str, str]) -> dict[str, str] | None:
    slug = normalize_slug(first_value(candidate, SLUG_COLUMNS))
    title = first_value(candidate, TITLE_COLUMNS) or title_from_slug(slug)
    if not slug or not title:
        return None
    total = first_value(candidate, TOTAL_COLUMNS)
    metric = first_value(candidate, METRIC_COLUMNS)
    return {
        "title": title,
        "slug": slug,
        "project_path": "",
        "rank": "",
        "total": total,
        "best_score": "",
        "metric": metric,
        "score_order": "",
        "source": "kaggle",
        "last_synced": "",
    }


def merge_discovered_rows(
    existing_rows: list[dict[str, str]],
    discovered_rows: list[dict[str, str]],
) -> tuple[list[dict[str, str]], int]:
    merged = list(existing_rows)
    seen = {row_key(row) for row in merged}
    added = 0

    for discovered in discovered_rows:
        row = row_from_competition(discovered)
        if row is None:
            continue
        key = row_key(row)
        if key in seen:
            continue
        seen.add(key)
        merged.append(row)
        added += 1

    return merged, added


def resolve_slug(title: str, min_confidence: float) -> tuple[str, float, list[str]]:
    candidates = list_competition_candidates(title)
    scored: list[tuple[float, str, str]] = []

    for candidate in candidates:
        candidate_title = first_value(candidate, TITLE_COLUMNS)
        slug = normalize_slug(first_value(candidate, SLUG_COLUMNS))
        if not slug:
            continue
        score = max(similarity(title, candidate_title), similarity(title, slug))
        scored.append((score, slug, candidate_title or slug))

    scored.sort(reverse=True, key=lambda item: item[0])
    if not scored:
        return "", 0.0, []

    best_score, best_slug, _ = scored[0]
    labels = [f"{slug} ({label}, confidence={score:.2f})" for score, slug, label in scored[:3]]
    if best_score >= min_confidence:
        return best_slug, best_score, labels
    return "", best_score, labels


def parse_float(value: str) -> float | None:
    value = (value or "").strip()
    if not value or value.lower() in {"none", "nan", "-"}:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def choose_best_score(submissions: list[dict[str, str]], score_order: str) -> str:
    scored = []
    for submission in submissions:
        score_text = first_value(submission, SCORE_COLUMNS)
        score = parse_float(score_text)
        if score is not None:
            scored.append((score, score_text))
    if not scored:
        return ""
    if score_order == "lower":
        return min(scored, key=lambda item: item[0])[1]
    return max(scored, key=lambda item: item[0])[1]


def fetch_submissions(slug: str, score_order: str) -> str:
    text = run_kaggle(["competitions", "submissions", slug, "-v", "-q"])
    submissions = read_csv_text(text)
    return choose_best_score(submissions, score_order)


def find_downloaded_leaderboard(folder: Path) -> Path | None:
    files = sorted(
        [*folder.glob("*.csv"), *folder.glob("*.zip")],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return files[0] if files else None


def read_leaderboard_file(path: Path) -> list[dict[str, str]]:
    if path.suffix == ".zip":
        with zipfile.ZipFile(path) as archive:
            csv_names = [name for name in archive.namelist() if name.endswith(".csv")]
            if not csv_names:
                return []
            with archive.open(csv_names[0]) as f:
                text = f.read().decode("utf-8-sig").splitlines()
                return list(csv.DictReader(text))
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def fetch_leaderboard(slug: str) -> list[dict[str, str]]:
    with tempfile.TemporaryDirectory(prefix="kaggle-leaderboard-") as tmp:
        folder = Path(tmp)
        run_kaggle(["competitions", "leaderboard", slug, "-d", "-p", str(folder), "-q"])
        path = find_downloaded_leaderboard(folder)
        if path is None:
            return []
        return read_leaderboard_file(path)


def score_matches(left: str, right: str) -> bool:
    left_num = parse_float(left)
    right_num = parse_float(right)
    if left_num is None or right_num is None:
        return left.strip() == right.strip()
    return abs(left_num - right_num) <= 1e-12


def infer_rank(
    leaderboard: list[dict[str, str]],
    best_score: str,
    team_name: str,
) -> tuple[str, str]:
    if not leaderboard:
        return "", ""

    total = str(len(leaderboard))
    normalized_team = team_name.strip().lower()

    for row in leaderboard:
        rank = first_value(row, RANK_COLUMNS)
        team = first_value(row, TEAM_COLUMNS).lower()
        score = first_value(row, SCORE_COLUMNS)
        if normalized_team and team == normalized_team and rank:
            return rank, total
        if best_score and score_matches(score, best_score) and rank:
            return rank, total

    return "", total


def sync_row(
    row: dict[str, str],
    team_name: str,
    resolve_missing_slugs: bool,
    min_slug_confidence: float,
) -> tuple[dict[str, str], str]:
    slug = normalize_slug(row.get("slug", ""))
    if slug and row.get("slug") != slug:
        row["slug"] = slug
    if not slug:
        if not resolve_missing_slugs:
            return row, "skipped: no slug"
        slug, confidence, candidates = resolve_slug(row.get("title", ""), min_slug_confidence)
        if not slug:
            suffix = f"; candidates: {', '.join(candidates)}" if candidates else ""
            return row, f"skipped: slug not resolved (confidence={confidence:.2f}){suffix}"
        row["slug"] = slug

    score_order = (row.get("score_order") or "higher").strip().lower()
    if score_order not in {"higher", "lower"}:
        score_order = "higher"

    best_score = fetch_submissions(slug, score_order) or row.get("best_score", "")
    leaderboard = fetch_leaderboard(slug)
    rank, total = infer_rank(leaderboard, best_score, team_name)

    if best_score:
        row["best_score"] = best_score
    if rank:
        row["rank"] = rank
    if total:
        row["total"] = total
    row["source"] = "kaggle"
    row["last_synced"] = datetime.now().strftime("%Y-%m-%d %H:%M")
    return row, f"updated: {slug}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch Kaggle competition stats.")
    parser.add_argument(
        "--profile",
        default="",
        help="Kaggle profile URL or username. Used as default team name.",
    )
    parser.add_argument(
        "--team-name",
        default="",
        help="Optional Kaggle team name for exact leaderboard rank matching.",
    )
    parser.add_argument(
        "--discover-entered",
        action="store_true",
        help="Add competitions from `kaggle competitions list --group entered` before syncing.",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=10,
        help="Maximum pages to read when discovering entered competitions.",
    )
    parser.add_argument(
        "--no-resolve-slugs",
        action="store_true",
        help="Do not search Kaggle for missing competition slugs.",
    )
    parser.add_argument(
        "--min-slug-confidence",
        type=float,
        default=0.74,
        help="Minimum confidence for accepting an auto-resolved slug.",
    )
    parser.add_argument(
        "--update-readme",
        action="store_true",
        help="Run scripts/update_readme.py after syncing stats.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero when an individual competition fails to sync.",
    )
    args = parser.parse_args()

    profile_name = parse_profile_name(args.profile)
    team_name = args.team_name or profile_name
    rows = read_rows()

    needs_kaggle = (
        args.discover_entered
        or not args.no_resolve_slugs
        or any((row.get("slug") or "").strip() for row in rows)
    )
    if needs_kaggle and kaggle_command() is None:
        print("Kaggle CLI is not installed. Install it with: python3 -m pip install kaggle")
        return 1

    if args.discover_entered:
        try:
            discovered = list_entered_competitions(args.max_pages)
        except KaggleError as exc:
            print(f"discover entered competitions: error: {exc}", flush=True)
            discovered = []
            had_discovery_error = True
        else:
            rows, added = merge_discovered_rows(rows, discovered)
            print(f"discover entered competitions: added {added}, found {len(discovered)}", flush=True)
            had_discovery_error = False
    else:
        had_discovery_error = False

    updated_rows = []
    had_error = had_discovery_error

    for row in rows:
        title = row.get("title") or row.get("slug") or "<untitled>"
        print(f"{title}: syncing...", flush=True)
        try:
            updated, status = sync_row(
                row,
                team_name=team_name,
                resolve_missing_slugs=not args.no_resolve_slugs,
                min_slug_confidence=args.min_slug_confidence,
            )
        except KaggleError as exc:
            updated = row
            status = f"error: {exc}"
            if args.strict:
                had_error = True
        updated_rows.append(updated)
        print(f"{title}: {status}", flush=True)

    write_rows(updated_rows)

    if args.update_readme:
        proc = subprocess.run([sys.executable, str(UPDATE_README)], cwd=ROOT, check=False)
        if proc.returncode != 0:
            had_error = True

    return 1 if had_error else 0


if __name__ == "__main__":
    raise SystemExit(main())

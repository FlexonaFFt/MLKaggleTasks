from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Iterable

import numpy as np
import pandas as pd

try:
    from sklearn.linear_model import Ridge
except ModuleNotFoundError:
    Ridge = None


@dataclass
class BlenderConfig:
    alpha: float = 1.0


class RidgeBlender:
    def __init__(self, cfg: BlenderConfig):
        if Ridge is None:
            raise ModuleNotFoundError("scikit-learn is required to use RidgeBlender.")
        self.cfg = cfg
        self.model = Ridge(alpha=cfg.alpha)

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)


@dataclass
class SubmissionBlendConfig:
    dataset_dir: Path
    output_dir: Path
    id_col: str = "id"
    target_col: str = "PitNextLap"
    clip_low: float = 1e-7
    clip_high: float = 1 - 1e-7
    public_core_count: int = 6
    public_diverse_score: float = 0.95381
    support_score: float = 0.95305
    variant_weight: float = 0.005
    save_default_submission: bool = True
    public_core_min_score: float = 0.95388
    own_variant_names: tuple[str, ...] = ("shallow", "reg")


@dataclass
class BlendCandidate:
    key: str
    group: str
    filename: str
    recipe: str
    weights: dict[str, float]
    try_order: int
    note: str = ""


class PitStopSubmissionBlender:
    """Small, opinionated blender for the F1 pit-stop submission workflow."""

    score_pattern = re.compile(r"^0\.\d+$")

    def __init__(self, cfg: SubmissionBlendConfig):
        self.cfg = cfg
        self.dataset_dir = Path(cfg.dataset_dir)
        self.output_dir = Path(cfg.output_dir)
        self.submission_dir = self.output_dir / "submissions"
        self.report_dir = self.output_dir / "reports"
        self.predictions: dict[str, np.ndarray] = {}
        self.meta = pd.DataFrame()
        self.base_ids: pd.Series | None = None
        self.groups: dict[str, list[str]] = {}
        self.blends: dict[str, np.ndarray] = {}

    def run(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        self.load_dataset()
        group_summary = self.build_groups()
        blend_summary = self.create_blends()
        comparison = self.compare_inputs()
        return group_summary, blend_summary, comparison

    def load_dataset(self) -> pd.DataFrame:
        records = []
        self.predictions = {}
        self.base_ids = None

        for source in ("public", "ours"):
            folder = self.dataset_dir / source
            if not folder.exists():
                continue
            for path in sorted(folder.rglob("*.csv")):
                record = self._load_submission(path, source)
                if record is not None:
                    records.append(record)

        if not records:
            raise FileNotFoundError(f"No valid submission CSV files found in {self.dataset_dir}")

        self.meta = pd.DataFrame(records).sort_values(
            ["source", "public_score", "name"], ascending=[True, False, True]
        ).reset_index(drop=True)
        self.report_dir.mkdir(parents=True, exist_ok=True)
        self.meta.to_csv(self.report_dir / "inputs.csv", index=False)
        return self.meta

    def build_groups(self) -> pd.DataFrame:
        public_scored = self.meta[self.meta["source"].eq("public") & self.meta["public_score"].notna()].copy()
        public_scored = public_scored.sort_values("public_score", ascending=False)

        top_external = self.meta[self.meta["group"].eq("top_external")]["name"].tolist()
        core = self.meta[self.meta["group"].eq("core")].sort_values("public_score", ascending=False)["name"].head(
            self.cfg.public_core_count
        ).tolist()
        if not core:
            core = public_scored[public_scored["public_score"].ge(self.cfg.public_core_min_score)].head(
                self.cfg.public_core_count
            )["name"].tolist()
        if len(core) < min(self.cfg.public_core_count, len(public_scored)) and not top_external:
            core = public_scored.head(self.cfg.public_core_count)["name"].tolist()

        diverse = self.meta[self.meta["group"].eq("diverse")]["name"].tolist() or self._public_names_by_score(self.cfg.public_diverse_score)
        support = self.meta[self.meta["group"].eq("support")]["name"].tolist() or self._public_names_by_score(self.cfg.support_score)
        own_base = self.meta[(self.meta["source"].eq("ours")) & (self.meta["group"].eq("main"))]["name"].tolist()
        own_variants = self.meta[(self.meta["source"].eq("ours")) & (self.meta["group"].eq("variants"))]["name"].tolist()

        self.groups = {
            "top_external": top_external,
            "public_core": core,
            "public_diverse": diverse,
            "public_support": support,
            "own_base": own_base,
            "own_variants": own_variants,
        }

        rows = [
            {"group": group, "count": len(names), "members": ", ".join(names) or "-"}
            for group, names in self.groups.items()
        ]
        group_summary = pd.DataFrame(rows)
        group_summary.to_csv(self.report_dir / "groups.csv", index=False)
        return group_summary

    def create_blends(self) -> pd.DataFrame:
        self._require_group("public_core")
        self._require_group("public_diverse")

        core = self._mean(self.groups["public_core"])
        diverse = self.predictions[self.groups["public_diverse"][0]]
        support = self.predictions[self.groups["public_support"][0]] if self.groups["public_support"] else None
        shallow = self.predictions.get("ours_shallow")
        reg = self.predictions.get("ours_reg")

        top_external_pred = self._mean(self.groups["top_external"]) if self.groups.get("top_external") else None
        b10 = self._clip(0.95 * core + 0.05 * diverse)
        self.blends = {"b10": b10}

        candidates = [
            BlendCandidate(
                key="b10",
                group="best",
                filename="b10.csv",
                recipe="0.950 core + 0.050 95381",
                weights={"core": 0.95, "95381": 0.05},
                try_order=1,
                note="Current best public-LB recipe",
            )
        ]

        order = 2
        if top_external_pred is not None:
            self.blends["tx"] = top_external_pred
            candidates.append(
                BlendCandidate(
                    key="tx",
                    group="top_external",
                    filename="tx.csv",
                    recipe="mean(0.95409_a, 0.95409_b)",
                    weights={"top_external": 1.0},
                    try_order=order,
                    note="Mean of new 0.95409 external submissions",
                )
            )
            order += 1
            pred = self._clip(0.90 * top_external_pred + 0.10 * b10)
            self.blends["tb"] = pred
            candidates.append(
                BlendCandidate(
                    key="tb",
                    group="top_external",
                    filename="tb.csv",
                    recipe="0.900 top_external + 0.100 b10",
                    weights={"top_external": 0.90, "b10": 0.10},
                    try_order=order,
                    note="Conservative blend of top external mean with b10",
                )
            )
            order += 1

        for variant_name, variant_pred, label in [
            ("vs", shallow, "shallow"),
            ("vr", reg, "reg"),
        ]:
            if variant_pred is None:
                continue
            pred = self._clip((0.95 - self.cfg.variant_weight) * core + 0.05 * diverse + self.cfg.variant_weight * variant_pred)
            self.blends[variant_name] = pred
            candidates.append(
                BlendCandidate(
                    key=variant_name,
                    group="catboost_variant",
                    filename=f"{variant_name}.csv",
                    recipe=f"{0.95 - self.cfg.variant_weight:.3f} core + 0.050 95381 + {self.cfg.variant_weight:.3f} {label}",
                    weights={"core": 0.95 - self.cfg.variant_weight, "95381": 0.05, label: self.cfg.variant_weight},
                    try_order=order,
                    note=f"Tiny {label} CatBoost injection",
                )
            )
            order += 1

        if shallow is not None and reg is not None:
            half = self.cfg.variant_weight / 2
            pred = self._clip((0.95 - self.cfg.variant_weight) * core + 0.05 * diverse + half * shallow + half * reg)
            self.blends["vm"] = pred
            candidates.append(
                BlendCandidate(
                    key="vm",
                    group="catboost_variant",
                    filename="vm.csv",
                    recipe=f"{0.95 - self.cfg.variant_weight:.3f} core + 0.050 95381 + {half:.4f} shallow + {half:.4f} reg",
                    weights={"core": 0.95 - self.cfg.variant_weight, "95381": 0.05, "shallow": half, "reg": half},
                    try_order=order,
                    note="Split tiny CatBoost variant injection",
                )
            )
            order += 1

        if support is not None:
            support_weight = 0.005
            pred = self._clip((0.95 - support_weight) * core + 0.05 * diverse + support_weight * support)
            self.blends["ds"] = pred
            candidates.append(
                BlendCandidate(
                    key="ds",
                    group="public_support",
                    filename="ds.csv",
                    recipe=f"{0.95 - support_weight:.3f} core + 0.050 95381 + {support_weight:.3f} 95305",
                    weights={"core": 0.95 - support_weight, "95381": 0.05, "95305": support_weight},
                    try_order=order,
                    note="Tiny support-public injection",
                )
            )

        rows = []
        for candidate in candidates:
            pred = self.blends[candidate.key]
            folder = self.submission_dir / candidate.group
            path = folder / candidate.filename
            self._save_submission(path, pred)
            rows.append(self._candidate_summary(candidate, path, pred, b10, core))

        if self.cfg.save_default_submission:
            self._save_submission(self.output_dir / "submission.csv", b10)

        summary = pd.DataFrame(rows).sort_values("try_order").reset_index(drop=True)
        summary.to_csv(self.report_dir / "blend_summary.csv", index=False)
        self._write_readable_summary(summary)
        return summary

    def compare_inputs(self) -> pd.DataFrame:
        if not self.blends:
            raise RuntimeError("Create blends before comparing inputs.")
        b10 = self.blends["b10"]
        rows = []
        for name, pred in self.predictions.items():
            rows.append(
                {
                    "name": name,
                    "source": self._source_for(name),
                    "public_score": self._score_for(name),
                    "corr_b10": np.corrcoef(pred, b10)[0, 1],
                    "delta_b10": np.abs(pred - b10).mean(),
                    "mean": pred.mean(),
                    "std": pred.std(),
                }
            )
        comparison = pd.DataFrame(rows).sort_values(["source", "public_score", "name"], ascending=[True, False, True])
        comparison.to_csv(self.report_dir / "input_comparison.csv", index=False)
        return comparison

    def display_blend_summary(self, summary: pd.DataFrame | None = None) -> pd.DataFrame:
        if summary is None:
            summary = pd.read_csv(self.report_dir / "blend_summary.csv")
        columns = ["try_order", "key", "group", "file", "recipe", "corr_b10", "delta_b10", "delta_core", "note"]
        return summary[columns].round({"corr_b10": 6, "delta_b10": 6, "delta_core": 6})

    def _load_submission(self, path: Path, source: str) -> dict | None:
        try:
            df = pd.read_csv(path)
        except Exception:
            return None
        if self.cfg.id_col not in df.columns:
            return None
        target_candidates = [col for col in df.columns if col != self.cfg.id_col]
        if self.cfg.target_col in df.columns:
            target_col = self.cfg.target_col
        elif len(target_candidates) == 1:
            target_col = target_candidates[0]
        else:
            return None
        df = df[[self.cfg.id_col, target_col]].rename(columns={target_col: self.cfg.target_col})
        df[self.cfg.target_col] = pd.to_numeric(df[self.cfg.target_col], errors="coerce")
        if df[self.cfg.target_col].isna().any() or df[self.cfg.id_col].duplicated().any():
            return None
        if self.base_ids is None:
            self.base_ids = df[self.cfg.id_col].copy()
        elif not self.base_ids.equals(df[self.cfg.id_col]):
            return None

        name = self._submission_name(path, source)
        pred = self._clip(df[self.cfg.target_col].to_numpy(dtype=float))
        self.predictions[name] = pred
        return {
            "name": name,
            "source": source,
            "public_score": self._public_score(path) if source == "public" else np.nan,
            "group": path.parent.name,
            "rows": len(df),
            "mean": pred.mean(),
            "std": pred.std(),
            "file": str(path.relative_to(self.dataset_dir)) if path.is_relative_to(self.dataset_dir) else str(path),
        }

    def _submission_name(self, path: Path, source: str) -> str:
        safe = path.stem.replace(".", "_").replace("-", "_")
        return f"{source}_{safe}"

    def _public_score(self, path: Path) -> float:
        return float(path.stem) if self.score_pattern.match(path.stem) else np.nan

    def _public_names_by_score(self, score: float, atol: float = 1e-8) -> list[str]:
        if self.meta.empty:
            return []
        matches = self.meta[
            self.meta["source"].eq("public")
            & np.isclose(self.meta["public_score"].astype(float), score, atol=atol, equal_nan=False)
        ]
        return matches["name"].tolist()

    def _require_group(self, group: str) -> None:
        if not self.groups.get(group):
            raise RuntimeError(f"Required blend group is empty: {group}")

    def _mean(self, names: Iterable[str]) -> np.ndarray:
        names = list(names)
        if not names:
            raise ValueError("Cannot average an empty prediction group.")
        return self._clip(np.vstack([self.predictions[name] for name in names]).mean(axis=0))

    def _clip(self, pred: np.ndarray) -> np.ndarray:
        return np.clip(np.asarray(pred, dtype=float), self.cfg.clip_low, self.cfg.clip_high)

    def _save_submission(self, path: Path, pred: np.ndarray) -> None:
        if self.base_ids is None:
            raise RuntimeError("No base ids loaded.")
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({self.cfg.id_col: self.base_ids.values, self.cfg.target_col: self._clip(pred)}).to_csv(path, index=False)

    def _candidate_summary(self, candidate: BlendCandidate, path: Path, pred: np.ndarray, b10: np.ndarray, core: np.ndarray) -> dict:
        return {
            "try_order": candidate.try_order,
            "key": candidate.key,
            "group": candidate.group,
            "file": str(path.relative_to(self.output_dir)),
            "recipe": candidate.recipe,
            "corr_b10": np.corrcoef(pred, b10)[0, 1],
            "delta_b10": np.abs(pred - b10).mean(),
            "corr_core": np.corrcoef(pred, core)[0, 1],
            "delta_core": np.abs(pred - core).mean(),
            "mean": pred.mean(),
            "std": pred.std(),
            "note": candidate.note,
        }

    def _write_readable_summary(self, summary: pd.DataFrame) -> None:
        readable = self.display_blend_summary(summary)
        readable.to_csv(self.report_dir / "blend_summary_readable.csv", index=False)
        with (self.report_dir / "blend_summary.md").open("w") as f:
            f.write("# Blend Summary\n\n")
            f.write(self._to_markdown_table(readable))
            f.write("\n")

    def _source_for(self, name: str) -> str:
        row = self.meta[self.meta["name"].eq(name)]
        return row.iloc[0]["source"] if len(row) else "unknown"

    def _score_for(self, name: str) -> float:
        row = self.meta[self.meta["name"].eq(name)]
        return float(row.iloc[0]["public_score"]) if len(row) and pd.notna(row.iloc[0]["public_score"]) else np.nan

    @staticmethod
    def _to_markdown_table(df: pd.DataFrame) -> str:
        columns = list(df.columns)
        rows = [[str(value) for value in row] for row in df.astype(object).itertuples(index=False, name=None)]
        widths = [
            max(len(str(column)), *(len(row[idx]) for row in rows)) if rows else len(str(column))
            for idx, column in enumerate(columns)
        ]

        def fmt_row(values):
            return "| " + " | ".join(str(value).ljust(widths[idx]) for idx, value in enumerate(values)) + " |"

        header = fmt_row(columns)
        separator = "| " + " | ".join("-" * width for width in widths) + " |"
        body = [fmt_row(row) for row in rows]
        return "\n".join([header, separator, *body])


def run_pitstop_blender(dataset_dir: str | Path, output_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cfg = SubmissionBlendConfig(dataset_dir=Path(dataset_dir), output_dir=Path(output_dir))
    return PitStopSubmissionBlender(cfg).run()

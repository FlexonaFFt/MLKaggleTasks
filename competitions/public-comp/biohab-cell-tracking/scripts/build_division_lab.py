#!/usr/bin/env python3
"""Build the division-diagnostics lab notebook from the pinned fusion stack.

Takes notebooks/biohub_harmonic_fusion_v17.ipynb (production, validator off),
forces the held-out validator on, and appends a division-diagnostics cell that
decomposes every GT division on the validator videos into observable facts:

- daughter detectability (min distance, dt) for each GT daughter lineage;
- presence/location of a predicted fork near the GT divider;
- branch coverage and same-component checks mirroring the official matcher;
- verdict string mapping the division to a failure class.

Output: notebooks/biohub_harmonic_fusion_division_lab.ipynb
"""

from __future__ import annotations

import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "notebooks" / "biohub_harmonic_fusion_v17.ipynb"
OUTPUT = ROOT / "notebooks" / "biohub_harmonic_fusion_division_lab.ipynb"
LOCK = ROOT / "harmonic_fusion_v17.lock.json"

PRESET_FLAG = "os.environ['BIOHUB_VALIDATOR_ENABLE'] = '0'"
VALIDATOR_MARKER = "validator_results.csv"
INSERT_AFTER_MARKER = "Per-sample validator rows written to"

DIAGNOSTIC_CELL = '''\
# Division diagnostics: decompose every GT division on the validator videos.
# Facts per division: daughter detectability, fork presence, branch coverage,
# component membership -> verdict class (detection gap vs gating vs topology).
import traceback
from collections import defaultdict

LAB_SCALE = (1.625, 0.40625, 0.40625)
LAB_MAX_DIST_UM = 7.0

def _lab_out_degree(edges):
    od = defaultdict(int)
    for s, t in edges:
        od[s] += 1
    return od

def _lab_succ(edges):
    succ = defaultdict(list)
    for s, t in edges:
        succ[s].append(t)
    return succ

def _lab_nearest_pred(pred_by_t, t, pos):
    best = (None, 1e9)
    for pid, pt, pz, py, px in pred_by_t.get(int(t), ()):
        d = point_distance_um(pos, (pz, py, px))
        if d < best[1]:
            best = (pid, d)
    return best
def _lab_components(node_ids, edges):
    parent = {n: n for n in node_ids}
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for s, t in edges:
        if s in parent and t in parent:
            ra, rb = find(s), find(t)
            if ra != rb:
                parent[rb] = ra
    return {n: find(n) for n in node_ids}

def _lab_lineage(gt_succ, root, t_div, depth=2):
    out = set()
    stack = [(root, 0)]
    while stack:
        node, d = stack.pop()
        if d > depth:
            continue
        if node not in out:
            out.add(node)
        for nxt in gt_succ.get(node, ()):
            stack.append((nxt, d + 1))
    return out

if not val_stems:
    print("DIVLAB: validator videos missing; enable the validator first.")
else:
    _val_dirs = sorted((REPO_DIR / "predictions").glob("*/unet_transformer_val/split_0"))
    if not _val_dirs:
        raise FileNotFoundError("DIVLAB: no unet_transformer_val predictions found")
    _val_dir = _val_dirs[0]
    rows = []
    category_counts = defaultdict(int)
    for stem in val_stems:
        try:
            gt_nodes, gt_edges = graph_to_plain(graph_from_geff(TRAIN_DIR / f"{stem}.geff"))
            pred_nodes, pred_edges = graph_to_plain(graph_from_geff(_val_dir / f"{stem}.geff"))
        except Exception as err:
            print(f"DIVLAB {stem}: load failed: {err!r}")
            continue
        pred_by_t = defaultdict(list)
        for pid, (t, z, y, x) in pred_nodes.items():
            pred_by_t[int(t)].append((pid, t, z, y, x))
        pred_od = _lab_out_degree(pred_edges)
        forks = [nid for nid, od in pred_od.items() if od >= 2]
        comp = _lab_components(set(pred_nodes), pred_edges)
        gt_succ = _lab_succ(gt_edges)
        gt_od = _lab_out_degree(gt_edges)
        dividers = [nid for nid, od in gt_od.items() if od >= 2]
        print(f"DIVLAB {stem}: gt_divisions={len(dividers)} pred_nodes={len(pred_nodes)} pred_forks={len(forks)}")
        for d in sorted(dividers, key=lambda n: gt_nodes[n][0]):
            t_div, dz, dy, dx = gt_nodes[d]
            dpos = (dz, dy, dx)
            daughters = sorted(gt_succ.get(d, ()))
            rec = {
                "dataset": stem, "gt_divider": d, "t": t_div,
                "n_daughters": len(daughters),
                "d1_detected": None, "d1_min_um": None, "d1_dt": None,
                "d2_detected": None, "d2_min_um": None, "d2_dt": None,
                "fork_found": False, "fork_dist_um": None, "fork_dt": None,
                "fork_same_comp_as_d1": None, "fork_same_comp_as_d2": None,
                "branch_coverage": 0, "verdict": "",
            }
            lineages = []
            for k, ch in enumerate(daughters[:2], start=1):
                lin = _lab_lineage(gt_succ, ch, t_div)
                lineages.append(lin)
                best = (None, 1e9, None)
                for g in lin:
                    gt_t, gz, gy, gx = gt_nodes[g]
                    pid, dist = _lab_nearest_pred(pred_by_t, gt_t, (gz, gy, gx))
                    if pid is not None and dist < best[1]:
                        best = (pid, dist, int(gt_t) - int(t_div))
                rec[f"d{k}_detected"] = best[0] is not None and best[1] <= LAB_MAX_DIST_UM
                rec[f"d{k}_min_um"] = None if best[0] is None else round(best[1], 2)
                rec[f"d{k}_dt"] = best[2]
            fork_best = (None, 1e9, None)
            for fid in forks:
                ft, fz, fy, fx = pred_nodes[fid]
                if abs(int(ft) - int(t_div)) > 1:
                    continue
                dist = point_distance_um(dpos, (fz, fy, fx))
                if dist < fork_best[1]:
                    fork_best = (fid, dist, int(ft) - int(t_div))
            if fork_best[0] is not None:
                rec["fork_found"] = True
                rec["fork_dist_um"] = round(fork_best[1], 2)
                rec["fork_dt"] = fork_best[2]
            matched_preds = []
            for k, lin in enumerate(lineages, start=1):
                cov = 0
                anchor = None
                for g in lin:
                    gt_t, gz, gy, gx = gt_nodes[g]
                    pid, dist = _lab_nearest_pred(pred_by_t, gt_t, (gz, gy, gx))
                    if pid is not None and dist <= LAB_MAX_DIST_UM:
                        cov = 1
                        if anchor is None:
                            anchor = pid
                rec["branch_coverage"] += cov
                matched_preds.append(anchor)
            pid_at_div, dist_at_div = _lab_nearest_pred(pred_by_t, t_div, dpos)
            anchor_ok = pid_at_div is not None and dist_at_div <= LAB_MAX_DIST_UM
            if rec["fork_found"]:
                fc = comp.get(fork_best[0])
                rec["fork_same_comp_as_d1"] = (matched_preds[0] is not None and comp.get(matched_preds[0]) == fc)
                rec["fork_same_comp_as_d2"] = (len(matched_preds) > 1 and matched_preds[1] is not None and comp.get(matched_preds[1]) == fc)
            if not rec["d1_detected"] and not rec["d2_detected"]:
                rec["verdict"] = "both_daughters_undetected"
            elif len(daughters) < 2:
                rec["verdict"] = "gt_not_a_true_split"
            elif rec["branch_coverage"] < 2:
                missing = [k for k in (1, 2) if not rec[f"d{k}_detected"]]
                rec["verdict"] = f"sister_undetected(d{','.join(str(m) for m in missing)})"
            elif not rec["fork_found"]:
                rec["verdict"] = "daughters_seen_no_fork_near_divider"
            elif not (rec["fork_same_comp_as_d1"] and rec["fork_same_comp_as_d2"]):
                rec["verdict"] = "fork_branches_in_different_components"
            else:
                rec["verdict"] = "local_tp_conditions_met"
            category_counts[rec["verdict"]] += 1
            rows.append(rec)
    import pandas as pd
    diag_df = pd.DataFrame(rows)
    if not diag_df.empty:
        diag_df.to_csv(WORKING_DIR / "division_diagnostics.csv", index=False)
        print(diag_df.to_string(index=False))
    print("DIVLAB verdict counts:", dict(category_counts))
    print("DIVLAB rows written to", WORKING_DIR / "division_diagnostics.csv")
'''


def cell_source(cell: dict) -> str:
    source = cell.get("source", "")
    return "".join(source) if isinstance(source, list) else source


def find_cell_index(notebook: dict, needle: str) -> int:
    for index, cell in enumerate(notebook["cells"]):
        if needle in cell_source(cell):
            return index
    raise RuntimeError(f"Marker not found in notebook: {needle!r}")


def build(notebook: dict) -> dict:
    validator_index = find_cell_index(notebook, INSERT_AFTER_MARKER)
    preset_index = find_cell_index(notebook, PRESET_FLAG)
    preset_source = cell_source(notebook["cells"][preset_index])
    notebook["cells"][preset_index]["source"] = preset_source.replace(
        PRESET_FLAG,
        "os.environ['BIOHUB_VALIDATOR_ENABLE'] = '1'  # lab build: validator on",
        1,
    )
    preset_now = cell_source(notebook["cells"][preset_index])
    if "BIOHUB_VALIDATOR_ENABLE'] = '1'" not in preset_now:
        raise RuntimeError("Failed to enable validator in preset")
    notebook["cells"][preset_index]["source"] = preset_now.replace(
        "BIOHUB_PRESET = 'kimi_v17_harmonic_allin'",
        "BIOHUB_PRESET = 'kimi_v17_harmonic_division_lab'",
        1,
    )
    diagnostic = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": DIAGNOSTIC_CELL.splitlines(keepends=True),
    }
    notebook["cells"].insert(validator_index + 1, diagnostic)
    return notebook


def validate(notebook: dict) -> None:
    source_all = "\n".join(cell_source(cell) for cell in notebook["cells"])
    for needle in (
        "BIOHUB_VALIDATOR_ENABLE'] = '1'",
        "division_diagnostics.csv",
        "kimi_v17_harmonic_division_lab",
    ):
        if needle not in source_all:
            raise RuntimeError(f"Lab notebook missing {needle!r}")
    if source_all.count("BIOHUB_VALIDATOR_ENABLE'] = '0'"):
        raise RuntimeError("Lab notebook still disables the validator")
    for index, cell in enumerate(notebook["cells"]):
        if cell.get("cell_type") != "code":
            continue
        try:
            ast.parse(cell_source(cell))
        except SyntaxError as error:
            raise RuntimeError(f"Cell {index} does not parse: {error}") from error
    diag = next(cell_source(c) for c in notebook["cells"] if "DIVLAB" in cell_source(c))
    diag_idx = [i for i, c in enumerate(notebook["cells"]) if "DIVLAB" in cell_source(c)][0]
    validator_idx = find_cell_index(notebook, INSERT_AFTER_MARKER)
    if diag_idx < validator_idx:
        raise RuntimeError("Diagnostics must run after the validator metrics cell")


def main() -> None:
    notebook = json.loads(SOURCE.read_text())
    version = "unknown"
    if LOCK.exists():
        version = json.loads(LOCK.read_text()).get("version", version)
    notebook = build(notebook)
    validate(notebook)
    OUTPUT.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n")
    print(f"Wrote {OUTPUT} (upstream v{version}, validator on + division diagnostics)")


if __name__ == "__main__":
    main()

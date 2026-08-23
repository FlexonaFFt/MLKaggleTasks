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
                "d1_detected": None, "d1_min_um": None, "d1_dt": None, "d1_split_um": None,
                "d2_detected": None, "d2_min_um": None, "d2_dt": None, "d2_split_um": None,
                "fork_found": False, "fork_dist_um": None, "fork_dt": None,
                "fork_same_comp_as_d1": None, "fork_same_comp_as_d2": None,
                "branch_coverage": 0, "verdict": "",
            }
            lineages = []
            for k, ch in enumerate(daughters[:2], start=1):
                lin = _lab_lineage(gt_succ, ch, t_div, depth=1)
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
                st, sz, sy, sx = gt_nodes[ch]
                spid, sdist = _lab_nearest_pred(pred_by_t, int(st), (sz, sy, sx))
                rec[f"d{k}_split_um"] = None if spid is None else round(sdist, 2)
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


EXPERIMENT_CELL = '''\
# Division-aware fill experiment: synthesize the missing sister at t+1.
# Signature of an unfilled division in the raw ILP graph:
#   P(t) -> M(t+1) -> A(t+2), with a sister B(t+2) near A whose parent is
#   not M (daughters merged/missed at t+1, resolved one frame later).
# Surgery: insert S(t+1) at B's position, add edges P->S and S->B so P
# becomes a fork with both daughter branches. Then rescore A/B with the
# official-metric sample scorer from the validator cell.
from collections import defaultdict

FILL_SISTER_MAX_UM = 5.0
FILL_PARENT_MAX_UM = 14.0
FILL_MAX_PER_DATASET = 500

def _fill_parents(edges):
    par = defaultdict(list)
    for s, t in edges:
        par[t].append(s)
    return par

def _fill_children(edges):
    ch = defaultdict(list)
    for s, t in edges:
        ch[s].append(t)
    return ch

def fill_divisions(pred_nodes, pred_edges):
    nodes = dict(pred_nodes)
    edges = list(pred_edges)
    parents = _fill_parents(edges)
    children = _fill_children(edges)
    by_t = defaultdict(list)
    for nid, (t, z, y, x) in nodes.items():
        by_t[int(t)].append(nid)
    next_id = max(nodes) + 1 if nodes else 1
    fills = 0
    new_forks = set()
    used = set()
    for m in sorted(nodes):
        tm, mz, my, mx = nodes[m]
        ps = parents.get(m, [])
        if len(ps) != 1:
            continue
        p = ps[0]
        cs = children.get(m, [])
        if len(cs) != 1 or len(children.get(p, [])) != 1:
            continue
        a = cs[0]
        ta, az, ay, ax = nodes[a]
        if int(ta) != int(tm) + 1:
            continue
        best = (None, 1e9)
        for b in by_t.get(int(ta), ()):
            if b in (a, m, p) or b in used:
                continue
            if m in parents.get(b, ()) or p in parents.get(b, ()):
                continue
            d = point_distance_um((az, ay, ax), (nodes[b][1], nodes[b][2], nodes[b][3]))
            if d <= FILL_SISTER_MAX_UM and d < best[1]:
                best = (b, d)
        b = best[0]
        if b is None:
            continue
        pz, py, px = nodes[p][1], nodes[p][2], nodes[p][3]
        if point_distance_um((pz, py, px), (nodes[b][1], nodes[b][2], nodes[b][3])) > FILL_PARENT_MAX_UM:
            continue
        # v2: the sister is usually already tracked from a wrong parent Q.
        # Re-parent B to the new synthetic node S instead of giving B two
        # parents (the official scorer rejects merged branches).
        reparented = False
        for q in parents.get(b, ()):
            edge = (q, b)
            if edge in edges:
                edges.remove(edge)
                reparented = True
        if reparented:
            parents = _fill_parents(edges)
            children = _fill_children(edges)
        s = next_id
        next_id += 1
        nodes[s] = (int(tm) + 1, nodes[b][1], nodes[b][2], nodes[b][3])
        edges.append((p, s))
        edges.append((s, b))
        used.add(b)
        new_forks.add(p)
        fills += 1
        if fills >= FILL_MAX_PER_DATASET:
            break
    return nodes, edges, fills, new_forks

if not val_stems:
    print("FILL: no validator videos; skipping experiment.")
else:
    _fill_dirs = sorted((REPO_DIR / "predictions").glob("*/unet_transformer_val/split_0"))
    if not _fill_dirs:
        raise FileNotFoundError("FILL: no unet_transformer_val predictions found")
    _fill_dir = _fill_dirs[0]
    _fill_rows = []
    for stem in val_stems:
        try:
            gt_nodes, gt_edges = graph_to_plain(graph_from_geff(TRAIN_DIR / f"{stem}.geff"))
            pred_nodes, pred_edges = graph_to_plain(graph_from_geff(_fill_dir / f"{stem}.geff"))
            t_true = read_estimated_true_node_count(TRAIN_DIR / f"{stem}.geff")
        except Exception as err:
            print(f"FILL {stem}: load failed: {err!r}")
            continue
        before = score_sample(pred_nodes, pred_edges, gt_nodes, gt_edges, t_true)
        filled_nodes, filled_edges, fills, new_forks = fill_divisions(pred_nodes, pred_edges)
        after = score_sample(filled_nodes, filled_edges, gt_nodes, gt_edges, t_true)
        gt_out = defaultdict(list)
        for s2, t2 in gt_edges:
            gt_out[s2].append(t2)
        gt_dividers = [g2 for g2, kids in gt_out.items() if len(kids) >= 2]
        covered = 0
        for gd in gt_dividers:
            gt_t, gz, gy, gx = gt_nodes[gd]
            hit = False
            for fid in new_forks:
                ft, fz, fy, fx = filled_nodes[fid]
                if abs(int(ft) - int(gt_t)) <= 1 and point_distance_um(
                    (gz, gy, gx), (fz, fy, fx)
                ) <= 7.0:
                    hit = True
                    break
            covered += 1 if hit else 0
        cover_text = f"{covered}/{len(gt_dividers)}"
        _fill_rows.append({
            "dataset": stem, "fills": fills, "gt_div_covered": cover_text,
            "div_tp_before": before["div_tp"], "div_tp_after": after["div_tp"],
            "div_fp_before": before["div_fp"], "div_fp_after": after["div_fp"],
            "div_fn_before": before["div_fn"], "div_fn_after": after["div_fn"],
            "edge_jac_before": round(before["edge_jaccard"], 4),
            "edge_jac_after": round(after["edge_jaccard"], 4),
            "adj_before": round(before["adjusted_edge_jaccard"], 4),
            "adj_after": round(after["adjusted_edge_jaccard"], 4),
        })
    import pandas as pd
    exp_df = pd.DataFrame(_fill_rows)
    if exp_df.empty:
        print("FILL: no samples scored.")
    else:
        print(exp_df.to_string(index=False))
        tp_a = int(exp_df.div_tp_after.sum()); fp_a = int(exp_df.div_fp_after.sum()); fn_a = int(exp_df.div_fn_after.sum())
        tp_b = int(exp_df.div_tp_before.sum()); den_b = int((exp_df.div_tp_before + exp_df.div_fp_before + exp_df.div_fn_before).sum())
        print(f"FILL divisions micro: before {tp_b}/{den_b} (jac={tp_b / max(1, den_b):.3f}) "
              f"after {tp_a}/{tp_a + fp_a + fn_a} (jac={tp_a / max(1, tp_a + fp_a + fn_a):.3f})")
        print(f"FILL mean adj_edge_jaccard: before={exp_df.adj_before.mean():.4f} after={exp_df.adj_after.mean():.4f}")
        exp_df.to_csv(WORKING_DIR / "fill_experiment.csv", index=False)
        print("FILL rows written to", WORKING_DIR / "fill_experiment.csv")
'''

TOPOLOGY_CELL = '''\
# Topology dump: local prediction subgraph around every GT division.
# Purpose: see the ACTUAL node/edge structure near divisions so the fill
# signature is designed from facts, not guesses.
if not val_stems:
    print("TOPO: no validator videos.")
else:
    _topo_dirs = sorted((REPO_DIR / "predictions").glob("*/unet_transformer_val/split_0"))
    _topo_dir = _topo_dirs[0]
    for stem in val_stems:
        try:
            gt_nodes, gt_edges = graph_to_plain(graph_from_geff(TRAIN_DIR / f"{stem}.geff"))
            pred_nodes, pred_edges = graph_to_plain(graph_from_geff(_topo_dir / f"{stem}.geff"))
        except Exception as err:
            print(f"TOPO {stem}: load failed: {err!r}")
            continue
        gt_succ = defaultdict(list)
        for s2, t2 in gt_edges:
            gt_succ[s2].append(t2)
        pred_parents = defaultdict(list)
        pred_children = defaultdict(list)
        for s2, t2 in pred_edges:
            pred_parents[t2].append(s2)
            pred_children[s2].append(t2)
        dividers = [g2 for g2, kids in ((n, gt_succ.get(n, [])) for n in gt_nodes) if len(kids) >= 2]
        print(f"TOPO {stem}: {len(dividers)} gt divisions")
        for d in sorted(dividers, key=lambda n: gt_nodes[n][0]):
            td, dz, dy, dx = gt_nodes[d]
            dpos = (dz, dy, dx)
            daughters = gt_succ.get(d, [])
            lin = set([d])
            for ch in daughters:
                lin.add(ch)
                for gc in gt_succ.get(ch, []):
                    lin.add(gc)
            print(f"  == GT div {d} t={td} pos=({dz:.1f},{dy:.1f},{dx:.1f}) daughters={daughters}")
            for g in sorted(lin):
                if g == d:
                    continue
                gt_t, gz, gy, gx = gt_nodes[g]
                print(f"    GT {g} t={gt_t} pos=({gz:.1f},{gy:.1f},{gx:.1f})")
            near = []
            for pid, (pt, pz, py, px) in pred_nodes.items():
                if abs(int(pt) - int(td)) > 3:
                    continue
                dist = point_distance_um(dpos, (pz, py, px))
                if dist <= 12.0:
                    near.append((int(pt), dist, pid))
            for pt, dist, pid in sorted(near):
                pars = [f"{q}@t{pred_nodes[q][0]}" for q in pred_parents.get(pid, [])]
                kids = [f"{c}@t{pred_nodes[c][0]}" for c in pred_children.get(pid, [])]
                pz, py, px = pred_nodes[pid][1], pred_nodes[pid][2], pred_nodes[pid][3]
                print(f"    PRED {pid} t={pt} d_div={dist:.2f}um pos=({pz:.1f},{py:.1f},{px:.1f}) parents={pars} children={kids}")
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
    experiment = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": EXPERIMENT_CELL.splitlines(keepends=True),
    }
    diag_index = find_cell_index(notebook, "division_diagnostics.csv")
    notebook["cells"].insert(diag_index + 1, experiment)
    topology = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": TOPOLOGY_CELL.splitlines(keepends=True),
    }
    exp_index = find_cell_index(notebook, "fill_experiment.csv")
    notebook["cells"].insert(exp_index + 1, topology)
    return notebook


def validate(notebook: dict) -> None:
    source_all = "\n".join(cell_source(cell) for cell in notebook["cells"])
    for needle in (
        "BIOHUB_VALIDATOR_ENABLE'] = '1'",
        "division_diagnostics.csv",
        "fill_experiment.csv",
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
    diag_idx = [i for i, c in enumerate(notebook["cells"]) if "DIVLAB" in cell_source(c)][0]
    exp_idx = [i for i, c in enumerate(notebook["cells"]) if "FILL" in cell_source(c)][0]
    validator_idx = find_cell_index(notebook, INSERT_AFTER_MARKER)
    if diag_idx < validator_idx or exp_idx <= diag_idx:
        raise RuntimeError("Diagnostics and fill experiment must run after validator metrics")


def main() -> None:
    notebook = json.loads(SOURCE.read_text())
    version = "unknown"
    if LOCK.exists():
        version = json.loads(LOCK.read_text()).get("version", version)
    notebook = build(notebook)
    validate(notebook)
    OUTPUT.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n")
    print(f"Wrote {OUTPUT} (upstream v{version}, validator on + diagnostics + fill experiment)")


if __name__ == "__main__":
    main()

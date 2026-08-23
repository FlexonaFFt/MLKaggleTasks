import unittest

import json
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from build_division_lab import (
    DIAGNOSTIC_CELL,
    build,
    validate,
)

SCALE = (1.625, 0.40625, 0.40625)


def _node(t, z, y, x):
    return (t, z, y, x)


class DivisionLabTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = json.loads(
            (Path(__file__).resolve().parents[1] / "notebooks" / "biohub_harmonic_fusion_v17.ipynb").read_text()
        )

    def test_build_enables_validator_and_inserts_diag_after_metrics(self):
        notebook = build(json.loads(json.dumps(type(self).source)))
        validate(notebook)
        diag_idx = next(
            i for i, c in enumerate(notebook["cells"]) if "DIVLAB" in "".join(c["source"])
        )
        metrics_idx = next(
            i for i, c in enumerate(notebook["cells"]) if "Per-sample validator rows" in "".join(c["source"])
        )
        self.assertGreater(diag_idx, metrics_idx)

    def _run_cell(self, tmp_path, gt_nodes, gt_edges, pred_nodes, pred_edges):
        """Exec the diagnostic cell against synthetic graphs; return namespace."""
        val_dir = tmp_path / "repo" / "predictions" / "unknown" / "unet_transformer_val" / "split_0"
        train_dir = tmp_path / "train"
        train_dir.mkdir(parents=True)
        val_dir.mkdir(parents=True)
        for stem in ("s1",):
            (val_dir / f"{stem}.geff").write_bytes(b"x")
            (train_dir / f"{stem}.geff").write_bytes(b"x")

        def graph_from_geff(path):
            class Sentinel:
                pass
            s = Sentinel()
            s.path = str(path)
            return s

        def graph_to_plain(graph):
            if "train" in graph.path:
                return dict(gt_nodes), list(gt_edges)
            return dict(pred_nodes), list(pred_edges)

        ns = {
            "val_stems": ["s1"],
            "REPO_DIR": tmp_path / "repo",
            "TRAIN_DIR": train_dir,
            "WORKING_DIR": tmp_path,
            "graph_from_geff": graph_from_geff,
            "graph_to_plain": graph_to_plain,
            "point_distance_um": lambda a, b: sum(
                ((a[i] - b[i]) * SCALE[i]) ** 2 for i in range(3)
            ) ** 0.5,
            "VOXEL_SCALE_UM": SCALE,
        }
        exec(DIAGNOSTIC_CELL, ns)  # noqa: S102 - trusted local test
        return ns

    def test_full_recovery_verdict(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            # GT: divider at t=5; two daughters at t=6; grandchildren t=7.
            gt = {
                1: _node(5, 100.0, 100.0, 100.0),
                2: _node(6, 101.0, 100.4, 100.0),
                3: _node(6, 99.0, 99.6, 100.0),
                4: _node(7, 101.5, 100.8, 100.0),
                5: _node(7, 98.5, 99.2, 100.0),
            }
            gt_edges = [(1, 2), (1, 3), (2, 4), (3, 5)]
            # Pred: same topology incl. a fork at the matched parent node.
            pred = {10 + k: v for k, v in enumerate(gt.values())}
            pred_edges = [(10, 11), (10, 12), (11, 13), (12, 14)]
            ns = self._run_cell(tmp, gt, gt_edges, pred, pred_edges)
            rows = ns["diag_df"]
            self.assertEqual(len(rows), 1)
            row = rows.iloc[0]
            self.assertEqual(row["verdict"], "local_tp_conditions_met")
            self.assertEqual(row["branch_coverage"], 2)
            self.assertTrue(row["fork_found"])

    def test_sister_undetected_verdict(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            gt = {
                1: _node(5, 100.0, 100.0, 100.0),
                2: _node(6, 101.0, 100.4, 100.0),
                3: _node(6, 92.0, 92.0, 100.0),  # far sister
                4: _node(7, 101.5, 100.8, 100.0),
            }
            gt_edges = [(1, 2), (1, 3), (2, 4)]
            pred = {
                10: _node(5, 100.0, 100.0, 100.0),
                11: _node(6, 101.0, 100.4, 100.0),
                13: _node(7, 101.5, 100.8, 100.0),
            }
            pred_edges = [(10, 11), (11, 13)]
            ns = self._run_cell(tmp, gt, gt_edges, pred, pred_edges)
            row = ns["diag_df"].iloc[0]
            self.assertIn("sister_undetected", row["verdict"])
            self.assertEqual(row["branch_coverage"], 1)

    def test_fork_in_different_component(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            gt = {
                1: _node(5, 100.0, 100.0, 100.0),
                2: _node(6, 101.0, 100.4, 100.0),
                3: _node(6, 99.0, 99.6, 100.0),
            }
            gt_edges = [(1, 2), (1, 3)]
            # Pred: fork near divider with a WRONG second child; the true
            # daughter2 is detected but as an isolated node in another comp.
            pred = {
                10: _node(5, 100.0, 100.0, 100.0),   # fork (parent)
                11: _node(6, 101.0, 100.4, 100.0),   # daughter1, tracked
                12: _node(6, 99.0, 99.6, 100.0),     # daughter2, isolated
                14: _node(6, 100.5, 100.2, 100.0),   # wrong second child
            }
            pred_edges = [(10, 11), (10, 14)]
            ns = self._run_cell(tmp, gt, gt_edges, pred, pred_edges)
            row = ns["diag_df"].iloc[0]
            self.assertTrue(row["d2_detected"])
            self.assertTrue(row["fork_found"])
            self.assertNotEqual(row["verdict"], "local_tp_conditions_met")


    def test_fill_experiment_creates_fork_and_division_tp(self):
        import tempfile
        from build_division_lab import EXPERIMENT_CELL
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            val_dir = tmp / "repo" / "predictions" / "unknown" / "unet_transformer_val" / "split_0"
            train_dir = tmp / "train"
            train_dir.mkdir(parents=True)
            val_dir.mkdir(parents=True)
            for stem in ("s1",):
                (val_dir / f"{stem}.geff").write_bytes(b"x")
                (train_dir / f"{stem}.geff").write_bytes(b"x")

            gt = {
                1: _node(5, 100.0, 100.0, 100.0),
                2: _node(6, 101.0, 100.4, 100.0),
                3: _node(6, 99.0, 99.6, 100.0),
                4: _node(7, 101.5, 100.8, 100.0),
                5: _node(7, 98.6, 99.3, 100.0),
            }
            gt_edges = [(1, 2), (1, 3), (2, 4), (3, 5)]
            # Adoption scenario: daughter2 (12) is tracked from the WRONG
            # parent 13; node 10 has a single child and adopts 12.
            pred = {
                10: _node(5, 100.0, 100.0, 100.0),
                11: _node(6, 101.0, 100.4, 100.0),
                12: _node(6, 99.0, 99.6, 100.0),
                13: _node(5, 100.0, 100.0, 100.0),
            }
            pred_edges = [(10, 11), (13, 12)]

            def graph_from_geff(path):
                class S:
                    pass
                s = S()
                s.path = str(path)
                return s

            def graph_to_plain(graph):
                if "train" in graph.path:
                    return dict(gt), list(gt_edges)
                return dict(pred), list(pred_edges)

            def forks_children(edges, f):
                return [t2 for s, t2 in edges if s == f]

            def nearest(nid_pos_by_t, t, pos):
                best = (None, 1e9)
                for pid, pos2 in nid_pos_by_t.get(int(t), ()):
                    d = sum(((pos[i] - pos2[i]) * SCALE[i]) ** 2 for i in range(3)) ** 0.5
                    if d < best[1]:
                        best = (pid, d)
                return best

            def mini_score(pn, pe, gn, ge, t_true):
                p_by_t = {}
                for pid, (t, z, y, x) in pn.items():
                    p_by_t.setdefault(int(t), []).append((pid, (z, y, x)))
                g_out = {}
                for s, t2 in ge:
                    g_out.setdefault(s, set()).add(t2)
                p2g = {}
                for gid, (t, gz, gy, gx) in gn.items():
                    pid, d = nearest(p_by_t, t, (gz, gy, gx))
                    if pid is not None and d <= 7.0 and gid not in p2g.values():
                        p2g[pid] = gid
                out_deg = {}
                for s, t2 in pe:
                    out_deg[s] = out_deg.get(s, 0) + 1
                forks = [n for n, k in out_deg.items() if k >= 2]
                n_gt_div = sum(1 for k in g_out.values() if len(k) >= 2)
                div_tp = 0
                for gd, kids in g_out.items():
                    if len(kids) < 2:
                        continue
                    gd_t, gdz, gdy, gdx = gn[gd]
                    pf, pd_ = nearest(p_by_t, int(gd_t), (gdz, gdy, gdx))
                    if pf is None or pd_ > 7.0 or out_deg.get(pf, 0) < 2:
                        continue
                    matched_kids = {p2g[c] for c in forks_children(pe, pf) if c in p2g}
                    if matched_kids & set(kids):
                        div_tp += 1
                te = {tuple(sorted(e)) for e in ge}
                pe_set = {tuple(sorted(e)) for e in pe}
                etp = len(pe_set & te)
                efp = len(pe_set - te)
                efn = len(te - pe_set)
                jac = etp / max(1, etp + efp + efn)
                adj = max(0.0, jac * (1 - 0.1 * (len(pn) - t_true) / t_true))
                djac = div_tp / max(1, n_gt_div)
                return {"div_tp": div_tp, "div_fp": 0, "div_fn": n_gt_div - div_tp,
                        "edge_jaccard": jac, "adjusted_edge_jaccard": adj,
                        "div_jaccard": djac}

            ns = {
                "val_stems": ["s1"],
                "REPO_DIR": tmp / "repo",
                "TRAIN_DIR": train_dir,
                "WORKING_DIR": tmp,
                "graph_from_geff": graph_from_geff,
                "graph_to_plain": graph_to_plain,
                "point_distance_um": lambda a, b: sum(
                    ((a[i] - b[i]) * SCALE[i]) ** 2 for i in range(3)
                ) ** 0.5,
                "score_sample": mini_score,
                "read_estimated_true_node_count": lambda p: 100.0,
            }
            exec(EXPERIMENT_CELL, ns)  # noqa: S102 - trusted local test
            row = ns["exp_df"].iloc[0]
            self.assertEqual(row["fills"], 1)
            self.assertEqual(int(row["div_tp_before"]), 0)
            self.assertGreaterEqual(int(row["div_tp_after"]), 1)


if __name__ == "__main__":
    unittest.main()

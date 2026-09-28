import collections
import ast
import json
import math
import os
import tempfile
from pathlib import Path


def main():
    source = Path(__file__).with_name("biohub-research-ilp040-readmit094.ipynb")
    nb = json.loads(source.read_text())
    graph = nb["cells"][5]["source"]
    tree = ast.parse(graph)
    readmit_tree = ast.Module(body=[n for n in tree.body if (isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id.startswith("READMIT_") for t in n.targets)) or (isinstance(n, ast.FunctionDef) and n.name in ("load_readmit_candidates", "readmit_open_track_ends"))], type_ignores=[])
    launch = nb["cells"][4]["source"]
    begin = launch.index("READMIT_CAPTURE_HELPER = ")
    end = launch.index("start_time = time.time()", begin)
    capture_patch = launch[begin:end]
    old = os.environ.get("BIOHUB_READMIT_MIN_SCORE")
    os.environ["BIOHUB_READMIT_MIN_SCORE"] = "0.94"
    try:
        scale = (1.625, 0.40625, 0.40625)

        def point(node):
            return tuple(float(node[k]) for k in ("z", "y", "x"))

        def distance(a, b):
            return math.sqrt(sum(((x - y) * factor) ** 2 for x, y, factor in zip(a, b, scale)))

        ns = {"os": os, "node_point": point, "point_distance_um": distance, "edge_distance_um": lambda a, b: distance(point(a), point(b))}
        exec(compile(readmit_tree, "<readmit-functions>", "exec"), ns)
        ns["READMIT_MAX_ADDED_FRACTION"] = 0.5

        def node(i, t, x):
            return {"node_id": i, "t": t, "z": 0, "y": 0, "x": x}

        original = {0: node(0, 0, 0), 1: node(1, 1, 4), 2: node(2, 3, 12), 3: node(3, 4, 16)}
        edges = [{"source_id": 0, "target_id": 1}, {"source_id": 2, "target_id": 3}]
        candidate = {"t": 2, "z": 0, "y": 0, "x": 8, "readmit_score": 0.95}
        stats = {}
        nodes, result = ns["readmit_open_track_ends"](dict(original), list(edges), stats, [candidate])
        assert stats["readmit_nodes"] == 1 and stats["readmit_bridges"] == 1, stats
        assert len(nodes) == 5 and len(result) == 4
        assert max(collections.Counter(e["target_id"] for e in result).values()) == 1
        assert max(collections.Counter(e["source_id"] for e in result).values()) == 1
        assert all(nodes[e["target_id"]]["t"] == nodes[e["source_id"]]["t"] + 1 for e in result)
        for rejected in [{**candidate, "readmit_score": 0.93}, {**candidate, "y": 30}]:
            stats = {}
            nodes, result = ns["readmit_open_track_ends"](dict(original), list(edges), stats, [rejected])
            assert stats["readmit_nodes"] == 0 and nodes == original and result == edges
        with_existing = {**original, 4: node(4, 2, 8)}
        stats = {}
        ns["readmit_open_track_ends"](with_existing, list(edges), stats, [candidate])
        assert stats["readmit_nodes"] == 0
        with tempfile.TemporaryDirectory() as directory:
            script = Path(directory) / "predict.py"
            script.write_text('def _detect_cells_pooled(heat, t, threshold, pool):\n    return []\n\ndef no_grad(function):\n    return function\n\n@no_grad\ndef predict_video(ds_path, det_logits, primary_det, blended_det, cfg, pool_k):\n    a = _detect_cells_pooled(primary_det[0], 0, cfg.det_threshold, pool_k)\n    b = _detect_cells_pooled(blended_det[0], 0, cfg.det_threshold, pool_k)\n    return _detect_cells_pooled(det_logits[0], 0, cfg.det_threshold, pool_k)\n')
            exec(capture_patch, {"_ps": script, "os": os})
            patched = script.read_text()
            compile(patched, str(script), "exec")
            assert patched.count("_readmit_dataset=ds_path.stem") == 1
            assert "@no_grad\ndef predict_video" in patched
        assert len(nb["cells"]) == 12
        for i, cell in enumerate(nb["cells"]):
            if cell["cell_type"] == "code":
                compile(cell["source"], str(source) + f":cell{i}", "exec")
                assert not cell["outputs"] and cell["execution_count"] is None
        print("PASS: bridge, score and motion rejection, duplicate avoidance, graph invariants, capture patch, notebook compilation")
    finally:
        if old is None:
            os.environ.pop("BIOHUB_READMIT_MIN_SCORE", None)
        else:
            os.environ["BIOHUB_READMIT_MIN_SCORE"] = old


if __name__ == "__main__":
    main()

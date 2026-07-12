#!/usr/bin/env python3
"""Validate Biohub CSV schema and basic directed-lineage invariants."""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path


COLUMNS = ["id", "dataset", "row_type", "node_id", "t", "z", "y", "x", "source_id", "target_id"]
INT_COLUMNS = ["node_id", "t", "z", "y", "x", "source_id", "target_id"]


def validate(path: Path) -> list[str]:
    errors: list[str] = []
    nodes: dict[tuple[str, int], int] = {}
    edges: list[tuple[str, int, int, int]] = []
    ids: set[int] = set()

    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != COLUMNS:
            return [f"columns must be exactly {COLUMNS}; got {reader.fieldnames}"]
        for line, row in enumerate(reader, 2):
            try:
                row_id = int(row["id"])
                values = {column: int(row[column]) for column in INT_COLUMNS}
            except ValueError:
                errors.append(f"line {line}: integer column contains a non-integer")
                continue
            if row_id in ids:
                errors.append(f"line {line}: duplicate id {row_id}")
            ids.add(row_id)
            key = (row["dataset"], values["node_id"])
            if row["row_type"] == "node":
                if values["node_id"] < 0 or values["t"] < 0:
                    errors.append(f"line {line}: node_id and t must be non-negative")
                if key in nodes:
                    errors.append(f"line {line}: duplicate node {key}")
                nodes[key] = values["t"]
            elif row["row_type"] == "edge":
                edges.append((row["dataset"], values["source_id"], values["target_id"], line))
            else:
                errors.append(f"line {line}: row_type must be node or edge")

    parents: Counter[tuple[str, int]] = Counter()
    children: Counter[tuple[str, int]] = Counter()
    seen_edges: set[tuple[str, int, int]] = set()
    adjacency: dict[tuple[str, int], list[int]] = defaultdict(list)
    for dataset, source, target, line in edges:
        edge = (dataset, source, target)
        if edge in seen_edges:
            errors.append(f"line {line}: duplicate edge {edge}")
        seen_edges.add(edge)
        source_key, target_key = (dataset, source), (dataset, target)
        if source_key not in nodes or target_key not in nodes:
            errors.append(f"line {line}: edge references a missing node")
            continue
        if nodes[target_key] != nodes[source_key] + 1:
            errors.append(f"line {line}: edge must connect consecutive frames")
        parents[target_key] += 1
        children[source_key] += 1
        adjacency[source_key].append(target)

    for node, count in parents.items():
        if count > 1:
            errors.append(f"node {node} has {count} parents")
    for node, count in children.items():
        if count > 2:
            errors.append(f"node {node} has {count} children")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("submission", type=Path)
    args = parser.parse_args()
    errors = validate(args.submission)
    if errors:
        for error in errors[:50]:
            print("ERROR:", error)
        raise SystemExit(f"Invalid submission: {len(errors)} error(s)")
    print("Submission structure is valid")


if __name__ == "__main__":
    main()

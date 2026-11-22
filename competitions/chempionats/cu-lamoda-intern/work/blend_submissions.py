"""Rank-blend сабмитов: score = средневзвешенный percentile-rank по файлам.
Использование: python blend_submissions.py out.csv f1.csv f2.csv [w1 w2 ...]
"""
import sys
import pandas as pd

out, rest = sys.argv[1], sys.argv[2:]
if len(rest) >= 4 and len(rest) % 2 == 0 and all(x.replace(".", "").isdigit() for x in rest[len(rest) // 2:]):
    files, w = rest[: len(rest) // 2], [float(x) for x in rest[len(rest) // 2:]]
else:
    files, w = rest, [1.0] * len(rest)

merged = None
for i, (f, wi) in enumerate(zip(files, w)):
    df = pd.read_csv(f)
    assert set(df.columns) >= {"customer_id", "score"}, df.columns
    part = df[["customer_id"]].copy()
    part[f"rank_{i}"] = df["score"].rank(pct=True) * wi
    merged = part if merged is None else merged.merge(part, on="customer_id", how="inner", validate="1:1")

rcols = [c for c in merged.columns if c.startswith("rank_")]
merged["score"] = (merged[rcols].sum(axis=1) / sum(w)).rank(pct=True)
merged[["customer_id", "score"]].to_csv(out, index=False)
print(f"blended {len(files)} files (weights {w}) -> {out}, rows: {len(merged)}")

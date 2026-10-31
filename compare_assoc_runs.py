"""Compare the per-gene assoc results two runs share, to a tolerance.

Byte equality is too strict across machines: BLAS picks CPU-specific code
paths, so the last bits of a float can differ between nodes. This checks
every column of every per_gene file both runs have instead: text columns must
match exactly, numbers to EPS relative to their size (absolute below 1).

    python compare_assoc_runs.py NEW_ASSOC_DIR OLD_ASSOC_DIR [--eps 1e-8]

Prints how many shared files differ by more than eps, then each column's
largest difference, worst first. Exit status 1 if any file differs.
"""
import argparse
import glob
import os
import sys
from typing import NamedTuple

import numpy as np
import pandas as pd

DEFAULT_EPS = 1e-8


class Worst(NamedTuple):
    diff: float
    file: str
    row: int
    new: object
    old: object


def relative_diff(new: pd.Series, old: pd.Series) -> np.ndarray:
    """|new - old| / max(|new|, |old|, 1); 0 where both are NaN, inf where one is."""
    x, y = new.to_numpy(float), old.to_numpy(float)
    with np.errstate(invalid="ignore"):
        d = np.abs(x - y) / np.maximum(np.maximum(np.abs(x), np.abs(y)), 1.0)
    one_nan = np.isnan(x) ^ np.isnan(y)
    return np.where(np.isnan(x) & np.isnan(y), 0.0, np.where(one_nan, np.inf, d))


def compare_frames(new: pd.DataFrame, old: pd.DataFrame, name: str) -> tuple[dict, list[str]]:
    """Per-column Worst for numeric columns, and messages for shape/text mismatches."""
    if list(new.columns) != list(old.columns) or len(new) != len(old):
        return {}, [f"SHAPE {name}: columns or row count differ"]
    worst, problems = {}, []
    for col in new.columns:
        x, y = new[col], old[col]
        if pd.api.types.is_numeric_dtype(x) and pd.api.types.is_numeric_dtype(y):
            d = relative_diff(x, y)
            i = int(np.argmax(d)) if len(d) else 0
            if len(d):
                worst[col] = Worst(float(d[i]), name, i, x.iloc[i], y.iloc[i])
        else:
            differs = (x.astype(str) != y.astype(str)).to_numpy()
            if differs.any():
                i = int(np.argmax(differs))
                problems.append(f"TEXT  {name} {col} row {i}: {x.iloc[i]!r} vs {y.iloc[i]!r}")
    return worst, problems


def shared_files(new_dir: str, old_dir: str) -> list[str]:
    paths = glob.glob(os.path.join(new_dir, "*_*", "per_gene", "*.tsv"))
    rels = (os.path.relpath(p, new_dir) for p in sorted(paths))
    return [r for r in rels if os.path.isfile(os.path.join(old_dir, r))]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("new_dir")
    ap.add_argument("old_dir")
    ap.add_argument("--eps", type=float, default=DEFAULT_EPS)
    args = ap.parse_args(argv)

    overall: dict[str, Worst] = {}
    n_bad = 0
    files = shared_files(args.new_dir, args.old_dir)
    for rel in files:
        new = pd.read_csv(os.path.join(args.new_dir, rel), sep="\t")
        old = pd.read_csv(os.path.join(args.old_dir, rel), sep="\t")
        worst, problems = compare_frames(new, old, rel)
        for line in problems:
            print(line)
        if problems or any(w.diff > args.eps for w in worst.values()):
            n_bad += 1
        for col, w in worst.items():
            if col not in overall or w.diff > overall[col].diff:
                overall[col] = w

    print(f"\n{len(files)} shared files, {n_bad} with a difference > {args.eps:g}\n")
    print(f"{'column':32s} {'max rel diff':>12s}  worst case")
    for col, w in sorted(overall.items(), key=lambda kv: -kv[1].diff):
        flag = "  <-- > eps" if w.diff > args.eps else ""
        print(f"{col:32s} {w.diff:12.2e}  {w.file} row {w.row}: {w.new!r} vs {w.old!r}{flag}")
    return 1 if n_bad else 0


if __name__ == "__main__":
    sys.exit(main())

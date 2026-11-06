"""Which scanned traits are not phenotypes, decided after the scan.

The deep trait scan (trait_scan.py) keeps every numeric field of every table,
including columns of derived ML tables that are not phenotypes:

    covariate proxy   |r| >= MAX_COVARIATE_R with sex or age over the people
                      measured (e.g. a weight-of-evidence encoding of sex, or a
                      women-only question's "nan" indicator, which marks men).
                      Sex and age are already in the model.
    missingness       one-hot "<field>_nan" indicators: whether a value was
                      missing, not what it was.
    not a phenotype   bare id / bookkeeping columns (code, index, fold, ...).

Ancestry-linked fields (birthplace, ethnic background, country of birth) are
kept but flagged, as they track ancestry more than biology.

Correlations need one pass over trait_values.npy (participants x traits); they
are cached next to it as trait_covariate_r.tsv.

    python trait_qc.py --scan DIR      # print the excluded traits
"""
from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

MAX_COVARIATE_R = 0.8
ROWS_PER_CHUNK = 1024   # x 18k traits x 8 B ~ 150 MB per temporary
CACHE_NAME = "trait_covariate_r.tsv"
BOOKKEEPING = re.compile(r"^(code|index|idx|id|fold|split|set|batch|group|row|"
                         r"unnamed(: ?\d+)?)$", re.I)
MISSINGNESS = re.compile(r"_nan$", re.I)
ANCESTRY_FIELDS = frozenset({"129", "130", "1647", "20115", "21000"})
ANCESTRY_NAME = re.compile(r"ethnic|ancestr|race|birth_?place|country_?of_?birth", re.I)


def base_field(field: str) -> str:
    """UKB field number of '21000-0.0_1002.0' / '30840' style names."""
    return re.split(r"[-_]", str(field))[0]


def covariate_r(values: np.ndarray, people: pd.DataFrame,
                log_every: float = 30.0) -> pd.DataFrame:
    """Pearson r of every trait column with sex and with age, over the rows where
    the trait is measured. One sequential pass in row chunks (the matrix is
    row-major and can be far larger than memory)."""
    n_rows, n_cols = values.shape
    covs = {"sex": people.sex.to_numpy(np.float64), "age": people.age.to_numpy(np.float64)}
    keys = ["n", "x", "xx"] + [f"{s}{c}" for c in covs for s in ("", "x_", "sq_")]
    acc = {k: np.zeros(n_cols) for k in keys}
    t0, last = time.monotonic(), time.monotonic()
    for start in range(0, n_rows, ROWS_PER_CHUNK):
        block = np.asarray(values[start:start + ROWS_PER_CHUNK], np.float64)
        mask = np.isfinite(block)
        x = np.where(mask, block, 0.0)
        acc["n"] += mask.sum(0)
        acc["x"] += x.sum(0)
        acc["xx"] += (x * x).sum(0)
        for name, cov in covs.items():
            c = cov[start:start + ROWS_PER_CHUNK, None]
            acc[name] += (mask * c).sum(0)
            acc[f"x_{name}"] += (x * c).sum(0)
            acc[f"sq_{name}"] += (mask * c * c).sum(0)
        if time.monotonic() - last > log_every:
            done = min(start + ROWS_PER_CHUNK, n_rows)
            elapsed = time.monotonic() - t0
            print(f"[trait_qc] covariate r: {done:,}/{n_rows:,} rows, elapsed "
                  f"{elapsed / 60:.1f}m, ETA {elapsed * (n_rows - done) / done / 60:.1f}m",
                  file=sys.stderr, flush=True)
            last = time.monotonic()
    out = {}
    n = np.maximum(acc["n"], 1)
    var_x = acc["xx"] / n - (acc["x"] / n) ** 2
    for name in covs:
        mean_c = acc[name] / n
        var_c = acc[f"sq_{name}"] / n - mean_c ** 2
        cov_xc = acc[f"x_{name}"] / n - (acc["x"] / n) * mean_c
        with np.errstate(divide="ignore", invalid="ignore"):
            r = cov_xc / np.sqrt(var_x * var_c)
        out[f"r_{name}"] = np.where((var_x > 1e-12) & (var_c > 1e-12), r, np.nan)
    return pd.DataFrame(out)


def load_covariate_r(scan: Path) -> pd.DataFrame:
    """trait_id, r_sex, r_age for every catalog trait; cached in the scan dir."""
    cache = scan / CACHE_NAME
    catalog = pd.read_csv(scan / "trait_catalog.tsv", sep="\t", dtype={"trait_id": str})
    if cache.is_file():
        cached = pd.read_csv(cache, sep="\t", dtype={"trait_id": str})
        if set(cached.trait_id) == set(catalog.trait_id):
            return cached
    people = pd.read_csv(scan / "participants.tsv", sep="\t")
    values = np.load(scan / "trait_values.npy", mmap_mode="r")
    r = covariate_r(values, people).iloc[catalog.column.to_numpy()].reset_index(drop=True)
    table = pd.concat([catalog[["trait_id"]], r], axis=1)
    tmp = cache.with_name(cache.name + ".tmp")
    table.to_csv(tmp, sep="\t", index=False)
    tmp.replace(cache)
    return table


def is_ancestry_linked(field: str) -> bool:
    """Birthplace / ethnicity / country of birth, by UKB field number or by name."""
    return base_field(field) in ANCESTRY_FIELDS or bool(ANCESTRY_NAME.search(str(field)))


def exclusion_reason(field: str, r_sex: float, r_age: float) -> str | None:
    name = str(field).split(":")[-1]
    if BOOKKEEPING.match(name):
        return "not a phenotype (id/bookkeeping column)"
    if MISSINGNESS.search(name):
        return "missingness indicator"
    for cov, r in (("sex", r_sex), ("age", r_age)):
        if np.isfinite(r) and abs(r) >= MAX_COVARIATE_R:
            return f"covariate proxy (r = {r:+.2f} with {cov})"
    return None


def classify(catalog: pd.DataFrame, covariate: pd.DataFrame) -> pd.DataFrame:
    """trait_id, field, r_sex, r_age, excluded (reason or ''), ancestry_linked."""
    table = catalog[["trait_id", "field"]].merge(covariate, on="trait_id", how="left")
    table["field"] = table.field.fillna(table.trait_id).astype(str)
    reasons = [exclusion_reason(f, rs, ra)
               for f, rs, ra in zip(table.field, table.r_sex, table.r_age)]
    return table.assign(excluded=[r or "" for r in reasons],
                        ancestry_linked=table.field.map(is_ancestry_linked))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--scan", required=True, help="trait_scan.py output directory")
    args = ap.parse_args(argv)
    scan = Path(args.scan)
    catalog = pd.read_csv(scan / "trait_catalog.tsv", sep="\t",
                          dtype={"trait_id": str, "field": str})
    table = classify(catalog, load_covariate_r(scan))
    dropped = table[table.excluded != ""]
    print(f"{len(dropped):,} of {len(table):,} traits excluded; "
          f"{int(table.ancestry_linked.sum())} flagged ancestry-linked")
    print(dropped.excluded.str.split(" \\(").str[0].value_counts().to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())

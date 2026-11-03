"""Test every scan gene against every scanned trait, All / F / M, all 60 embeddings.

Each worker loads a gene's cached PCA scores once (gene_pca_cache.py) and loops
over every trait and sex in memory. The tests are the vendored FM-GWAS
script's own (fmgwas/insample_hpp_AG.analytic_tests): partial F for
quantitative traits, logistic LRT for binary ones, reduced model = intercept +
standardized age and sex (age only in single-sex cells), full model = reduced +
the gene's PCA scores. The only change from run_assoc.sh is the one Ning asked
for: the PCA is fitted once per gene (on all scan participants) and reused,
not refitted inside each cell.

Work is checkpointed per gene and block of traits:
    <out>/per_gene/<gene>/part<k>.tsv.gz
so a rerun only does the blocks that are missing.

    python gene_scan.py --scan DIR --cache DIR --genes genes.tsv --out DIR \
        [--rank 0 --world-size 8] [--traits-per-part 50]
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from gene_pca_cache import fmgwas, load_cache

SEXES = (("All", None), ("F", 0), ("M", 1))
MIN_CELL_N = 500
MIN_CELL_MINOR = 20
LRT_ARGS = SimpleNamespace(lrt_max_iter=100, lrt_tol=1e-10)
OUT_COLUMNS = ["gene_id", "trait_id", "sex", "kind", "embed_type", "n_samples", "n_cases",
               "gene_pca_components", "df", "stat", "p", "status"]


def log(message: str) -> None:
    print(f"[{time.strftime('%F %T')}] [gene_scan] {message}", file=sys.stderr, flush=True)


class Scan:
    """trait_scan.py output: catalog, participant table and values (memory-mapped)."""

    def __init__(self, root: Path):
        self.catalog = pd.read_csv(root / "trait_catalog.tsv", sep="\t")
        self.people = pd.read_csv(root / "participants.tsv", sep="\t")
        self.values = np.load(root / "trait_values.npy", mmap_mode="r")
        if self.values.shape != (len(self.people), len(self.catalog)):
            raise ValueError(f"trait_values.npy is {self.values.shape}, expected "
                             f"({len(self.people)}, {len(self.catalog)})")


def cell_design(fm, scan: Scan, column: int, sex_code: int | None,
                gene_pos: np.ndarray) -> dict | None:
    """Cohort, outcome and reduced design of one trait x sex cell (as prepare_gene)."""
    y_all = np.asarray(scan.values[:, column], dtype=np.float64)
    sex = scan.people.sex.to_numpy()
    keep = np.isfinite(y_all) & (gene_pos >= 0)
    if sex_code is not None:
        keep &= sex == sex_code
    if keep.sum() < MIN_CELL_N:
        return None
    y = y_all[keep]
    cov_names = ["age", "sex"] if sex_code is None else ["age"]
    cov = fm.standardize(scan.people.loc[keep, cov_names].to_numpy(np.float64))[0]
    is_binary = bool(np.array_equal(np.unique(y), [0.0, 1.0]))
    if is_binary and min(y.sum(), len(y) - y.sum()) < MIN_CELL_MINOR:
        return None
    ids = scan.people.participant_id.to_numpy(np.int64)[keep]
    return dict(y=y, base=np.column_stack([np.ones(len(y)), cov]), gindex=gene_pos[keep],
                n=len(y), is_binary=is_binary, cohort_hash=fm.digest_arrays(ids))


def test_cell(fm, gene: str, trait: pd.Series, sex: str, cell: dict,
              scores: dict[str, np.ndarray], components: dict) -> list[dict]:
    tests = ["F"] if trait.kind == "quantitative" else ["LRT"]
    out = []
    for embed, matrix in scores.items():
        row = dict(gene_id=gene, embed_type=embed, trait=trait.trait_id, sex=sex,
                   n_samples=cell["n"], participant_ids_sha256=cell["cohort_hash"],
                   cell_pca_id="none")
        fm.analytic_tests(row, cell["y"], cell["base"], matrix[cell["gindex"]], tests,
                          cell["is_binary"], LRT_ARGS)
        f_test = tests == ["F"]
        out.append(dict(
            gene_id=gene, trait_id=trait.trait_id, sex=sex, kind=trait.kind,
            embed_type=embed, n_samples=cell["n"],
            n_cases=int(cell["y"].sum()) if cell["is_binary"] else None,
            gene_pca_components=components.get(embed),
            df=row.get("df1") if f_test else row.get("df_lrt"),
            stat=row.get("F_obs") if f_test else row.get("LRT_obs"),
            p=row.get("p_F") if f_test else row.get("p_LRT"),
            status=row.get("status")))
    return out


def run_part(fm, scan: Scan, gene: str, traits: pd.DataFrame, gene_pos: np.ndarray,
             scores: dict, components: dict) -> pd.DataFrame:
    rows: list[dict] = []
    for _, trait in traits.iterrows():
        for sex, code in SEXES:
            cell = cell_design(fm, scan, int(trait.column), code, gene_pos)
            if cell is None:
                continue
            rows.extend(test_cell(fm, gene, trait, sex, cell, scores, components))
    return pd.DataFrame(rows, columns=OUT_COLUMNS)


def atomic_tsv_gz(path: Path, frame: pd.DataFrame) -> None:
    tmp = path.with_name(path.name + ".tmp")
    frame.to_csv(tmp, sep="\t", index=False, compression="gzip")
    os.replace(tmp, path)


def scan_gene(fm, scan: Scan, cache: Path, gene: str, out: Path, part_size: int) -> int:
    """Run the missing parts of one gene; returns how many parts ran."""
    gene_dir = out / "per_gene" / gene
    parts = [scan.catalog.iloc[i:i + part_size] for i in range(0, len(scan.catalog), part_size)]
    todo = [k for k in range(len(parts)) if not (gene_dir / f"part{k}.tsv.gz").exists()]
    if not todo:
        return 0
    ids, scores, meta = load_cache(cache, gene)
    if not scores:
        log(f"[FAILED] {gene}: no cached PCA ({meta.get('error', 'no embeddings')})")
        return 0
    pos = {int(p): i for i, p in enumerate(ids)}
    gene_pos = np.array([pos.get(int(p), -1) for p in scan.people.participant_id], np.int64)
    gene_dir.mkdir(parents=True, exist_ok=True)
    for k in todo:
        atomic_tsv_gz(gene_dir / f"part{k}.tsv.gz",
                      run_part(fm, scan, gene, parts[k], gene_pos, scores,
                               meta["components"]))
    return len(todo)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--scan", required=True, help="trait_scan.py output directory")
    ap.add_argument("--cache", required=True, help="gene_pca_cache.py output directory")
    ap.add_argument("--genes", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--world-size", type=int, default=1)
    ap.add_argument("--traits-per-part", type=int, default=50)
    args = ap.parse_args(argv)

    fm = fmgwas()
    scan = Scan(Path(args.scan))
    genes = fm.read_genes(args.genes)[args.rank::args.world_size]
    out = Path(args.out)
    log(f"rank {args.rank}/{args.world_size}: {len(genes)} genes x {len(scan.catalog)} traits")
    start, ran = time.monotonic(), 0
    for number, gene in enumerate(genes, 1):
        t0 = time.monotonic()
        parts = scan_gene(fm, scan, Path(args.cache), gene, out, args.traits_per_part)
        if parts:
            ran += 1
            eta = (time.monotonic() - start) / ran * (len(genes) - number)
            log(f"[{number}/{len(genes)}] {gene}: {parts} part(s) in "
                f"{(time.monotonic() - t0) / 60:.1f} min, ETA {eta / 3600:.1f} h")
    return 0


if __name__ == "__main__":
    sys.exit(main())

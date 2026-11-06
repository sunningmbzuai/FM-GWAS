"""Test every scan gene against every scanned trait, All / F / M, all 60 embeddings.

Each worker loads a gene's cached PCA scores once (gene_pca_cache.py) and loops
over every trait and sex in memory. The tests are the vendored FM-GWAS
script's own (fmgwas/insample_hpp_AG.analytic_tests): partial F for
quantitative traits, logistic LRT for binary ones, reduced model = intercept +
standardized age and sex (age only in single-sex cells), full model = reduced +
the gene's PCA scores. The only change from run_assoc.sh is the one Ning asked
for: the PCA is fitted once per gene and sex sample (All, F, M) and reused
for every trait of that sex, not refitted inside each trait cell.

Genetic PCs are never tested as traits (trait_rules.is_genetic_pc). The default
covariates are Ning's (FM-GWAS paper and script: age and sex, no genetic PCs).
--genetic-pcs N adds the scan's genetic PCs 1..N to the reduced model, a
sensitivity analysis for ancestry confounding; write it to its own --out.

Work is checkpointed per gene and block of traits:
    <out>/per_gene/<gene>/part<k>.tsv.gz
so a rerun only does the blocks that are missing.

    python gene_scan.py --scan DIR --cache DIR --genes genes.tsv --out DIR \
        [--rank 0 --world-size 8] [--traits-per-part 50] [--genetic-pcs 10]
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

import trait_rules as R
from gene_pca_cache import SEXES, fmgwas, load_cache

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

    def genetic_pcs(self, n: int) -> np.ndarray | None:
        """participants x n matrix of genetic PCs 1..n from the scan, None if n == 0."""
        if n <= 0:
            return None
        numbers = self.catalog.trait_id.map(R.genetic_pc_number)
        column = {int(k): int(c) for k, c in zip(numbers, self.catalog.column) if pd.notna(k)}
        missing = [k for k in range(1, n + 1) if k not in column]
        if missing:
            raise ValueError(f"genetic PC(s) {missing} not in the trait catalog "
                             f"({len(column)} found)")
        return np.column_stack([np.asarray(self.values[:, column[k]], np.float64)
                                for k in range(1, n + 1)])


def cell_design(fm, scan: Scan, column: int, sex_code: int | None,
                gene_pos: np.ndarray, pcs: np.ndarray | None = None) -> dict | None:
    """Cohort, outcome and reduced design of one trait x sex cell (as prepare_gene).
    pcs: optional participants x k genetic PCs, added to the covariates."""
    y_all = np.asarray(scan.values[:, column], dtype=np.float64)
    sex = scan.people.sex.to_numpy()
    keep = np.isfinite(y_all) & (gene_pos >= 0)
    if pcs is not None:
        keep &= np.isfinite(pcs).all(axis=1)
    if sex_code is not None:
        keep &= sex == sex_code
    if keep.sum() < MIN_CELL_N:
        return None
    y = y_all[keep]
    cov_names = ["age", "sex"] if sex_code is None else ["age"]
    cov = scan.people.loc[keep, cov_names].to_numpy(np.float64)
    if pcs is not None:
        cov = np.column_stack([cov, pcs[keep]])
    cov = fm.standardize(cov)[0]
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


def positions(scan: Scan, ids: np.ndarray) -> np.ndarray:
    """Row of each scan participant in a cached sample, -1 when absent."""
    pos = {int(p): i for i, p in enumerate(ids)}
    return np.array([pos.get(int(p), -1) for p in scan.people.participant_id], np.int64)


def run_part(fm, scan: Scan, gene: str, traits: pd.DataFrame, samples: dict,
             pcs: np.ndarray | None = None) -> pd.DataFrame:
    """samples: {sex: (gene_pos, scores, components)} -- each sex its own PCA."""
    rows: list[dict] = []
    for _, trait in traits.iterrows():
        if R.is_genetic_pc(trait.trait_id):
            continue
        for sex, code in SEXES:
            if sex not in samples:
                continue
            gene_pos, scores, components = samples[sex]
            cell = cell_design(fm, scan, int(trait.column), code, gene_pos, pcs)
            if cell is None:
                continue
            rows.extend(test_cell(fm, gene, trait, sex, cell, scores, components))
    return pd.DataFrame(rows, columns=OUT_COLUMNS)


def atomic_tsv_gz(path: Path, frame: pd.DataFrame) -> None:
    tmp = path.with_name(path.name + ".tmp")
    frame.to_csv(tmp, sep="\t", index=False, compression="gzip")
    os.replace(tmp, path)


def scan_gene(fm, scan: Scan, cache: Path, gene: str, out: Path, part_size: int,
              pcs: np.ndarray | None = None) -> int:
    """Run the missing parts of one gene; returns how many parts ran."""
    gene_dir = out / "per_gene" / gene
    parts = [scan.catalog.iloc[i:i + part_size] for i in range(0, len(scan.catalog), part_size)]
    todo = [k for k in range(len(parts)) if not (gene_dir / f"part{k}.tsv.gz").exists()]
    if not todo:
        return 0
    by_sex, meta = load_cache(cache, gene)
    if not by_sex:
        log(f"[FAILED] {gene}: no cached PCA ({meta.get('error', 'no embeddings')})")
        return 0
    samples = {sex: (positions(scan, ids), scores, meta["sexes"][sex]["components"])
               for sex, (ids, scores) in by_sex.items()}
    gene_dir.mkdir(parents=True, exist_ok=True)
    for k in todo:
        atomic_tsv_gz(gene_dir / f"part{k}.tsv.gz", run_part(fm, scan, gene, parts[k], samples, pcs))
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
    ap.add_argument("--genetic-pcs", type=int, default=0,
                    help="genetic PCs 1..N as extra covariates (default 0: Ning's model)")
    args = ap.parse_args(argv)

    fm = fmgwas()
    scan = Scan(Path(args.scan))
    genes = fm.read_genes(args.genes)[args.rank::args.world_size]
    out = Path(args.out)
    pcs = scan.genetic_pcs(args.genetic_pcs)
    log(f"covariates: age, sex" + (f", genetic PCs 1-{args.genetic_pcs}" if pcs is not None else ""))
    log(f"rank {args.rank}/{args.world_size}: {len(genes)} genes x {len(scan.catalog)} traits")
    start, ran = time.monotonic(), 0
    for number, gene in enumerate(genes, 1):
        t0 = time.monotonic()
        parts = scan_gene(fm, scan, Path(args.cache), gene, out, args.traits_per_part, pcs)
        if parts:
            ran += 1
            eta = (time.monotonic() - start) / ran * (len(genes) - number)
            log(f"[{number}/{len(genes)}] {gene}: {parts} part(s) in "
                f"{(time.monotonic() - t0) / 60:.1f} min, ETA {eta / 3600:.1f} h")
    return 0


if __name__ == "__main__":
    sys.exit(main())

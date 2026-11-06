"""Collapse gene_scan.py output to one row per gene x trait x sex.

For each cell: the minimum p over the 60 embeddings, the embedding it came
from, how many embeddings returned a valid p, and whether the minimum passes
the FM-GWAS genome-wide threshold (2.5e-6) and a Bonferroni threshold over
every gene x trait x sex cell in the scan.

Genetic PCs (trait_rules.is_genetic_pc) are ancestry axes, not traits: their
cells are dropped. Many tables repeat the same UKB field (derived train/test
splits of the main dataset), so a second table keeps one row per gene x field x
sex: the table with the smallest p, plus how many tables carried the field.

With --scan (the trait_scan.py directory), traits that are not phenotypes are
dropped as well -- covariate proxies, missingness indicators, bookkeeping
columns (trait_qc.py) -- and listed with the reason in
gene_trait_scan.excluded_traits.tsv; ancestry-linked fields are flagged.

    python summarize_gene_scan.py --out DIR [--catalog trait_catalog.tsv] [--workers N]
                                  [--scan TRAIT_SCAN_DIR] [--from-summary]
writes, under DIR:
    gene_trait_scan.tsv              gene x trait x sex
    gene_trait_scan.hits.tsv         ... at p < 2.5e-6
    gene_trait_scan.by_field.tsv     gene x UKB field x sex
    gene_trait_scan.by_field.hits.tsv
--from-summary rebuilds all four from an existing gene_trait_scan.tsv instead
of re-reading the per-gene parts.

Parts are read in parallel (one per worker process; default: all cores but 2)
with a live progress / ETA line on stderr.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

import trait_qc
import trait_rules as R

GENOME_WIDE = 2.5e-6
KEYS = ["gene_id", "trait_id", "sex"]
PART_DTYPES = {"gene_id": str, "trait_id": str, "sex": str, "kind": str,
               "embed_type": str, "status": str}
CELL_COLUMNS = KEYS + ["kind", "n_samples", "n_cases", "p_min", "best_embedding",
                       "best_stat", "best_df", "best_components", "status",
                       "n_embeddings_tested"]
FIELD_KEYS = ["gene_id", "field", "sex"]
SPARE_CORES = 2
PROGRESS_EVERY = 500   # parts between progress lines


def best_per_cell(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.assign(p=pd.to_numeric(frame.p, errors="coerce"))
    valid = frame.dropna(subset=["p"])
    n_valid = valid.groupby(KEYS).size().rename("n_embeddings_tested")
    best = valid.loc[valid.groupby(KEYS).p.idxmin()].set_index(KEYS)
    best = best.rename(columns={"p": "p_min", "embed_type": "best_embedding",
                                "stat": "best_stat", "df": "best_df",
                                "gene_pca_components": "best_components"})
    cols = ["kind", "n_samples", "n_cases", "p_min", "best_embedding", "best_stat",
            "best_df", "best_components", "status"]
    return best[cols].join(n_valid).reset_index()


def summarize_part(path: Path) -> pd.DataFrame:
    return best_per_cell(pd.read_csv(path, sep="\t", dtype=PART_DTYPES))


def default_workers() -> int:
    return max(1, (os.cpu_count() or 1) - SPARE_CORES)


def read_parts(parts: list[Path], workers: int) -> list[pd.DataFrame]:
    """best_per_cell of every part, in order, with a progress / ETA line."""
    t0, total, frames = time.time(), len(parts), []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for i, frame in enumerate(pool.map(summarize_part, parts, chunksize=64), 1):
            frames.append(frame)
            if i % PROGRESS_EVERY == 0 or i == total:
                elapsed = time.time() - t0
                eta = elapsed * (total - i) / i
                print(f"[summary] {i}/{total} parts ({100 * i // total}%), "
                      f"elapsed {elapsed / 60:.1f}m, ETA {eta / 60:.1f}m",
                      file=sys.stderr, flush=True)
    return frames


def read_cells(out: Path, workers: int, from_summary: bool) -> pd.DataFrame:
    """One row per gene x trait x sex: from the parts, or from a previous summary."""
    previous = out / "gene_trait_scan.tsv"
    if from_summary:
        if not previous.is_file():
            raise SystemExit(f"--from-summary: no {previous}")
        print(f"[summary] rebuilding from {previous}", file=sys.stderr, flush=True)
        return pd.read_csv(previous, sep="\t", dtype=PART_DTYPES)[CELL_COLUMNS]
    parts = sorted((out / "per_gene").glob("*/part*.tsv.gz"))
    if not parts:
        raise SystemExit(f"no results under {out / 'per_gene'}")
    print(f"[summary] {len(parts)} parts, {workers} worker(s)", file=sys.stderr, flush=True)
    return pd.concat(read_parts(parts, workers), ignore_index=True)


def flag(table: pd.DataFrame) -> pd.DataFrame:
    """Genome-wide and Bonferroni (over this table's rows) flags, sorted by p."""
    n = max(len(table), 1)
    return (table.assign(genome_wide=table.p_min < GENOME_WIDE,
                         bonferroni=table.p_min < 0.05 / n)
            .sort_values(["p_min", "best_stat"], ascending=[True, False]))


def by_field(cells: pd.DataFrame) -> pd.DataFrame:
    """One row per gene x field x sex: the table with the smallest p (ties: largest
    statistic), and how many tables carried that field."""
    keyed = cells.assign(field=cells.field.fillna(cells.trait_id).astype(str))
    ranked = keyed.sort_values(["p_min", "best_stat"], ascending=[True, False])
    best = ranked.drop_duplicates(FIELD_KEYS).set_index(FIELD_KEYS)
    tables = keyed.groupby(FIELD_KEYS).trait_id.agg(n_tables="size",
                                                   tables=lambda t: ";".join(sorted(t)))
    return best.drop(columns=["genome_wide", "bonferroni"], errors="ignore") \
        .join(tables).reset_index()


def write(table: pd.DataFrame, out: Path, stem: str) -> None:
    table.to_csv(out / f"{stem}.tsv", sep="\t", index=False)
    table[table.genome_wide].to_csv(out / f"{stem}.hits.tsv", sep="\t", index=False)


def report(name: str, table: pd.DataFrame) -> None:
    print(f"{name}: {len(table):,} rows, {int(table.genome_wide.sum()):,} at p < "
          f"{GENOME_WIDE:g} ({table[table.genome_wide].gene_id.nunique()} genes), "
          f"{int(table.bonferroni.sum()):,} at Bonferroni 0.05/{len(table):,}")


def drop_non_phenotypes(cells: pd.DataFrame, scan: Path, out: Path) -> pd.DataFrame:
    """trait_qc exclusions out, ancestry flag in; excluded traits listed in a file."""
    catalog = pd.read_csv(scan / "trait_catalog.tsv", sep="\t",
                          dtype={"trait_id": str, "field": str})
    qc = trait_qc.classify(catalog, trait_qc.load_covariate_r(scan))
    excluded = qc[(qc.excluded != "") & qc.trait_id.isin(cells.trait_id)]
    excluded.to_csv(out / "gene_trait_scan.excluded_traits.tsv", sep="\t", index=False)
    dropped = cells.trait_id.isin(excluded.trait_id)
    print(f"[summary] dropped {int(dropped.sum()):,} cells of {len(excluded):,} non-phenotype "
          f"traits (see gene_trait_scan.excluded_traits.tsv)", file=sys.stderr, flush=True)
    flags = qc.set_index("trait_id").ancestry_linked
    kept = cells[~dropped]
    return kept.assign(ancestry_linked=kept.trait_id.map(flags).fillna(False).astype(bool))


def summarize(out: Path, catalog: Path | None, workers: int = 1,
              from_summary: bool = False, scan: Path | None = None) -> pd.DataFrame:
    cells = read_cells(out, workers, from_summary)
    pcs = cells.trait_id.map(R.is_genetic_pc)
    if pcs.any():
        print(f"[summary] dropped {int(pcs.sum()):,} genetic-PC cells (ancestry, not traits)",
              file=sys.stderr, flush=True)
    cells = cells[~pcs]
    if scan is not None:
        cells = drop_non_phenotypes(cells, scan, out)
    if catalog is not None and catalog.is_file():
        meta = pd.read_csv(catalog, sep="\t", dtype={"trait_id": str, "field": str})
        cells = cells.merge(meta[["trait_id", "title", "path", "field"]],
                            on="trait_id", how="left")
    else:
        cells = cells.assign(title=None, path=None, field=None)
    table = flag(cells)
    write(table, out, "gene_trait_scan")
    fields = flag(by_field(table))
    write(fields, out, "gene_trait_scan.by_field")
    report("gene x trait x sex", table)
    report("gene x UKB field x sex", fields)
    return table


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--catalog")
    ap.add_argument("--workers", type=int, default=default_workers())
    ap.add_argument("--scan", help="trait_scan.py directory: drop non-phenotype traits")
    ap.add_argument("--from-summary", action="store_true",
                    help="rebuild from an existing gene_trait_scan.tsv, skip the parts")
    args = ap.parse_args(argv)
    summarize(Path(args.out), Path(args.catalog) if args.catalog else None,
              max(1, args.workers), args.from_summary,
              Path(args.scan) if args.scan else None)
    return 0


if __name__ == "__main__":
    sys.exit(main())

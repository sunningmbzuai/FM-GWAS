"""Quick read of a gene_scan summary (summarize_gene_scan.py output), one row per
gene x UKB field x sex, genetic PCs already dropped.

Two thresholds: Ning's genome-wide 2.5e-6 (the FM-GWAS paper's), and a strict
Bonferroni over every row AND the 60 embeddings the minimum p was taken over.

    python top_hits.py <results dir>   # dir holding gene_trait_scan.by_field.tsv
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

N_EMBEDDINGS = 60
ALPHA = 0.05
GENOME_WIDE = 2.5e-6
STEM = "gene_trait_scan.by_field"
COLS = ["gene_id", "sex", "trait", "n_tables", "p_min", "best_stat", "best_embedding",
        "n_samples", "n_cases"]
TOP_N = 30


def count_rows(path: Path) -> int:
    with open(path) as handle:
        return sum(1 for _ in handle) - 1


def label(hits: pd.DataFrame) -> pd.Series:
    """Field title when the scan found a UKB dictionary, else the UKB field id."""
    title = hits["title"] if "title" in hits else pd.Series(index=hits.index, dtype=str)
    return title.where(title.notna() & (title != ""), hits.field.fillna(hits.trait_id))


def show(name: str, series: pd.Series) -> None:
    print(f"{name}:")
    print(series.sort_values(ascending=False).head(TOP_N).to_string(), "\n")


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        print(__doc__, file=sys.stderr)
        return 2
    results = Path(argv[0])
    n_rows = count_rows(results / f"{STEM}.tsv")
    hits = pd.read_csv(results / f"{STEM}.hits.tsv", sep="\t",
                       dtype={"trait_id": str, "field": str})
    hits = hits.assign(trait=label(hits))
    strict_p = ALPHA / (n_rows * N_EMBEDDINGS)
    strict = hits[hits.p_min < strict_p]
    print(f"{n_rows:,} gene x field x sex rows")
    print(f"  p < {GENOME_WIDE:g} (FM-GWAS):        {len(hits):,} rows, "
          f"{hits.gene_id.nunique()} genes, {hits.field.nunique()} fields")
    print(f"  p < {strict_p:.2e} (x {N_EMBEDDINGS} embeddings): {len(strict):,} rows, "
          f"{strict.gene_id.nunique()} genes, {strict.field.nunique()} fields; "
          f"{int((strict.p_min == 0).sum())} below float range (p shown as 0)")
    if "ancestry_linked" in hits:
        anc = hits[hits.ancestry_linked.astype(bool)]
        print(f"  ancestry-linked fields (birthplace, ethnicity, country of birth): "
              f"{len(anc):,} of the FM-GWAS hits, {anc.gene_id.nunique()} genes")
    print()
    show("fields per gene (FM-GWAS threshold)", hits.groupby("gene_id").field.nunique())
    show("genes per field (FM-GWAS threshold)", hits.groupby("trait").gene_id.nunique())
    top = hits.sort_values(["p_min", "best_stat"], ascending=[True, False])
    print(f"top {TOP_N} (p = 0 means below 1e-308; ranked by statistic):")
    print(top[COLS].head(TOP_N).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

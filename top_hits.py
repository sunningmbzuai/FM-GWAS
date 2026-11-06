"""Quick read of a gene_scan summary: hits that survive Bonferroni over every
gene x trait x sex cell AND the 60 embeddings the minimum p was taken over.

    python top_hits.py <results dir>   # dir holding gene_trait_scan.tsv
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

N_EMBEDDINGS = 60
ALPHA = 0.05
COLS = ["gene_id", "sex", "trait", "p_min", "best_stat", "best_embedding", "n_samples",
        "n_cases"]


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        print(__doc__, file=sys.stderr)
        return 2
    results = Path(argv[0])
    n_cells = sum(1 for _ in open(results / "gene_trait_scan.tsv")) - 1
    hits = pd.read_csv(results / "gene_trait_scan.hits.tsv", sep="\t",
                       dtype={"trait_id": str})
    # Titles are empty when trait_scan found no UKB Showcase dictionary:
    # fall back to the trait id (table:field).
    title = hits["title"] if "title" in hits else pd.Series(index=hits.index, dtype=str)
    hits = hits.assign(trait=title.where(title.notna() & (title != ""), hits.trait_id))
    threshold = ALPHA / (n_cells * N_EMBEDDINGS)
    strict = hits[hits.p_min < threshold]
    print(f"{n_cells:,} cells; {len(hits):,} at p < 2.5e-6; threshold "
          f"{ALPHA}/({n_cells:,} x {N_EMBEDDINGS}) = {threshold:.2e}")
    print(f"{len(strict):,} cells pass it, {strict.gene_id.nunique()} genes, "
          f"{strict.trait_id.nunique()} traits; "
          f"{int((strict.p_min == 0).sum())} with p below float range (shown as 0)\n")
    print("cells per gene:")
    print(strict.groupby("gene_id").size().sort_values(ascending=False)
          .head(20).to_string(), "\n")
    print("traits per gene (top 20 genes):")
    print(strict.groupby("gene_id").trait_id.nunique().sort_values(ascending=False)
          .head(20).to_string(), "\n")
    print("cells per trait:")
    print(strict.groupby("trait").size().sort_values(ascending=False)
          .head(30).to_string(), "\n")
    # p underflows to 0 for the strongest cells: break ties on the statistic.
    top = strict.sort_values(["p_min", "best_stat"], ascending=[True, False])
    print("top 30 cells (p = 0 means below 1e-308; ranked by statistic):")
    print(top[COLS].head(30).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

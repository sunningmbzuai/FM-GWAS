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
COLS = ["gene_id", "sex", "title", "p_min", "best_embedding", "n_samples", "n_cases"]


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        print(__doc__, file=sys.stderr)
        return 2
    results = Path(argv[0])
    n_cells = sum(1 for _ in open(results / "gene_trait_scan.tsv")) - 1
    hits = pd.read_csv(results / "gene_trait_scan.hits.tsv", sep="\t")
    threshold = ALPHA / (n_cells * N_EMBEDDINGS)
    strict = hits[hits.p_min < threshold]
    print(f"{n_cells:,} cells; {len(hits):,} at p < 2.5e-6; threshold "
          f"{ALPHA}/({n_cells:,} x {N_EMBEDDINGS}) = {threshold:.2e}")
    print(f"{len(strict):,} cells pass it, {strict.gene_id.nunique()} genes, "
          f"{strict.title.nunique()} distinct trait titles\n")
    print("cells per gene:")
    print(strict.groupby("gene_id").size().sort_values(ascending=False)
          .head(20).to_string(), "\n")
    print("top 30 cells:")
    print(strict.sort_values("p_min")[COLS].head(30).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

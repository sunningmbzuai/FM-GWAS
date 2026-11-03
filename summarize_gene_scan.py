"""Collapse gene_scan.py output to one row per gene x trait x sex.

For each cell: the minimum p over the 60 embeddings, the embedding it came
from, how many embeddings returned a valid p, and whether the minimum passes
the FM-GWAS genome-wide threshold (2.5e-6) and a Bonferroni threshold over
every gene x trait x sex cell in the scan.

    python summarize_gene_scan.py --out DIR [--catalog trait_catalog.tsv]
writes DIR/gene_trait_scan.tsv and DIR/gene_trait_scan.hits.tsv.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

GENOME_WIDE = 2.5e-6
KEYS = ["gene_id", "trait_id", "sex"]


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


def summarize(out: Path, catalog: Path | None) -> pd.DataFrame:
    parts = sorted((out / "per_gene").glob("*/part*.tsv.gz"))
    if not parts:
        raise SystemExit(f"no results under {out / 'per_gene'}")
    table = pd.concat([best_per_cell(pd.read_csv(p, sep="\t")) for p in parts],
                      ignore_index=True)
    n_cells = len(table)
    table["genome_wide"] = table.p_min < GENOME_WIDE
    table["bonferroni"] = table.p_min < 0.05 / max(n_cells, 1)
    if catalog is not None and catalog.is_file():
        meta = pd.read_csv(catalog, sep="\t")[["trait_id", "title", "path", "field"]]
        table = table.merge(meta, on="trait_id", how="left")
    table = table.sort_values("p_min")
    table.to_csv(out / "gene_trait_scan.tsv", sep="\t", index=False)
    table[table.genome_wide].to_csv(out / "gene_trait_scan.hits.tsv", sep="\t", index=False)
    print(f"{n_cells:,} gene x trait x sex cells from {len(parts)} part file(s); "
          f"{int(table.genome_wide.sum())} at p < {GENOME_WIDE:g}, "
          f"{int(table.bonferroni.sum())} at Bonferroni 0.05/{n_cells:,}")
    return table


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--catalog")
    args = ap.parse_args(argv)
    summarize(Path(args.out), Path(args.catalog) if args.catalog else None)
    return 0


if __name__ == "__main__":
    sys.exit(main())

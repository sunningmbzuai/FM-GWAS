"""Report one gene's HPP list membership and UKB p-values in every trait/sex cell.

Answers "was this hit tested in the same sex group as the HPP analysis?".

Usage:
    python check_gene_by_sex.py                          # SLC16A13, Cholesterol
    python check_gene_by_sex.py ENSG00000174327 Cholesterol
"""
import glob
import os
import sys

import pandas as pd

import config
from compute_recovery import (GENE_ID_COLUMN, P_COLUMN, SIGNIFICANCE_THRESHOLD,
                              strip_version)

DEFAULT_GENE = "ENSG00000174327"  # SLC16A13
DEFAULT_TRAIT = "Cholesterol"
TOP_ROWS = 20


def in_hpp_list(gene: str, trait: str, sex_name: str) -> str:
    suffix = config.SEX_SUFFIXES[sex_name]
    path = os.path.join(config.HPP_UNION_GENE_DIR, f"{trait}_extreme{suffix}.tsv")
    if not os.path.exists(path):
        return "no HPP list file"
    frame = pd.read_csv(path, sep="\t")
    if GENE_ID_COLUMN not in frame.columns:
        return f"no {GENE_ID_COLUMN} column in {path}"
    ids = {strip_version(g) for g in frame[GENE_ID_COLUMN].dropna()}
    return "YES" if gene in ids else "no"


def ukb_rows(gene: str, trait: str, sex_name: str) -> pd.DataFrame | None:
    result_dir = os.path.join(config.ASSOC_DIR, f"{trait}_{sex_name}")
    for path in sorted(glob.glob(os.path.join(result_dir, "*.tsv"))):
        frame = pd.read_csv(path, sep="\t")
        if GENE_ID_COLUMN in frame.columns:
            ids = {strip_version(g) for g in frame[GENE_ID_COLUMN].dropna()}
        else:
            ids = {strip_version(os.path.splitext(os.path.basename(path))[0])}
        if gene in ids:
            if GENE_ID_COLUMN in frame.columns:
                frame = frame[frame[GENE_ID_COLUMN].map(strip_version) == gene]
            return frame
    return None


def main() -> None:
    gene = strip_version(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_GENE
    trait = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_TRAIT
    print(f"gene={gene} trait={trait} threshold={SIGNIFICANCE_THRESHOLD:g}\n")

    for sex_name in config.SEX_SUFFIXES:
        print(f"===== {trait} {sex_name} =====")
        print(f"in HPP significant list: {in_hpp_list(gene, trait, sex_name)}")
        frame = ukb_rows(gene, trait, sex_name)
        if frame is None:
            print("UKB: not tested\n")
            continue
        if P_COLUMN not in frame.columns:
            print(f"UKB: no {P_COLUMN} column: {list(frame.columns)}\n")
            continue
        p = pd.to_numeric(frame[P_COLUMN], errors="coerce")
        n_sig = int((p < SIGNIFICANCE_THRESHOLD).sum())
        print(f"UKB: tested, min p={p.min():.3g}, "
              f"{n_sig}/{len(p)} views significant")
        print(frame.assign(**{P_COLUMN: p}).sort_values(P_COLUMN)
              .head(TOP_ROWS).to_string(index=False))
        print()


if __name__ == "__main__":
    main()

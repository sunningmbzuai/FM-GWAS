"""Per-gene replication under two label sets, side by side.

Ning's 2026-11-02 request reruns the disease genes with the HPP-aligned
labels (ukb_traits_hpp.py). This joins the new replication table with the
earlier one(s) made under the paper labels, so each (gene, trait, sex) shows
its cohort, case count and minimum p under both, plus the export's Bonferroni
flag (which genes pass Bonferroni in HPP and which only FDR).

    python compare_label_runs.py --new NEW.tsv --old OLD1.tsv [OLD2.tsv ...] \
        --export novel_genes_fdr_1026.tsv --out OUT.tsv \
        [--new-name hpp --old-name paper]

Old tables are searched in order; the first holding a key wins (the FDR-only
genes are in the 1026 run, the Bonferroni genes in the 1019 run).
"""
import argparse
import os

import pandas as pd

KEYS = ["bare_gene_id", "trait", "sex"]
CARRIED = ["n_samples", "n_cases", "ukbb_p_min", "ukbb_best_embed_type",
           "replicated_genomewide", "replicated_targeted"]


def with_bare_id(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.assign(bare_gene_id=frame["gene_id"].astype(str)
                        .str.split(".").str[0])


def first_hit(tables: list[pd.DataFrame]) -> pd.DataFrame:
    """Rows of every table, keeping the first table's row for a shared key."""
    if not tables:
        return pd.DataFrame(columns=KEYS + CARRIED)
    stacked = pd.concat([with_bare_id(t) for t in tables], ignore_index=True)
    return stacked.drop_duplicates(KEYS, keep="first")


def compare(new: pd.DataFrame, olds: list[pd.DataFrame], export: pd.DataFrame,
            new_name: str = "hpp", old_name: str = "paper") -> pd.DataFrame:
    """One row per new target: Bonferroni flag, then old vs new columns."""
    new = with_bare_id(new)
    old = first_hit(olds).reindex(columns=KEYS + CARRIED)
    flags = with_bare_id(export)[KEYS + ["Bonferroni"]].drop_duplicates(KEYS)
    out = (new[["gene_id", *KEYS, *CARRIED]]
           .merge(flags, on=KEYS, how="left")
           .merge(old, on=KEYS, how="left", suffixes=(f"_{new_name}", f"_{old_name}")))
    ordered = ["gene_id", "trait", "sex", "Bonferroni"] + [
        f"{c}_{s}" for c in CARRIED for s in (old_name, new_name)]
    return out[ordered].sort_values(["trait", "sex", f"ukbb_p_min_{new_name}"])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--new", required=True)
    ap.add_argument("--old", nargs="*", default=[])
    ap.add_argument("--export", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--new-name", default="hpp",
                    help="column suffix for the new run (default: hpp)")
    ap.add_argument("--old-name", default="paper",
                    help="column suffix for the earlier run(s) (default: paper)")
    args = ap.parse_args()

    olds = []
    for path in args.old:
        if os.path.exists(path):
            olds.append(pd.read_csv(path, sep="\t"))
        else:
            print(f"[warn] earlier table not found, skipped: {path}")
    table = compare(pd.read_csv(args.new, sep="\t"), olds,
                    pd.read_csv(args.export, sep="\t"),
                    new_name=args.new_name, old_name=args.old_name)
    table.to_csv(args.out, sep="\t", index=False)
    print(table.to_string(index=False, float_format=lambda v: f"{v:.3g}"))
    print(f"\nWrote {len(table)} rows to {args.out}")


if __name__ == "__main__":
    main()

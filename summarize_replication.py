"""UKBB replication of Ning's HPP target list: the three tables she asked for.

For each (gene, trait, sex) in config.TARGET_GENES_TSV it reads that cell's
insample_hpp_AG.py output under config.ASSOC_DIR and writes

    <out>.per_embedding.tsv  UKBB p_F_analytic per embedding type (long form)
    <out>.tsv                one row per target: HPP p, best UKBB p and the
                             embedding type it came from, PCs kept, variant
                             counts, replication flags
    <out>.rates.tsv          replication rate over all targets and over the
                             ones UKBB could test

Two replication thresholds, both on the minimum p across embedding types:
  genomewide   p < 2.5e-6, the discovery threshold compute_recovery.py uses,
               so this matches every earlier recovery number
  targeted     p < 0.05 / n_targets, Bonferroni over the targets alone. The
               min over embedding types is not corrected for, so treat this
               one as lenient.

Variant counts come from count_gene_variants.py's output (pass --variants).

    python summarize_replication.py --variants ukbb_variants_per_gene.tsv \
        --out ukbb_replication_1003
"""
import argparse
import glob
import os

import pandas as pd

import config
from compute_recovery import SIGNIFICANCE_THRESHOLD, strip_version

TARGETED_ALPHA = 0.05
PCA_SUFFIX = "_pca"
STATUS_TESTED = "tested"
STATUS_NOT_TESTED = "no_ukbb_result"
VARIANT_COLUMNS = ["chrom", "start", "end", "length_bp", "n_variants",
                   "n_polymorphic", "variant_status"]


def find_result(assoc_dir: str, trait: str, sex: str, gene_id: str) -> str | None:
    """The cell's result tsv for this gene, whatever version suffix it carries."""
    cell = os.path.join(assoc_dir, f"{trait}_{sex}")
    hits = glob.glob(os.path.join(cell, f"{strip_version(gene_id)}.*tsv"))
    hits += glob.glob(os.path.join(cell, f"{strip_version(gene_id)}.tsv"))
    return sorted(set(hits))[0] if hits else None


def per_embedding_p(targets: pd.DataFrame, assoc_dir: str) -> pd.DataFrame:
    """Long table: target gene_id, trait, sex, embed_type, num_feature, p."""
    frames = []
    for row in targets.itertuples(index=False):
        path = find_result(assoc_dir, row.trait, row.sex, row.gene_id)
        if path is None:
            continue
        result = pd.read_csv(path, sep="\t")
        if result.empty:
            continue
        frames.append(pd.DataFrame({
            "gene_id": row.gene_id,
            "trait": row.trait,
            "sex": row.sex,
            "embed_type": result["embed_type"].str.split(PCA_SUFFIX).str[0],
            "num_feature": result["num_feature"],
            "ukbb_p": result["p_F_analytic"],
        }))
    columns = ["gene_id", "trait", "sex", "embed_type", "num_feature", "ukbb_p"]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=columns)


def per_gene_summary(targets: pd.DataFrame, long: pd.DataFrame,
                     counts: pd.DataFrame | None) -> pd.DataFrame:
    keys = ["gene_id", "trait", "sex"]
    base = targets[keys + ["p_F_analytic_min"]].rename(
        columns={"p_F_analytic_min": "hpp_p_min"})
    if long.empty:
        best = pd.DataFrame(columns=keys + ["ukbb_p_min", "ukbb_best_embed_type",
                                            "num_feature"])
    else:
        best = (long.sort_values("ukbb_p").groupby(keys, as_index=False).first()
                .rename(columns={"ukbb_p": "ukbb_p_min",
                                 "embed_type": "ukbb_best_embed_type"}))
    out = base.merge(best, on=keys, how="left")
    out["status"] = out["ukbb_p_min"].notna().map(
        {True: STATUS_TESTED, False: STATUS_NOT_TESTED})
    targeted = TARGETED_ALPHA / len(targets)
    out["replicated_genomewide"] = out["ukbb_p_min"] < SIGNIFICANCE_THRESHOLD
    out["replicated_targeted"] = out["ukbb_p_min"] < targeted
    if counts is not None:
        # count_gene_variants.py has its own status column (sequenced or why
        # not); keep it under a name that cannot collide with ours.
        # It also repeats the target columns (trait, sex, HPP p); take only
        # what it adds, or the merge doubles them into _x/_y pairs.
        bare = (counts.rename(columns={"status": "variant_status"})
                .assign(_bare=counts["gene_id"].map(strip_version)))
        bare = bare[["_bare"] + [c for c in VARIANT_COLUMNS if c in bare.columns]]
        bare = bare.drop_duplicates("_bare")
        out = (out.assign(_bare=out["gene_id"].map(strip_version))
               .merge(bare, on="_bare", how="left").drop(columns="_bare"))
    return out


def replication_rates(summary: pd.DataFrame) -> dict:
    tested = summary["status"] == STATUS_TESTED
    rates = {"n_targets": len(summary), "n_tested": int(tested.sum())}
    for flag in ("genomewide", "targeted"):
        n = int(summary[f"replicated_{flag}"].sum())
        rates[f"n_replicated_{flag}"] = n
        rates[f"rate_{flag}_of_targets"] = n / len(summary) if len(summary) else float("nan")
        rates[f"rate_{flag}_of_tested"] = n / tested.sum() if tested.any() else float("nan")
    rates["targeted_threshold"] = TARGETED_ALPHA / len(summary) if len(summary) else float("nan")
    rates["genomewide_threshold"] = SIGNIFICANCE_THRESHOLD
    return rates


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--targets", default=config.TARGET_GENES_TSV)
    ap.add_argument("--assoc_dir", default=config.ASSOC_DIR)
    ap.add_argument("--variants", help="count_gene_variants.py output tsv")
    ap.add_argument("--out", default=os.path.join(config.ASSOC_DIR, "ukbb_replication"))
    args = ap.parse_args()

    targets = pd.read_csv(args.targets, sep="\t")
    counts = pd.read_csv(args.variants, sep="\t") if args.variants else None
    long = per_embedding_p(targets, args.assoc_dir)
    summary = per_gene_summary(targets, long, counts)
    rates = replication_rates(summary)

    long.to_csv(f"{args.out}.per_embedding.tsv", sep="\t", index=False)
    summary.to_csv(f"{args.out}.tsv", sep="\t", index=False)
    pd.DataFrame([rates]).to_csv(f"{args.out}.rates.tsv", sep="\t", index=False)
    print(summary.to_string(index=False, float_format=lambda v: f"{v:.3g}"))
    print()
    for key, value in rates.items():
        print(f"{key:<32}{value}")


if __name__ == "__main__":
    main()

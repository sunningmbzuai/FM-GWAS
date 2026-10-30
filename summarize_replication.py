"""UKBB replication of Ning's HPP target list: the three tables she asked for.

For each (gene, trait, sex) in config.TARGET_GENES_TSV it reads that cell's
fmgwas/insample_hpp_AG.py output under config.ASSOC_DIR and writes

    <out>.per_embedding.tsv  UKBB p per embedding type (long form), with the
                             test it came from (LRT for disease traits, F for
                             quantitative ones) and the script's status
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

Power: each row carries the sample and case counts the association script
recorded for the best embedding type, n_eff (4 / (1/cases + 1/controls) for
disease traits, n_samples for quantitative ones), and min_detectable_r2, the
smallest variance explained that a chi-square test on gene_pca_components df
detects with 80% power at the targeted threshold (ncp = n_eff * r2 / (1 - r2)).
For the logistic LRT that is a rough guide, not an exact power calculation;
use it to tell an underpowered cell from a true non-replication.

Before anything else it checks that the target list is the current HPP export
(build_target_genes.SOURCE_LISTS, novel_genes_fdr_1026.tsv): the file exists and
holds exactly the target list's (gene, trait, sex) rows. A mismatch is printed
as a warning, since results for a stale list still mean something.

Variant counts come from count_gene_variants.py's output (pass --variants).

    python summarize_replication.py --variants ukbb_variants_per_gene.tsv \
        --out ukbb_replication
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd
from scipy import optimize, stats

import assoc_settings
import build_target_genes
import config
from compute_recovery import PER_GENE_DIR, SIGNIFICANCE_THRESHOLD, p_values, strip_version

TARGETED_ALPHA = 0.05
TARGET_POWER = 0.8
STATUS_TESTED = "tested"
STATUS_NOT_TESTED = "no_ukbb_result"
VARIANT_COLUMNS = ["chrom", "start", "end", "length_bp", "n_variants",
                   "n_polymorphic", "variant_status"]


def find_result(assoc_dir: str, trait: str, sex: str, gene_id: str) -> str | None:
    """The cell's result tsv for this gene, whatever version suffix it carries."""
    cell = os.path.join(assoc_dir, f"{trait}_{sex}", PER_GENE_DIR)
    hits = glob.glob(os.path.join(cell, f"{strip_version(gene_id)}.*tsv"))
    hits += glob.glob(os.path.join(cell, f"{strip_version(gene_id)}.tsv"))
    return sorted(set(hits))[0] if hits else None


LONG_COLUMNS = ["gene_id", "trait", "sex", "embed_type", "gene_pca_components",
                "test", "ukbb_p", "status", "n_samples", "n_cases"]


def per_embedding_p(targets: pd.DataFrame, assoc_dir: str) -> pd.DataFrame:
    """Long table: one row per target and embedding type.

    status is the script's own verdict for the test used (lrt_status for the
    LRT, status for F): ok, or e.g. full_separation / not_converged, whose p
    the script still reports. Nothing is filtered on it here.
    """
    frames = []
    for row in targets.itertuples(index=False):
        path = find_result(assoc_dir, row.trait, row.sex, row.gene_id)
        if path is None:
            continue
        result = pd.read_csv(path, sep="\t")
        if result.empty:
            continue
        test = assoc_settings.stat_test(row.trait)
        status_column = "lrt_status" if test == "LRT" else "status"
        frames.append(pd.DataFrame({
            "gene_id": row.gene_id,
            "trait": row.trait,
            "sex": row.sex,
            "embed_type": result["embed_type"],
            "gene_pca_components": result.get("gene_pca_components"),
            "test": test,
            "ukbb_p": p_values(result, assoc_settings.p_column(row.trait), path),
            "status": result.get(status_column, result.get("status")),
            # Older result files predate these columns.
            "n_samples": result.get("n_samples"),
            "n_cases": result.get("n_cases"),
        }))
    return (pd.concat(frames, ignore_index=True)[LONG_COLUMNS] if frames
            else pd.DataFrame(columns=LONG_COLUMNS))


def per_gene_summary(targets: pd.DataFrame, long: pd.DataFrame,
                     counts: pd.DataFrame | None) -> pd.DataFrame:
    keys = ["gene_id", "trait", "sex"]
    base = targets[keys + ["p_F_analytic_min"]].rename(
        columns={"p_F_analytic_min": "hpp_p_min"})
    if long.empty:
        best = pd.DataFrame(columns=keys + ["ukbb_p_min", "ukbb_best_embed_type",
                                            "gene_pca_components", "test",
                                            "ukbb_test_status", "n_samples",
                                            "n_cases"])
    else:
        # The whole best row, not groupby.first(), which takes each column's
        # first non-null value and can mix rows.
        best = (long.dropna(subset=["ukbb_p"]).sort_values("ukbb_p", kind="stable")
                .drop_duplicates(keys)
                .rename(columns={"ukbb_p": "ukbb_p_min",
                                 "embed_type": "ukbb_best_embed_type",
                                 "status": "ukbb_test_status"}))
    out = base.merge(best, on=keys, how="left")
    out["status"] = out["ukbb_p_min"].notna().map(
        {True: STATUS_TESTED, False: STATUS_NOT_TESTED})
    targeted = TARGETED_ALPHA / len(targets)
    out["replicated_genomewide"] = out["ukbb_p_min"] < SIGNIFICANCE_THRESHOLD
    out["replicated_targeted"] = out["ukbb_p_min"] < targeted
    out = add_power(out, targeted)
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


def check_target_source(targets: pd.DataFrame,
                        sources: list[str] = build_target_genes.SOURCE_LISTS
                        ) -> tuple[bool, str]:
    """Is the target list exactly the rows of the current HPP export(s)?"""
    names = ", ".join(os.path.basename(p) for p in sources)
    missing = [p for p in sources if not os.path.exists(p)]
    if missing:
        return False, f"target source export missing: {', '.join(missing)}"

    def keys(rows) -> set[tuple[str, str, str]]:
        return {(strip_version(g), t, s) for g, t, s in rows}

    exported = keys((r["gene_id"], r["trait"], r["sex"])
                    for r in build_target_genes.union_sources(sources))
    listed = keys(targets[["gene_id", "trait", "sex"]].itertuples(index=False))
    if exported == listed:
        return True, f"target list matches {names} ({len(exported)} rows)"
    return False, (f"target list does not match {names}: "
                   f"{len(exported - listed)} in the export but not the target list, "
                   f"{len(listed - exported)} in the target list but not the export "
                   f"-- rerun build_target_genes.py")


def effective_n(n_samples: pd.Series, n_cases: pd.Series) -> pd.Series:
    """4 / (1/cases + 1/controls) for disease traits, n_samples otherwise."""
    n_controls = n_samples - n_cases
    balanced = 4 / (1 / n_cases + 1 / n_controls)
    return balanced.where(n_cases.notna(), n_samples)


def min_detectable_r2(n_eff: float, df: float, alpha: float,
                      power: float = TARGET_POWER) -> float:
    """Smallest r2 a df-degree chi-square test detects at alpha with this power."""
    if not (np.isfinite(n_eff) and np.isfinite(df)) or n_eff <= 0 or df < 1:
        return float("nan")
    critical = stats.chi2.isf(alpha, df)
    ncp = optimize.brentq(
        lambda lam: stats.ncx2.sf(critical, df, lam) - power, 1e-9, 1e6)
    return ncp / (ncp + n_eff)


def add_power(summary: pd.DataFrame, alpha: float) -> pd.DataFrame:
    n_samples = pd.to_numeric(summary["n_samples"], errors="coerce")
    n_cases = pd.to_numeric(summary["n_cases"], errors="coerce")
    n_eff = effective_n(n_samples, n_cases)
    df = pd.to_numeric(summary["gene_pca_components"], errors="coerce")
    return summary.assign(
        n_samples=n_samples,
        n_cases=n_cases,
        n_controls=n_samples - n_cases,
        n_eff=n_eff,
        min_detectable_r2=[min_detectable_r2(n, k, alpha)
                           for n, k in zip(n_eff, df)],
    )


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
    ok, message = check_target_source(targets)
    print(f"[{'ok' if ok else 'warn'}] {message}")
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

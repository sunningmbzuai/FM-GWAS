"""Recovery rate: what fraction of HPP-significant genes replicate in UKB.

This is the validation the FM-GWAS repo defines (README, "Significance
Filtering & Replication Validation"): run the identical pipeline on an
independent cohort, filter to p_F_analytic < 2.5e-6, deduplicate by gene_id,
and report the proportion of HPP-significant genes that come out significant
here too. MAGMA, GSEA and Open Targets are paper-only analyses with no code in
the repo, and are deliberately out of scope.

The HPP-significant gene lists are HPP_union_gene/<trait>_extreme[_F|_M].tsv --
they are already thresholded, so they are the denominator, not a candidate set
to re-filter.

Per gene we take the MINIMUM p_F_analytic across embedding types, matching the
paper's union rule ("for each gene, the minimum p-value across embedding
types"). Reads only the association TSVs, so it runs in any env with pandas.
"""
import os

import pandas as pd

import config
import target_genes

SIGNIFICANCE_THRESHOLD = 2.5e-6   # README + paper Methods, Bonferroni
GENE_ID_COLUMN = "gene_id"
P_COLUMN = "p_F_analytic"


def load_hpp_genes(trait: str, sex_name: str) -> set[str]:
    """The HPP-significant genes for one trait/sex -- the recovery denominator."""
    suffix = config.SEX_SUFFIXES[sex_name]
    path = os.path.join(config.HPP_UNION_GENE_DIR, f"{trait}_extreme{suffix}.tsv")
    if not os.path.exists(path):
        return set()
    frame = pd.read_csv(path, sep="\t")
    if GENE_ID_COLUMN not in frame.columns:
        raise KeyError(f"{path} has no {GENE_ID_COLUMN} column: {list(frame.columns)}")
    genes = set(strip_version(g) for g in frame[GENE_ID_COLUMN].dropna())
    if config.RESTRICT_TO_TARGET_GENES:
        # Key-association mode: the denominator is this cell's target genes,
        # not the full HPP list, or every untested gene would read as a
        # replication failure.
        genes &= target_genes.target_gene_ids_for(trait, sex_name)
    return genes


def strip_version(gene_id: str) -> str:
    """ENSG00000141510.16 -> ENSG00000141510.

    The GTF carries versioned ids and the gene lists may not, so both sides are
    compared unversioned; a version mismatch would otherwise read as a total
    replication failure.
    """
    return str(gene_id).split(".")[0]


def union_p_values(result_dir: str) -> pd.Series:
    """gene_id -> minimum p_F_analytic across embedding types, for one trait/sex.

    insample_hpp_AG.py writes one TSV per gene with a row per embedding type.
    Genes whose TSV is missing simply do not appear -- an unrun gene is not a
    non-replicating one, and the caller reports the two separately.
    """
    if not os.path.isdir(result_dir):
        return pd.Series(dtype="float64")

    best: dict[str, float] = {}
    for fname in sorted(os.listdir(result_dir)):
        if not fname.endswith(".tsv"):
            continue
        frame = pd.read_csv(os.path.join(result_dir, fname), sep="\t")
        if P_COLUMN not in frame.columns:
            raise KeyError(
                f"{os.path.join(result_dir, fname)} has no {P_COLUMN} column: "
                f"{list(frame.columns)}"
            )
        gene_id = strip_version(
            frame[GENE_ID_COLUMN].iloc[0] if GENE_ID_COLUMN in frame.columns
            else os.path.splitext(fname)[0]
        )
        p_min = pd.to_numeric(frame[P_COLUMN], errors="coerce").min()
        if pd.notna(p_min):
            best[gene_id] = min(best.get(gene_id, 1.0), float(p_min))
    return pd.Series(best, dtype="float64").sort_index()


def significant_genes(p_values: pd.Series,
                      threshold: float = SIGNIFICANCE_THRESHOLD) -> set[str]:
    return set(p_values.index[p_values < threshold])


def recovery(hpp_genes: set[str], ukb_p: pd.Series,
             threshold: float = SIGNIFICANCE_THRESHOLD) -> dict:
    """Recovery of one trait/sex, separating non-replication from non-coverage."""
    ukb_significant = significant_genes(ukb_p, threshold)
    tested = hpp_genes & set(ukb_p.index)
    recovered = hpp_genes & ukb_significant
    return {
        "hpp_significant": len(hpp_genes),
        "tested_in_ukb": len(tested),
        "not_tested_in_ukb": len(hpp_genes - set(ukb_p.index)),
        "recovered": len(recovered),
        "recovery_rate": len(recovered) / len(hpp_genes) if hpp_genes else float("nan"),
        "recovery_rate_of_tested": (
            len(recovered) / len(tested) if tested else float("nan")
        ),
        "ukb_significant_total": len(ukb_significant),
        "ukb_significant_novel": len(ukb_significant - hpp_genes),
    }


def main() -> None:
    rows, all_hpp, all_recovered, all_ukb_sig = [], set(), set(), set()

    for trait in config.TRAITS:
        for sex_name in config.SEX_SUFFIXES:
            hpp_genes = load_hpp_genes(trait, sex_name)
            if not hpp_genes:
                print(f"[skip] {trait} {sex_name}: no HPP gene list")
                continue

            result_dir = os.path.join(config.ASSOC_DIR, f"{trait}_{sex_name}")
            ukb_p = union_p_values(result_dir)
            if ukb_p.empty:
                print(f"[skip] {trait} {sex_name}: no association results in "
                      f"{result_dir}")
                continue

            stats = recovery(hpp_genes, ukb_p)
            rows.append({"trait": trait, "sex": sex_name, **stats})
            print(f"{trait} {sex_name}: {stats['recovered']}/{stats['hpp_significant']} "
                  f"recovered ({stats['recovery_rate']:.1%}), "
                  f"{stats['not_tested_in_ukb']} not tested, "
                  f"{stats['ukb_significant_novel']} UKB-only")

            all_hpp |= hpp_genes
            all_recovered |= hpp_genes & significant_genes(ukb_p)
            all_ukb_sig |= significant_genes(ukb_p)

    if not rows:
        raise SystemExit(
            f"No association results found under {config.ASSOC_DIR} -- run the "
            f"assoc stage of run_pipeline.sh first."
        )

    os.makedirs(config.ASSOC_DIR, exist_ok=True)
    summary = pd.DataFrame(rows)
    out_path = os.path.join(config.ASSOC_DIR, "recovery_summary.tsv")
    summary.to_csv(out_path, sep="\t", index=False)

    genes_path = os.path.join(config.ASSOC_DIR, "ukb_significant_genes.tsv")
    pd.DataFrame({
        "gene_id": sorted(all_ukb_sig),
        "in_hpp_union": [g in all_hpp for g in sorted(all_ukb_sig)],
    }).to_csv(genes_path, sep="\t", index=False)

    overall = len(all_recovered) / len(all_hpp) if all_hpp else float("nan")
    print()
    print(f"HPP union (deduplicated by gene_id): {len(all_hpp):,} genes")
    print(f"Recovered in UKB at p < {SIGNIFICANCE_THRESHOLD:g}: "
          f"{len(all_recovered):,} ({overall:.1%})")
    print(f"UKB-significant overall: {len(all_ukb_sig):,} "
          f"({len(all_ukb_sig - all_hpp):,} not in the HPP union)")
    print(f"Wrote {out_path}")
    print(f"Wrote {genes_path}")


if __name__ == "__main__":
    main()

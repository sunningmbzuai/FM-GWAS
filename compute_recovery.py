"""Recovery rate: what fraction of HPP-significant genes replicate in UKB.

This is the validation the FM-GWAS repo defines (README, "Significance
Filtering & Replication Validation"): run the identical pipeline on an
independent cohort, filter to p < 2.5e-6, deduplicate by gene_id,
and report the proportion of HPP-significant genes that come out significant
here too. MAGMA, GSEA and Open Targets are paper-only analyses with no code in
the repo, and are deliberately out of scope.

The HPP-significant gene lists are HPP_union_gene/<trait>_extreme[_F|_M].tsv --
they are already thresholded, so they are the denominator, not a candidate set
to re-filter.

Per gene we take the MINIMUM p across embedding types, matching the paper's
union rule ("for each gene, the minimum p-value across embedding types"). The p
column is the cell's test (assoc_settings.p_column): p_F for quantitative
traits, p_LRT for disease traits. Reads only the association TSVs, so it runs
in any env with pandas.
"""
import os

import pandas as pd

import assoc_settings
import config
import target_genes

SIGNIFICANCE_THRESHOLD = 2.5e-6   # README + paper Methods, Bonferroni
GENE_ID_COLUMN = "gene_id"
PER_GENE_DIR = "per_gene"   # insample_hpp_AG.gene_result_path


def hpp_list_path(trait: str, sex_name: str) -> str:
    suffix = config.SEX_SUFFIXES[sex_name]
    return os.path.join(config.HPP_UNION_GENE_DIR, f"{trait}_extreme{suffix}.tsv")


def load_hpp_genes(trait: str, sex_name: str) -> set[str]:
    """The HPP-significant genes for one trait/sex -- the recovery denominator."""
    path = hpp_list_path(trait, sex_name)
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


def skip_reason(trait: str, sex_name: str) -> str:
    """Why load_hpp_genes came back empty: no list, or no targets in it."""
    path = hpp_list_path(trait, sex_name)
    if not os.path.exists(path):
        return f"no HPP gene list at {path}"
    n_genes = pd.read_csv(path, sep="\t")[GENE_ID_COLUMN].nunique()
    return (f"HPP gene list has {n_genes} genes, none of them among this "
            f"cell's target genes ({os.path.basename(path)})")


def strip_version(gene_id: str) -> str:
    """ENSG00000141510.16 -> ENSG00000141510.

    The GTF carries versioned ids and the gene lists may not, so both sides are
    compared unversioned; a version mismatch would otherwise read as a total
    replication failure.
    """
    return str(gene_id).split(".")[0]


def gene_results(result_dir: str) -> dict[str, pd.DataFrame]:
    """Unversioned gene_id -> that gene's result rows, for one trait/sex cell.

    fmgwas/insample_hpp_AG.py writes <cell>/per_gene/<gene>.tsv with a row per
    embedding type. Genes whose TSV is missing simply do not appear -- an unrun
    gene is not a non-replicating one, and the caller reports the two
    separately.
    """
    per_gene = os.path.join(result_dir, PER_GENE_DIR)
    if not os.path.isdir(per_gene):
        return {}
    results = {}
    for fname in sorted(os.listdir(per_gene)):
        if not fname.endswith(".tsv"):
            continue
        frame = pd.read_csv(os.path.join(per_gene, fname), sep="\t")
        gene_id = strip_version(
            frame[GENE_ID_COLUMN].iloc[0] if GENE_ID_COLUMN in frame.columns
            and not frame.empty else os.path.splitext(fname)[0]
        )
        results[gene_id] = frame
    return results


def p_values(frame: pd.DataFrame, p_column: str, where: str = "") -> pd.Series:
    """The p column of one gene's rows; all-NaN when every row failed input.

    A gene whose feature file could not be read gets input_failed rows with no
    statistics at all, so the column is absent; that is "not tested", not an
    error. Any other result without the column is.
    """
    if p_column in frame.columns:
        return pd.to_numeric(frame[p_column], errors="coerce")
    if "status" in frame.columns and (frame["status"] == "input_failed").all():
        return pd.Series(float("nan"), index=frame.index)
    raise KeyError(f"{where} has no {p_column} column: {list(frame.columns)}")


def union_p_values(result_dir: str, p_column: str) -> pd.Series:
    """gene_id -> minimum p across embedding types, for one trait/sex."""
    best: dict[str, float] = {}
    for gene_id, frame in gene_results(result_dir).items():
        p_min = p_values(frame, p_column, f"{result_dir} {gene_id}").min()
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
                print(f"[skip] {trait} {sex_name}: {skip_reason(trait, sex_name)}")
                continue

            result_dir = os.path.join(config.ASSOC_DIR, f"{trait}_{sex_name}")
            ukb_p = union_p_values(result_dir, assoc_settings.p_column(trait))
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

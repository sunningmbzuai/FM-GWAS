"""Restrict the whole replication run to a short list of key associations.

The default denominator is the entire HPP union (thousands of genes), which
costs a full embedding pass over every gene. When only the headline novel hits
matter, TARGET_GENES_TSV names them and every stage -- gene sequences,
embeddings, association, recovery -- narrows to that list.

The TSV is the top_novel_genes.tsv produced upstream: one row per
(gene_id, trait, sex) with the HPP p-value range. gene_id is versioned there
and versions need not match the GRCh37 GTF, so matching is always unversioned.

A target the HPP gene files do not list still runs. FM-GWAS reads only
gene_id from those files (insample_hpp_AG.py, get_hpp_embedding-AG_clean.py),
so the row is made from the target list with the HPP-only columns empty.
Ning's full-cohort exports (novel_genes_1006.tsv) are mostly such genes: the
HPP_union_gene files on disk come from the July run.

Sex semantics: a row is a sex-specific hit (M or F). The "All" cell of a trait
gets the union of that trait's rows across sexes -- the pooled cohort is a
legitimate test of a sex-specific hit, and dropping it would leave the All
cells empty.
"""
import os

import pandas as pd

import config

GENE_ID_COLUMN = "gene_id"
TRAIT_COLUMN = "trait"
SEX_COLUMN = "sex"
CANDIDATE_GENES_FILE = "candidate_genes.tsv"


def strip_version(gene_id: str) -> str:
    """ENSG00000141510.16 -> ENSG00000141510."""
    return str(gene_id).split(".")[0]


def load_targets(path: str | None = None) -> pd.DataFrame:
    """The target table, with an added unversioned gene id column."""
    path = path or config.TARGET_GENES_TSV
    frame = pd.read_csv(path, sep="\t")
    missing = {GENE_ID_COLUMN, TRAIT_COLUMN, SEX_COLUMN} - set(frame.columns)
    if missing:
        raise KeyError(f"{path} is missing column(s) {sorted(missing)}: "
                       f"{list(frame.columns)}")
    return frame.assign(bare_gene_id=frame[GENE_ID_COLUMN].map(strip_version))


def target_versioned_ids(path: str | None = None) -> dict[str, str]:
    """Unversioned -> versioned target id, first occurrence in file order."""
    frame = load_targets(path)
    return dict(zip(frame["bare_gene_id"][::-1], frame[GENE_ID_COLUMN][::-1]))


def extend_with_targets(gene_ids: list[str], path: str | None = None
                        ) -> tuple[list[str], list[str]]:
    """Union genes that are targets, plus targets the union does not list.

    Union ids keep their own version; an absent target uses the target list's
    id. Returns (kept, added), added in target-list order.
    """
    wanted = target_versioned_ids(path)
    kept = [g for g in gene_ids if strip_version(g) in wanted]
    present = {strip_version(g) for g in kept}
    added = [versioned for bare, versioned in ordered_targets(wanted)
             if bare not in present]
    return kept + added, added


def ordered_targets(wanted: dict[str, str]) -> list[tuple[str, str]]:
    """(bare, versioned) pairs sorted by versioned id, for a stable order."""
    return sorted(wanted.items(), key=lambda item: item[1])


def write_candidate_genes(hpp_path: str, out_path: str,
                          targets_path: str | None = None) -> int:
    """HPP candidate_genes.tsv plus every target it lacks; returns n added.

    The embedding script reads its gene list from this file, so a target
    missing from it would never be embedded.
    """
    frame = read_gene_file(hpp_path)
    present = set(frame[GENE_ID_COLUMN].map(strip_version))
    extra = [versioned for bare, versioned
             in ordered_targets(target_versioned_ids(targets_path))
             if bare not in present]
    rows = pd.DataFrame({GENE_ID_COLUMN: extra}).reindex(columns=frame.columns)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    pd.concat([frame, rows], ignore_index=True).to_csv(
        out_path, sep="\t", index=False)
    return len(extra)


def target_gene_ids(path: str | None = None) -> set[str]:
    """Every target gene, unversioned -- the gene-sequence/embedding scope."""
    return set(load_targets(path)["bare_gene_id"])


def target_gene_ids_for(trait: str, sex_name: str,
                        path: str | None = None) -> set[str]:
    """Target genes for one trait/sex cell, unversioned. "All" = both sexes."""
    frame = load_targets(path)
    rows = frame[frame[TRAIT_COLUMN] == trait]
    if sex_name != "All":
        rows = rows[rows[SEX_COLUMN] == sex_name]
    return set(rows["bare_gene_id"])


def filter_gene_file(hpp_path: str, out_path: str, trait: str,
                     sex_name: str, targets_path: str | None = None) -> int:
    """Write the HPP gene file keeping only this cell's target genes.

    The HPP file's own columns are preserved verbatim -- FM-GWAS reads more of
    them than gene_id, so this subsets rows and never rebuilds the table.
    Returns the number of rows kept; 0 means nothing to run for this cell.
    """
    frame = read_gene_file(hpp_path)
    wanted = target_gene_ids_for(trait, sex_name, targets_path)
    kept = frame[frame[GENE_ID_COLUMN].map(strip_version).isin(wanted)]
    if sex_name == "All":
        kept = add_sex_specific_rows(kept, hpp_path, trait, wanted)
    kept = add_target_only_rows(kept, wanted, targets_path)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    kept.to_csv(out_path, sep="\t", index=False)
    return len(kept)


def read_gene_file(path: str) -> pd.DataFrame:
    """One HPP gene file, with its own columns preserved verbatim."""
    frame = pd.read_csv(path, sep="\t")
    if GENE_ID_COLUMN not in frame.columns:
        raise KeyError(f"{path} has no {GENE_ID_COLUMN} column: "
                       f"{list(frame.columns)}")
    return frame


def add_sex_specific_rows(kept: pd.DataFrame, hpp_path: str, trait: str,
                          wanted: set[str]) -> pd.DataFrame:
    """Rows for target genes the pooled HPP file does not list.

    <trait>_extreme.tsv holds what HPP called significant with both sexes
    pooled, so a hit found in one sex only is absent from it. Intersecting with
    that file alone would drop the gene and never run the pooled UKB test for
    a hit HPP found in one sex only. The gene's row is taken
    from whichever sex-specific file carries it; the columns FM-GWAS reads are
    gene coordinates, which do not differ between a trait's files.
    """
    directory = os.path.dirname(hpp_path)
    missing = wanted - set(kept[GENE_ID_COLUMN].map(strip_version))
    extra = []
    for suffix in ("_F", "_M"):
        if not missing:
            break
        path = os.path.join(directory, f"{trait}_extreme{suffix}.tsv")
        if not os.path.exists(path):
            continue
        frame = read_gene_file(path)
        rows = frame[frame[GENE_ID_COLUMN].map(strip_version).isin(missing)]
        if rows.empty:
            continue
        extra.append(rows.reindex(columns=kept.columns))
        missing -= set(rows[GENE_ID_COLUMN].map(strip_version))
    if not extra:
        return kept
    return pd.concat([kept, *extra], ignore_index=True)


def add_target_only_rows(kept: pd.DataFrame, wanted: set[str],
                         targets_path: str | None = None) -> pd.DataFrame:
    """gene_id-only rows for this cell's targets that no HPP file listed."""
    missing = wanted - set(kept[GENE_ID_COLUMN].map(strip_version))
    if not missing:
        return kept
    versioned = target_versioned_ids(targets_path)
    rows = pd.DataFrame({GENE_ID_COLUMN: sorted(versioned[b] for b in missing)})
    return pd.concat([kept, rows.reindex(columns=kept.columns)],
                     ignore_index=True)


def main() -> None:
    """Materialise one filtered gene file per trait/sex under TARGET_GENE_DIR."""
    if not config.RESTRICT_TO_TARGET_GENES:
        raise SystemExit("config.RESTRICT_TO_TARGET_GENES is False -- nothing to do.")
    if not os.path.exists(config.TARGET_GENES_TSV):
        raise SystemExit(f"{config.TARGET_GENES_TSV} not found.")

    targets = load_targets()
    print(f"Target list: {targets['bare_gene_id'].nunique()} genes, "
          f"{len(targets)} (gene, trait, sex) associations")

    written = 0
    for trait in config.TRAITS:
        for sex_name, suffix in config.SEX_SUFFIXES.items():
            hpp_path = os.path.join(config.HPP_UNION_GENE_DIR,
                                    f"{trait}_extreme{suffix}.tsv")
            if not os.path.exists(hpp_path):
                continue
            out_path = os.path.join(config.TARGET_GENE_DIR,
                                    f"{trait}_extreme{suffix}.tsv")
            n_kept = filter_gene_file(hpp_path, out_path, trait, sex_name)
            print(f"  {trait} {sex_name}: {n_kept} target gene(s)")
            written += 1

    if not written:
        raise SystemExit(f"No HPP gene files found under {config.HPP_UNION_GENE_DIR}.")

    n_added = write_candidate_genes(
        os.path.join(config.HPP_UNION_GENE_DIR, CANDIDATE_GENES_FILE),
        os.path.join(config.TARGET_GENE_DIR, CANDIDATE_GENES_FILE))
    print(f"  {CANDIDATE_GENES_FILE}: added {n_added} target gene(s) "
          f"absent from the HPP list")
    print(f"Wrote {written} filtered gene files to {config.TARGET_GENE_DIR}")


if __name__ == "__main__":
    main()

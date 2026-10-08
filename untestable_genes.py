"""Which target genes UKB can actually test, and why the rest cannot.

Two ways a target is untestable with UKB array genotypes:

    no_sequence       build_gene_sequences.py wrote no TSV: not protein-coding
                      on chr1-22 in the GRCh37 GTF, over 128 kb, or no array
                      variant inside the gene (its log names which).
    single_haplotype  every participant carries the same haplotype, so the
                      dedup index holds one distinct sequence and the features
                      are constant across people.

Two distinct sequences is already testable: carriers of the second haplotype
differ from everyone else. Note the dedup log's distinct_rows counts PACKED
rows (two sequences each) in seqs mode; distinct_seqs is the number to read.

    python untestable_genes.py                  # current target list
    python untestable_genes.py --out status.tsv # also write the full table

Run after run_embeddings.sh has deduplicated the genes; a gene with a sequence
but no index yet is reported as not_deduplicated rather than guessed.
"""
import argparse
import json
import os

import pandas as pd

import config
import target_genes

NO_SEQUENCE = "no_sequence"
SINGLE_HAPLOTYPE = "single_haplotype"
NOT_DEDUPLICATED = "not_deduplicated"
TESTABLE = "testable"
UNTESTABLE = (NO_SEQUENCE, SINGLE_HAPLOTYPE)


def files_by_bare_id(directory: str, suffix: str) -> dict[str, str]:
    """Unversioned gene id -> path, for every <gene_id><suffix> in the dir."""
    if not os.path.isdir(directory):
        return {}
    return {target_genes.strip_version(name[:-len(suffix)]):
            os.path.join(directory, name)
            for name in os.listdir(directory) if name.endswith(suffix)}


def distinct_sequences(index_path: str) -> int | None:
    """n_distinct from a dedup index, or None if it is missing or unreadable."""
    try:
        with open(index_path) as handle:
            return int(json.load(handle)["n_distinct"])
    except (OSError, ValueError, KeyError):
        return None


def gene_status(has_sequence: bool, n_distinct: int | None) -> str:
    if not has_sequence:
        return NO_SEQUENCE
    if n_distinct is None:
        return NOT_DEDUPLICATED
    return SINGLE_HAPLOTYPE if n_distinct <= 1 else TESTABLE


def classify(targets_path: str, gene_seq_dir: str,
             unique_dir: str) -> pd.DataFrame:
    """One row per target association with its status and haplotype count."""
    targets = target_genes.load_targets(targets_path)
    sequences = files_by_bare_id(gene_seq_dir, ".tsv")
    indexes = files_by_bare_id(unique_dir, ".index.json")
    n_distinct = [distinct_sequences(indexes[bare]) if bare in indexes else None
                  for bare in targets["bare_gene_id"]]
    status = [gene_status(bare in sequences, n)
              for bare, n in zip(targets["bare_gene_id"], n_distinct)]
    return pd.DataFrame({
        "gene_id": targets[target_genes.GENE_ID_COLUMN],
        "trait": targets[target_genes.TRAIT_COLUMN],
        "sex": targets[target_genes.SEX_COLUMN],
        "status": status,
        "n_distinct_seqs": pd.array(n_distinct, dtype="Int64"),
    })


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targets", default=config.TARGET_GENES_TSV)
    parser.add_argument("--gene_seq_dir", help="default: config.GENE_SEQ_DIR")
    parser.add_argument("--unique_dir", help="default: config.UNIQUE_GENE_SEQ_DIR")
    parser.add_argument("--out", help="also write the full table here (TSV)")
    args = parser.parse_args(argv)
    # Resolved only when not given, so explicit paths need no UKBV_* settings.
    args.gene_seq_dir = args.gene_seq_dir or config.GENE_SEQ_DIR
    args.unique_dir = args.unique_dir or config.UNIQUE_GENE_SEQ_DIR
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    frame = classify(args.targets, args.gene_seq_dir, args.unique_dir)
    if args.out:
        frame.to_csv(args.out, sep="\t", index=False)

    print(frame.groupby(["trait", "sex", "status"]).size()
          .rename("n_genes").to_string())
    untestable = frame[frame["status"].isin(UNTESTABLE)]
    print(f"\n{len(untestable)} untestable target gene(s):")
    if not untestable.empty:
        print(untestable.to_string(index=False))
    pending = int((frame["status"] == NOT_DEDUPLICATED).sum())
    if pending:
        print(f"\n{pending} gene(s) not deduplicated yet -- rerun after "
              f"run_embeddings.sh")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

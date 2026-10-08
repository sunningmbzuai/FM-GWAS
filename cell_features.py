"""Which genes of an assoc cell still have no features on disk.

run_assoc.sh calls this after a cell succeeds, before touching its .done
marker. insample_hpp_AG.py silently skips a gene with no feature table, so a
cell run before the GPU pass finishes "succeeds" on a subset; marking it done
would make every later run skip it and the late genes would never be tested.
Leaving the marker off is cheap: the rerun skips genes whose .tsv exists.

Features live one directory per gene (feature_tables/<gene>/<gene>.npz, or
embeddings/<gene>/<gene>.pt as the fallback root). Matching is unversioned,
since the gene files and feature directories need not agree on versions.

    python cell_features.py <gene_file> <feature_root>
    exit 0 = every gene has features, 1 = some do not (listed on stdout)

Stdlib only: it runs under whichever interpreter run_assoc.sh picked.
"""
import csv
import os
import sys

GENE_ID_COLUMN = "gene_id"


def strip_version(gene_id: str) -> str:
    """ENSG00000141510.16 -> ENSG00000141510."""
    return str(gene_id).split(".")[0]


def read_gene_ids(gene_file: str) -> list[str]:
    """gene_id column of a tab-separated HPP/target gene file, in file order."""
    with open(gene_file, newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if GENE_ID_COLUMN not in (reader.fieldnames or []):
            raise KeyError(f"{gene_file} has no {GENE_ID_COLUMN} column: "
                           f"{reader.fieldnames}")
        return [row[GENE_ID_COLUMN] for row in reader]


def genes_with_features(feature_root: str) -> set[str]:
    """Unversioned ids of every non-empty gene directory under the root."""
    if not os.path.isdir(feature_root):
        return set()
    return {strip_version(entry.name) for entry in os.scandir(feature_root)
            if entry.is_dir() and any(os.scandir(entry.path))}


def genes_without_features(gene_file: str, feature_root: str) -> list[str]:
    """Genes in the file with no feature directory, in file order."""
    present = genes_with_features(feature_root)
    return [gene for gene in read_gene_ids(gene_file)
            if strip_version(gene) not in present]


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: cell_features.py <gene_file> <feature_root>",
              file=sys.stderr)
        return 2
    missing = genes_without_features(*argv)
    for gene in missing:
        print(gene)
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

"""Which genes of an assoc cell still need fmgwas/insample_hpp_AG.py.

The script writes <out_dir>/per_gene/<quoted gene_id>.tsv per gene and skips a
gene whose file exists, so a rerun only does what is left. run_assoc.sh uses
the pending count to decide how many ranks (--rank/--world_size, the script's
own split) a cell gets.

One case needs help. A gene whose features cannot be read is not skipped
silently, as the July script did: the script writes it a result file of
input_failed rows, and that file would make every later run skip the gene.
The usual causes are no feature file yet ("expected exactly one .pt/.npz
feature file") and a table in the old format that export_features.py has not
rewritten yet. clear_stale removes an input_failed file once the gene's
feature file exists and either the failure was "no feature file" or the
feature file is newer than the failure, so the gene is retried on the new
features. A failure on the current features is the script's verdict and is
left alone.

    python assoc_pending.py count <gene_file> <out_dir>
    python assoc_pending.py clear_stale <gene_file> <out_dir> <feature_root>

Stdlib only: it runs under whichever interpreter run_assoc.sh picked.
"""
import csv
import os
import sys
from urllib.parse import quote

GENE_ID_COLUMN = "gene_id"
MISSING_FEATURES = "expected exactly one .pt/.npz feature file"


def read_gene_ids(gene_file: str) -> list[str]:
    with open(gene_file, newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if GENE_ID_COLUMN not in (reader.fieldnames or []):
            raise KeyError(f"{gene_file} has no {GENE_ID_COLUMN} column: "
                           f"{reader.fieldnames}")
        return list(dict.fromkeys(row[GENE_ID_COLUMN].strip() for row in reader
                                  if row[GENE_ID_COLUMN].strip()))


def result_path(out_dir: str, gene_id: str) -> str:
    """insample_hpp_AG.gene_result_path."""
    return os.path.join(out_dir, "per_gene", f"{quote(gene_id, safe='')}.tsv")


def feature_files(gene_id: str, feature_root: str) -> list[str]:
    """The files insample_hpp_AG.locate_features considers."""
    return [p for p in (os.path.join(feature_root, gene_id, f"{gene_id}.pt"),
                         os.path.join(feature_root, gene_id, f"{gene_id}.npz"),
                         os.path.join(feature_root, f"{gene_id}.pt"),
                         os.path.join(feature_root, f"{gene_id}.npz"))
             if os.path.isfile(p)]


def has_features(gene_id: str, feature_root: str) -> bool:
    """insample_hpp_AG.locate_features would find exactly one file."""
    return len(feature_files(gene_id, feature_root)) == 1


def input_failures(path: str) -> list[dict] | None:
    """The rows of an all-input_failed result file, else None."""
    with open(path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if rows and all(r.get("status") == "input_failed" for r in rows):
        return rows
    return None


def is_stale(path: str, gene_id: str, feature_root: str) -> bool:
    """An input failure that the gene's current features may not repeat."""
    files = feature_files(gene_id, feature_root)
    rows = input_failures(path)
    if len(files) != 1 or rows is None:
        return False
    missing = all(MISSING_FEATURES in (r.get("error") or "") for r in rows)
    return missing or os.path.getmtime(files[0]) > os.path.getmtime(path)


def pending_genes(gene_file: str, out_dir: str) -> list[str]:
    return [g for g in read_gene_ids(gene_file)
            if not os.path.exists(result_path(out_dir, g))]


def clear_stale(gene_file: str, out_dir: str, feature_root: str) -> list[str]:
    """Delete input failures the gene's current features may not repeat."""
    cleared = []
    for gene_id in read_gene_ids(gene_file):
        path = result_path(out_dir, gene_id)
        if os.path.exists(path) and is_stale(path, gene_id, feature_root):
            os.remove(path)
            cleared.append(gene_id)
    return cleared


def main(argv: list[str]) -> int:
    if len(argv) == 3 and argv[0] == "count":
        print(len(pending_genes(argv[1], argv[2])))
        return 0
    if len(argv) == 4 and argv[0] == "clear_stale":
        for gene_id in clear_stale(*argv[1:]):
            print(gene_id)
        return 0
    print(__doc__, file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

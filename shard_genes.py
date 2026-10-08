"""Split an assoc cell's gene file into disjoint shards of untested genes.

insample_hpp_AG.py tests a cell's genes one after another, ~45 min each at
43,680 participants, so a 70-gene cell is two days in one process. run_assoc.sh
runs one process per shard instead. A gene counts as tested once
<save_path>/<gene_id>.tsv exists (the upstream script's own skip rule), so a
rerun only shards what is left, and since every gene writes its own file the
shards can share a save_path.

    python shard_genes.py count <gene_file> <save_path>
    python shard_genes.py split <gene_file> <save_path> <n_shards> <shard_dir>

split prints one shard path per line; none when nothing is pending. Stdlib
only: it runs under whichever interpreter run_assoc.sh picked.
"""
import csv
import glob
import os
import sys

GENE_ID_COLUMN = "gene_id"


def read_gene_file(gene_file: str) -> tuple[list[str], list[dict]]:
    with open(gene_file, newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        fields = list(reader.fieldnames or [])
        if GENE_ID_COLUMN not in fields:
            raise KeyError(f"{gene_file} has no {GENE_ID_COLUMN} column: "
                           f"{fields}")
        return fields, list(reader)


def is_tested(gene_id: str, save_path: str) -> bool:
    return os.path.exists(os.path.join(save_path, f"{gene_id}.tsv"))


def pending_rows(gene_file: str, save_path: str) -> tuple[list[str], list[dict]]:
    fields, rows = read_gene_file(gene_file)
    return fields, [r for r in rows if not is_tested(r[GENE_ID_COLUMN], save_path)]


def pending_genes(gene_file: str, save_path: str) -> list[str]:
    return [r[GENE_ID_COLUMN] for r in pending_rows(gene_file, save_path)[1]]


def write_shards(gene_file: str, save_path: str, n_shards: int,
                 shard_dir: str) -> list[str]:
    """Round-robin the pending rows into at most n_shards files.

    Any shard files left by an earlier split are removed first, so a stale
    shard can never be picked up next to the new ones.
    """
    fields, rows = pending_rows(gene_file, save_path)
    os.makedirs(shard_dir, exist_ok=True)
    for stale in glob.glob(os.path.join(shard_dir, "shard*.tsv")):
        os.remove(stale)
    n = min(max(1, n_shards), len(rows))
    paths = []
    for i in range(n):
        path = os.path.join(shard_dir, f"shard{i}.tsv")
        with open(path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t",
                                    lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows[i::n])
        paths.append(path)
    return paths


def main(argv: list[str]) -> int:
    if len(argv) == 3 and argv[0] == "count":
        print(len(pending_genes(argv[1], argv[2])))
        return 0
    if len(argv) == 5 and argv[0] == "split":
        for path in write_shards(argv[1], argv[2], int(argv[3]), argv[4]):
            print(path)
        return 0
    print(__doc__, file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

"""Collapse a gene's per-participant rows to the distinct ones.

A gene TSV has one row per participant: (participant_id, ref_seq, hap1, hap2).
ref_seq is byte-identical on every row and array genotypes leave a gene with
few variants, so the distinct (ref, hap1, hap2) triples number in the tens
while the file has 43,706 rows. The embedding pass re-embeds every duplicate,
which is essentially the whole cost of the stage: measured 131,118 sequences
per gene against 37 distinct ones.

    stats    how many distinct rows each gene really has (no GPU needed)
    prepare  write the distinct-row TSV plus the index mapping every
             participant back to its row in that TSV

Two dedup levels, chosen with --mode:

    rows   distinct (ref, hap1, hap2) triples. One row per distinct triple,
           and the embedding script forwards both haplotypes of it.
    seqs   distinct SEQUENCES (default). A gene with H distinct haplotypes has
           up to H^2 distinct triples but only H distinct sequences, so this is
           another order of magnitude: 37 triples from 6 haplotypes cost 74
           forward passes as rows and 6 as sequences.

"seqs" is exact, not an approximation, because get_track_mean() never mixes
the haplotypes: it pools mut1 and mut2 separately and concatenates the two
results. A participant's saved vector is therefore [f(hap1), f(hap2)] for a
per-sequence f, and rebuilding it from per-sequence pieces reproduces it
bit for bit.

The packing exploits that same independence to avoid touching the model
script: distinct sequences are paired two per row (mut1 = one, mut2 = the
next), so a row's output holds two sequences' features side by side and
ceil(S/2) rows cover S sequences. The GPU script sees an ordinary TSV.

Genes are processed in parallel threads (--jobs): each one streams ~5 GB off
a FUSE mount, so the time goes on the filesystem rather than the CPU and the
GIL is released for the duration of every read.

Rows are streamed, never loaded as a frame: one gene is ~5 GB of sequence text
(43,706 rows x 3 columns x ~37 kb), and nothing here needs more than one row at
a time.
"""
import argparse
import csv
import hashlib
import json
import os
import queue
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

import config

try:                      # tqdm lives in the GPU env and most others here
    from tqdm.auto import tqdm
except ImportError:       # ...but the dedup must not require it
    tqdm = None

SEQ_COLUMNS = ("ref_seq", "mutation_seq_1", "mutation_seq_2")
ID_COLUMN = "participant_id"
COLUMNS = (ID_COLUMN,) + SEQ_COLUMNS

# A gene's sequences are tens of kb; the default 128 kB field limit would
# truncate them into silently wrong rows.
csv.field_size_limit(sys.maxsize)


def row_key(row: dict) -> str:
    """Hash of a row's sequence triple -- the identity that decides duplicates.

    Hashed rather than kept whole: holding every distinct triple twice costs
    hundreds of MB for no benefit, and sha1 collisions across tens of rows are
    not a realistic failure mode.
    """
    digest = hashlib.sha1()
    for column in SEQ_COLUMNS:
        digest.update(row[column].encode("ascii"))
        digest.update(b"\x00")
    return digest.hexdigest()


# Which terminal line each worker's bar owns. A thread takes a slot for the
# duration of one gene and returns it, so N workers use N stable lines instead
# of scribbling over each other.
_BAR_SLOTS: "queue.Queue[int]" = queue.Queue()


def estimate_rows(path: str) -> int | None:
    """Rows in a gene TSV, from the file size and one line -- no full read.

    Every sequence in a gene is the same length (the packing
    depends on it), so every data line is the
    same width bar a digit or two of participant id. Dividing the file size by
    one line's width therefore lands within a fraction of a percent, which is
    all a progress bar needs. Returns None if the file is unreadable or has no
    data rows, and the bar simply goes without a total.
    """
    try:
        size = os.path.getsize(path)
        with open(path) as fh:
            header = fh.readline()
            first = fh.readline()
        if not first:
            return None
        return max(1, round((size - len(header)) / len(first)))
    except OSError:
        return None


def row_progress(rows, gene_id: str, total: int | None = None):
    """Wrap the row stream in a per-gene tqdm bar, where tqdm is available.

    The total is an estimate from the file size (see estimate_rows), so the
    bar can show a percentage and an ETA without a counting pass. Without
    tqdm the rows pass through untouched -- progress is nice, not
    load-bearing.
    """
    if tqdm is None:
        return rows
    try:
        slot = _BAR_SLOTS.get_nowait()
    except queue.Empty:
        return rows

    def generate():
        bar = tqdm(total=total, desc=gene_id, position=slot, leave=False,
                   unit="row", unit_scale=True, dynamic_ncols=True)
        try:
            for item in rows:
                bar.update(1)
                yield item
        finally:
            bar.close()
            _BAR_SLOTS.put(slot)

    return generate()


def write_index(index: dict, path: str) -> None:
    """Write the index atomically.

    Via a .partial and a rename, like the TSV beside it: an interrupted write
    otherwise leaves truncated JSON, and the resume path then dies reading its
    own leftovers instead of rebuilding the gene.
    """
    tmp_path = f"{path}.partial"
    with open(tmp_path, "w") as fh:
        json.dump(index, fh)
    os.replace(tmp_path, path)


def read_index(path: str) -> dict | None:
    """The index, or None if it is missing or unreadable.

    A corrupt index means the gene was interrupted mid-write, which is a
    reason to rebuild it, not to abort the run: the rebuild is idempotent and
    costs one gene.
    """
    try:
        with open(path) as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def seq_key(sequence: str) -> str:
    """Hash of one sequence -- the identity that decides duplicate sequences."""
    return hashlib.sha1(sequence.encode("ascii")).hexdigest()


def read_rows(path: str):
    """Stream (row_number, row dict) without materializing the file.

    Lines are split by hand rather than handed to csv.DictReader, which
    buffers a large chunk per call -- punishing when one line is ~110 kB on a
    FUSE mount, where it reads as a hang. Sequences are plain ACGT with no
    tabs or quoting, so there is nothing for a real CSV parser to do.
    """
    with open(path) as fh:
        header_line = fh.readline()
        if not header_line:
            raise KeyError(f"{path} is empty")
        header = header_line.rstrip("\n").split("\t")
        missing = set(COLUMNS) - set(header)
        if missing:
            raise KeyError(f"{path} is missing column(s) {sorted(missing)}: "
                           f"{header}")
        for i, line in enumerate(fh):
            fields = line.rstrip("\n").split("\t")
            if len(fields) != len(header):
                raise ValueError(f"{path} row {i} has {len(fields)} field(s), "
                                 f"header has {len(header)}")
            yield i, dict(zip(header, fields))


def gene_stats(path: str) -> dict:
    gene_id = os.path.basename(path)[:-len(".tsv")]
    keys, n_rows = set(), 0
    for _, row in row_progress(read_rows(path), gene_id, estimate_rows(path)):
        keys.add(row_key(row))
        n_rows += 1
    return {
        "gene_id": gene_id,
        "participants": n_rows,
        "distinct_rows": len(keys),
        "speedup": n_rows / len(keys) if keys else float("nan"),
    }


def prepare_gene(path: str, out_dir: str, force: bool = False) -> dict:
    """Write <out_dir>/<gene>.tsv (distinct rows) and <gene>.index.json.

    The TSV keeps the schema the embedding script expects, with a synthetic
    participant_id per distinct row. The index records, for every original
    participant in order, which of those rows is theirs.

    A gene whose TSV and index are both already there is left alone: re-reading
    5 GB of sequence to rebuild a file that cannot have changed is the slowest
    part of a resumed run. Both files must exist -- a TSV without its index is
    unusable, so that counts as unfinished, not as done.
    """
    gene_id = os.path.basename(path)[:-len(".tsv")]
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{gene_id}.tsv")
    index_path = os.path.join(out_dir, f"{gene_id}.index.json")
    tmp_path = f"{out_path}.partial"

    index = None if force else read_index(index_path)
    if index is not None and os.path.exists(out_path) and "row_index" in index:
        return {"gene_id": gene_id, "participants": len(index["participant_id"]),
                "distinct_rows": index["n_distinct"],
                "speedup": (len(index["participant_id"]) / index["n_distinct"]
                            if index["n_distinct"] else float("nan")),
                "skipped": True}

    slot_of: dict[str, int] = {}
    participants: list[str] = []
    row_index: list[int] = []

    with open(tmp_path, "w", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=COLUMNS, delimiter="\t",
                                lineterminator="\n")
        writer.writeheader()
        for _, row in row_progress(read_rows(path), gene_id,
                                   estimate_rows(path)):
            key = row_key(row)
            slot = slot_of.get(key)
            if slot is None:
                slot = len(slot_of)
                slot_of[key] = slot
                writer.writerow({ID_COLUMN: f"uniq{slot}",
                                 **{c: row[c] for c in SEQ_COLUMNS}})
            participants.append(row[ID_COLUMN])
            row_index.append(slot)
    os.replace(tmp_path, out_path)

    write_index({
        "gene_id": gene_id,
        "participant_id": participants,
        "row_index": row_index,
        "n_distinct": len(slot_of),
    }, index_path)

    return {"gene_id": gene_id, "participants": len(participants),
            "distinct_rows": len(slot_of),
            "speedup": len(participants) / len(slot_of) if slot_of else float("nan"),
            "skipped": False}


def prepare_gene_seqs(path: str, out_dir: str, force: bool = False) -> dict:
    """Sequence-level dedup: write the packed TSV and its index.

    The TSV holds each distinct HAPLOTYPE sequence once, two per row, so the
    unmodified embedding script covers S sequences in ceil(S/2) rows. ref_seq
    is required by the script's dataset but never reaches the saved output
    (get_track_mean reads mut1/mut2 only), so it repeats the row's own first
    sequence rather than carrying a third distinct one.

    The index records, per participant, the sequence slot of each haplotype --
    in haplotype order, never sorted: the two halves of the saved vector are
    hap1 then hap2, and swapping them would silently transpose every
    participant's features.

    An odd number of sequences leaves the last row's second half unused; it
    repeats the last sequence, which costs one duplicate pass and keeps the
    TSV rectangular.
    """
    gene_id = os.path.basename(path)[:-len(".tsv")]
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{gene_id}.tsv")
    index_path = os.path.join(out_dir, f"{gene_id}.index.json")
    tmp_path = f"{out_path}.partial"

    index = None if force else read_index(index_path)
    if index is not None and os.path.exists(out_path):
        if index.get("mode") == "seqs":
            return {"gene_id": gene_id,
                    "participants": len(index["participant_id"]),
                    "distinct_rows": index["n_packed_rows"],
                    "distinct_seqs": index["n_distinct"],
                    "speedup": (2 * len(index["participant_id"]) /
                                index["n_distinct"]
                                if index["n_distinct"] else float("nan")),
                    "skipped": True}

    # The distinct sequences are held in memory, unlike the row-level path
    # which needs only hashes: a gene has tens of them at ~37 kb, so ~1 MB,
    # while the file it is streamed from is ~5 GB.
    slot_of: dict[str, int] = {}
    ordered: list[str] = []
    participants: list[str] = []
    hap_slots: list[list[int]] = []
    for _, row in row_progress(read_rows(path), gene_id, estimate_rows(path)):
        slots = []
        for column in (SEQ_COLUMNS[1], SEQ_COLUMNS[2]):
            key = seq_key(row[column])
            slot = slot_of.get(key)
            if slot is None:
                slot = len(ordered)
                slot_of[key] = slot
                ordered.append(row[column])
            slots.append(slot)
        participants.append(row[ID_COLUMN])
        hap_slots.append(slots)

    with open(tmp_path, "w", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=COLUMNS, delimiter="\t",
                                lineterminator="\n")
        writer.writeheader()
        for pair_index, (first, second) in enumerate(pack_pairs(ordered)):
            writer.writerow({ID_COLUMN: f"uniq{pair_index}",
                             SEQ_COLUMNS[0]: first,
                             SEQ_COLUMNS[1]: first,
                             SEQ_COLUMNS[2]: second})
    os.replace(tmp_path, out_path)

    n_packed = (len(ordered) + 1) // 2
    write_index({
        "gene_id": gene_id,
        "mode": "seqs",
        "participant_id": participants,
        "hap_slots": hap_slots,
        "n_distinct": len(ordered),
        "n_packed_rows": n_packed,
    }, index_path)

    return {"gene_id": gene_id, "participants": len(participants),
            "distinct_rows": n_packed, "distinct_seqs": len(ordered),
            "speedup": (2 * len(participants) / len(ordered)
                        if ordered else float("nan")),
            "skipped": False}


def pack_pairs(sequences: list[str]):
    """[a, b, c] -> (a, b), (c, c). The odd tail repeats rather than pads."""
    for i in range(0, len(sequences), 2):
        first = sequences[i]
        second = sequences[i + 1] if i + 1 < len(sequences) else first
        yield first, second


def gene_paths(gene_dir: str) -> list[str]:
    return sorted(os.path.join(gene_dir, f) for f in os.listdir(gene_dir)
                  if f.endswith(".tsv"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("stats", "prepare"))
    parser.add_argument("--mode", choices=("seqs", "rows"), default="seqs",
                        help="dedup distinct sequences (default) or distinct "
                             "(ref, hap1, hap2) rows")
    parser.add_argument("--gene_dir", default=config.GENE_SEQ_DIR)
    parser.add_argument("--out_dir", default=config.UNIQUE_GENE_SEQ_DIR)
    parser.add_argument("--force", action="store_true",
                        help="rebuild genes whose distinct-row TSV already exists")
    parser.add_argument("--jobs", type=int, default=8,
                        help="genes to process in parallel; each streams ~5 GB "
                             "off FUSE, so this waits on I/O, not the CPU")
    args = parser.parse_args()

    paths = gene_paths(args.gene_dir)
    if not paths:
        raise SystemExit(f"no gene TSVs under {args.gene_dir}")

    for slot in range(max(1, args.jobs)):
        _BAR_SLOTS.put(slot + 1)   # line 0 stays free for the [k/n] lines

    def process(path: str) -> dict:
        gene_id = os.path.basename(path)[:-len(".tsv")]
        if tqdm is None:
            print(f"  reading {gene_id}", flush=True)
        if args.command == "stats":
            return gene_stats(path)
        if args.mode == "seqs":
            return prepare_gene_seqs(path, args.out_dir, force=args.force)
        return prepare_gene(path, args.out_dir, force=args.force)

    print(f"{len(paths)} gene(s), {args.jobs} at a time", flush=True)
    total_rows = total_distinct = 0
    done = 0
    # as_completed, not map: map yields in SUBMISSION order, so a slow first
    # gene hides the fact that dozens of others already finished -- which
    # reads as a hang on a run where every gene takes minutes.
    with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as pool:
        futures = {pool.submit(process, path): path for path in paths}
        for future in as_completed(futures):
            row = future.result()
            done += 1
            total_rows += row["participants"]
            total_distinct += row["distinct_rows"]
            print(f"[{done}/{len(paths)}] {row}", flush=True)

    factor = total_rows / total_distinct if total_distinct else float("nan")
    print(f"\n{len(paths)} genes: {total_rows:,} rows -> {total_distinct:,} "
          f"distinct ({factor:.0f}x fewer forward passes)")
    if args.command == "prepare":
        print(f"Wrote {args.out_dir}")


if __name__ == "__main__":
    main()

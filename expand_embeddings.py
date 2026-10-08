"""Fan a deduplicated gene embedding back out to one row per participant.

get_hpp_embedding-AG_clean.py saves {"subject_id": [...], "concat_*": tensor}
where every tensor's dim 0 runs along subjects in dataset order and every row
is [f(mut1), f(mut2)] for a per-sequence f. Run on the deduplicated TSVs those
subjects are synthetic ("uniq<k>"), so the per-participant file is a gather
with the index dedup_gene_sequences.py recorded. Identical input gives
identical embeddings, so this reproduces the undeduplicated result exactly --
a gather, not an approximation.

Both dedup modes land here, told apart by the index's "mode":

    rows   a row IS a participant's row, so the gather is one index_select.
    seqs   a row holds two distinct sequences side by side. Each tensor is
           split into its per-sequence halves, and a participant's vector is
           rebuilt as [half(hap1), half(hap2)] -- in that order, because the
           two halves are not interchangeable.

Every gene with a .pt goes through here, including ones an earlier run
embedded, because the variance check is the only thing that catches a gene
whose embeddings are constant. --check_only runs that check alone, which
run_embeddings.sh does BEFORE the model pass so a bad gene left by an earlier
run surfaces immediately rather than hours later.

Needs torch, so it runs on the GPU node alongside the embedding stage
(run_embeddings.sh chains it automatically).
"""
import argparse
import json
import os

import torch

import config

SUBJECT_KEY = "subject_id"

# A gene whose embedding is constant across distinct sequences carries no
# signal for the association test: every participant gets the same features and
# the F-test sees a column of zeros variance. That is never a real result --
# it means the sequences never reached the model, the wrong column was read, or
# the forward pass returned a padding-only output. Checked per gene rather than
# discovered later as an inexplicable p-value.
VARIANCE_EPS = 0.0


def expand(embedding: dict, index: dict) -> dict:
    """Per-distinct-row embedding + index -> per-participant embedding."""
    subjects = embedding.get(SUBJECT_KEY)
    if subjects is None:
        raise KeyError(f"embedding has no {SUBJECT_KEY}: {sorted(embedding)}")

    # The mirror of the seqs-mode check: a packed embedding has far fewer
    # subjects than this index has distinct rows, and pairing them would be
    # silently wrong rather than short.
    if len(subjects) != index["n_distinct"]:
        raise ValueError(
            f"{index['gene_id']}: embedding has {len(subjects)} subjects but "
            f"the index expects {index['n_distinct']} distinct row(s) -- the "
            f".pt and the index come from different dedup runs; delete the "
            f"embedding and re-embed"
        )

    # The embedding script may reorder or shard subjects, so map by name rather
    # than trusting that row k is "uniq<k>".
    position = {str(s): i for i, s in enumerate(subjects)}
    missing = [f"uniq{k}" for k in range(index["n_distinct"])
               if f"uniq{k}" not in position]
    if missing:
        raise ValueError(
            f"{index['gene_id']}: {len(missing)} distinct row(s) were never "
            f"embedded (e.g. {missing[:5]}) -- the embedding run is incomplete"
        )

    gather = torch.tensor([position[f"uniq{slot}"] for slot in index["row_index"]],
                          dtype=torch.long)
    out = {SUBJECT_KEY: list(index["participant_id"])}
    for key, value in embedding.items():
        if key == SUBJECT_KEY:
            continue
        if not isinstance(value, torch.Tensor):
            out[key] = value
            continue
        if value.shape[0] != len(subjects):
            raise ValueError(
                f"{index['gene_id']}: {key} has {value.shape[0]} rows for "
                f"{len(subjects)} subjects -- not a per-subject tensor"
            )
        out[key] = value.index_select(0, gather).contiguous()
    return out


def sequence_halves(value: torch.Tensor, n_distinct: int,
                    gene_id: str, key: str) -> torch.Tensor:
    """(packed rows, 2*half) -> (n_distinct, half), one row per sequence.

    Packed row r holds sequences 2r and 2r+1, so the halves in row-major order
    ARE the sequences in slot order. An odd sequence count leaves a trailing
    duplicate half, which is dropped.
    """
    width = value.shape[-1]
    if width % 2:
        raise ValueError(f"{gene_id}: {key} is {width} wide, which cannot be "
                         f"two haplotype halves")
    half = width // 2
    halves = value.reshape(-1, half)
    if halves.shape[0] < n_distinct:
        raise ValueError(
            f"{gene_id}: {key} holds {halves.shape[0]} sequence halves for "
            f"{n_distinct} distinct sequences -- the embedding run is incomplete"
        )
    return halves[:n_distinct]


def per_sequence(embedding: dict, index: dict) -> dict:
    """Packed embedding -> {key: (n_distinct, half)}, one row per sequence.

    This, not the packed tensor, is what the variance check must look at: two
    sequences ride in one packed row, so a gene with two distinct sequences
    has a single row and would read as constant however much its sequences
    differ.
    """
    subjects = embedding.get(SUBJECT_KEY)
    if subjects is None:
        raise KeyError(f"embedding has no {SUBJECT_KEY}: {sorted(embedding)}")

    gene_id = index["gene_id"]
    n_packed = index["n_packed_rows"]
    # An embedding left by a row-mode run has one subject per distinct TRIPLE,
    # not per packed row. Its "uniq<k>" names would resolve and its halves
    # would be read as sequence slots, producing silently wrong features for
    # every participant, so the count is checked before anything is gathered.
    if len(subjects) != n_packed:
        raise ValueError(
            f"{gene_id}: embedding has {len(subjects)} subjects but the index "
            f"expects {n_packed} packed row(s) -- the .pt and the index come "
            f"from different dedup runs; delete the embedding and re-embed"
        )
    position = {str(s): i for i, s in enumerate(subjects)}
    missing = [f"uniq{k}" for k in range(n_packed) if f"uniq{k}" not in position]
    if missing:
        raise ValueError(
            f"{gene_id}: {len(missing)} packed row(s) were never embedded "
            f"(e.g. {missing[:5]}) -- the embedding run is incomplete"
        )

    # The script may reorder or shard subjects, so put the packed rows back in
    # slot order before their halves are read as sequence slots.
    order = torch.tensor([position[f"uniq{k}"] for k in range(n_packed)],
                         dtype=torch.long)
    out = {}
    for key, value in embedding.items():
        if key == SUBJECT_KEY or not isinstance(value, torch.Tensor):
            continue
        if value.shape[0] != len(subjects):
            raise ValueError(
                f"{gene_id}: {key} has {value.shape[0]} rows for "
                f"{len(subjects)} subjects -- not a per-subject tensor"
            )
        out[key] = sequence_halves(value.index_select(0, order),
                                   index["n_distinct"], gene_id, key)
    return out


def expand_seqs(embedding: dict, index: dict) -> dict:
    """Per-sequence embedding + hap slots -> per-participant embedding."""
    hap_slots = torch.tensor(index["hap_slots"], dtype=torch.long)
    if hap_slots.ndim != 2 or hap_slots.shape[1] != 2:
        raise ValueError(f"{index['gene_id']}: hap_slots is not a list of pairs")

    by_sequence = per_sequence(embedding, index)
    out = {SUBJECT_KEY: list(index["participant_id"])}
    for key, value in embedding.items():
        if key == SUBJECT_KEY:
            continue
        if not isinstance(value, torch.Tensor):
            out[key] = value
            continue
        per_seq = by_sequence[key]
        # hap1 half then hap2 half, matching get_track_mean's torch.cat order.
        out[key] = torch.cat([per_seq.index_select(0, hap_slots[:, 0]),
                              per_seq.index_select(0, hap_slots[:, 1])],
                             dim=-1).contiguous()
    return out


def distinct_row_variance(embedding: dict) -> dict[str, float]:
    """Max spread across DISTINCT rows, per tensor.

    Measured before the fan-out: after it, duplicated rows would inflate
    nothing but would make a degenerate tensor harder to spot.
    """
    spread = {}
    for key, value in embedding.items():
        if key == SUBJECT_KEY or not isinstance(value, torch.Tensor):
            continue
        if value.shape[0] < 2:
            spread[key] = float("nan")
            continue
        spread[key] = float(value.float().std(dim=0).max())
    return spread


def check_variance(gene_id: str, embedding: dict, n_distinct: int) -> dict:
    """Flag a gene whose embeddings do not vary across its distinct sequences.

    With one distinct row there is nothing to compare, so the check reports
    "single" rather than failing -- a gene with no variants legitimately gives
    every participant the same sequence.
    """
    spread = distinct_row_variance(embedding)
    if n_distinct < 2:
        return {"status": "single", "max_std": float("nan"), "flat_tracks": []}
    flat = sorted(k for k, v in spread.items() if not (v > VARIANCE_EPS))
    return {
        "status": "flat" if len(flat) == len(spread) else
                  "partially_flat" if flat else "ok",
        "max_std": max((v for v in spread.values() if v == v), default=float("nan")),
        "flat_tracks": flat,
    }


UP_TO_DATE = "up_to_date"


def is_up_to_date(out_path: str, *sources: str) -> bool:
    """The expanded .pt exists and is no older than anything it was built from.

    Without this every run re-expands every embedded gene -- 100+ genes, each a
    43,706-row write to the FUSE mount -- and every rank does it again.
    """
    if not os.path.exists(out_path):
        return False
    built = os.path.getmtime(out_path)
    return all(os.path.getmtime(src) <= built for src in sources)


def expand_gene(gene_id: str, unique_root: str, index_dir: str,
                out_root: str, check_only: bool = False,
                force: bool = False) -> dict:
    src = os.path.join(unique_root, gene_id, f"{gene_id}.pt")
    index_path = os.path.join(index_dir, f"{gene_id}.index.json")
    if not os.path.exists(src):
        raise FileNotFoundError(src)
    if not os.path.exists(index_path):
        raise FileNotFoundError(index_path)

    out_path = os.path.join(out_root, gene_id, f"{gene_id}.pt")
    if not check_only and not force and is_up_to_date(out_path, src, index_path):
        return {"gene_id": gene_id, "variance": UP_TO_DATE, "path": out_path}

    with open(index_path) as fh:
        index = json.load(fh)
    embedding = torch.load(src, map_location="cpu", weights_only=False)
    seqs_mode = index.get("mode") == "seqs"
    # In seqs mode the packed rows are not the unit of variation; the
    # sequences inside them are.
    measured = ({SUBJECT_KEY: [], **per_sequence(embedding, index)}
                if seqs_mode else embedding)
    variance = check_variance(gene_id, measured, index["n_distinct"])
    if check_only:
        return {"gene_id": gene_id,
                "distinct_rows": index.get("n_packed_rows",
                                           index["n_distinct"]),
                "participants": len(index["participant_id"]),
                "variance": variance["status"], "max_std": variance["max_std"],
                "flat_tracks": variance["flat_tracks"], "path": None}
    out = expand_seqs(embedding, index) if seqs_mode else expand(embedding, index)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    # Temp + rename: a killed run must not leave a half-written .pt that the
    # assoc stage would happily torch.load.
    tmp_path = f"{out_path}.partial"
    torch.save(out, tmp_path)
    os.replace(tmp_path, out_path)
    return {"gene_id": gene_id,
            "distinct_rows": index.get("n_packed_rows", index["n_distinct"]),
            "participants": len(index["participant_id"]),
            "variance": variance["status"], "max_std": variance["max_std"],
            "flat_tracks": variance["flat_tracks"], "path": out_path}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--unique_root", default=config.UNIQUE_FEATURE_ROOT)
    parser.add_argument("--index_dir", default=config.UNIQUE_GENE_SEQ_DIR)
    parser.add_argument("--out_root", default=config.FEATURE_ROOT)
    parser.add_argument("--gene_id", default=None,
                        help="expand just this gene (default: all embedded)")
    parser.add_argument("--check_only", action="store_true",
                        help="run the variance check without writing anything; "
                             "used before the model pass to vet embeddings an "
                             "earlier run left behind")
    parser.add_argument("--force", action="store_true",
                        help="re-expand genes whose expanded .pt is up to date")
    args = parser.parse_args()

    if args.gene_id:
        gene_ids = [args.gene_id]
    else:
        gene_ids = sorted(
            d for d in os.listdir(args.unique_root)
            if os.path.exists(os.path.join(args.unique_root, d, f"{d}.pt"))
        ) if os.path.isdir(args.unique_root) else []
    if not gene_ids:
        raise SystemExit(f"no embedded genes under {args.unique_root}")

    results = []
    for gene_id in gene_ids:
        result = expand_gene(gene_id, args.unique_root, args.index_dir,
                             args.out_root, check_only=args.check_only,
                             force=args.force)
        results.append(result)
        print(result, flush=True)

    n_current = sum(r["variance"] == UP_TO_DATE for r in results)
    print(f"\nChecked {len(gene_ids)} gene(s)" if args.check_only
          else f"\nExpanded {len(gene_ids) - n_current} gene(s) to "
               f"{args.out_root} ({n_current} already up to date)")
    flat = [r for r in results if r["variance"] in ("flat", "partially_flat")]
    single = [r for r in results if r["variance"] == "single"]
    if single:
        print(f"{len(single)} gene(s) have a single distinct sequence, so "
              f"constant embeddings are expected: "
              f"{[r['gene_id'] for r in single]}")
    if flat:
        for result in flat:
            print(f"WARNING {result['gene_id']}: embeddings do not vary across "
                  f"{result['distinct_rows']} distinct sequences "
                  f"(max std {result['max_std']:.3g}, flat tracks "
                  f"{result['flat_tracks']})")
        raise SystemExit(
            f"{len(flat)} gene(s) produced constant embeddings -- the "
            f"association test on them would be meaningless. Investigate "
            f"before trusting any result for them."
        )


if __name__ == "__main__":
    main()

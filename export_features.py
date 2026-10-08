"""Turn the torch .pt embeddings into a torch-free feature table per gene.

Why: assoc does statistics, not tensor work, but it used to torch.load() the
.pt files -- which pinned it to the GPU cluster, because deep_learning is the
only env with a torch compiled against that cluster's glibc. The tensor half
of assoc is one deterministic reshape of the saved embeddings, so it belongs
with the embedding stage: run it once on the GPU node, write plain arrays, and
the association tests then run anywhere numpy runs.

The views written here are exactly the ones insample_hpp_AG.py builds, so the
export is a move, not a reimplementation: the concat_* tracks as saved, their
TSS-500 counterparts, and the add_* views that fold the two haplotype halves
of a concat track together.

Runs in an env with torch (the GPU node, chained off run_embeddings.sh):
    python export_features.py                  # every gene with a .pt
    python export_features.py --genes ENSG...  # a subset
    python export_features.py --format csv     # plain text instead of .npz
"""
import argparse
import os
import sys

import numpy as np
import torch

import config

SUBJECT_KEY = "subject_id"

# view name -> key in the saved .pt. The four tracks, each in its full-gene
# and TSS-500 form.
CONCAT_VIEWS = {
    "concat_cage": "concat_cage_mean",
    "concat_rnaseq": "concat_rna_seq_mean",
    "concat_atac": "concat_atac_mean",
    "concat_dnase": "concat_dnase_mean",
    "concat_tss_cage": "concat_cage_tss500_mean",
    "concat_tss_rnaseq": "concat_rna_seq_tss500_mean",
    "concat_tss_atac": "concat_atac_tss500_mean",
    "concat_tss_dnase": "concat_dnase_tss500_mean",
}

# add_<x> sums the two halves of concat_<x>. The half widths are the track
# dimensions insample_hpp_AG.py hardcodes; they are asserted against the data
# rather than inferred, so an embedding with a different track layout fails
# loudly here instead of producing a silently wrong fold.
TRACK_DIM = {"cage": 546, "rnaseq": 667, "atac": 167, "dnase": 305}

ADD_VIEWS = {f"add_{name.removeprefix('concat_')}": name for name in CONCAT_VIEWS}


def track_of(view: str) -> str:
    """concat_tss_rnaseq -> rnaseq"""
    return view.rsplit("_", 1)[-1]


def fold_halves(view: str, values: np.ndarray) -> np.ndarray:
    half = TRACK_DIM[track_of(view)]
    width = values.shape[-1]
    if width != 2 * half:
        raise ValueError(
            f"{view} is {width} wide but {track_of(view)} tracks are "
            f"{half} per haplotype -- the embedding layout changed"
        )
    return values[..., :half] + values[..., half:]


def build_views(data: dict) -> dict[str, np.ndarray]:
    """The feature views assoc tests, as float32 numpy arrays."""
    missing = [key for key in CONCAT_VIEWS.values() if key not in data]
    if missing:
        raise KeyError(f"embedding is missing {missing}; has {sorted(data)}")

    # .float() before numpy: the TSS tracks are saved in a reduced dtype and
    # numpy's copy=False path warns (and would keep bf16) otherwise.
    concat = {view: data[key].float().numpy()
              for view, key in CONCAT_VIEWS.items()}
    views = dict(concat)
    for view, source in ADD_VIEWS.items():
        views[view] = fold_halves(source, concat[source])
    return views


def export_gene(gene_id: str, feature_root: str, out_root: str,
                fmt: str = "npz", force: bool = False) -> dict:
    src = os.path.join(feature_root, gene_id, f"{gene_id}.pt")
    if not os.path.exists(src):
        raise FileNotFoundError(src)

    out_dir = os.path.join(out_root, gene_id)
    out_path = os.path.join(out_dir, f"{gene_id}.{'npz' if fmt == 'npz' else 'csv'}")
    marker = out_path if fmt == "npz" else os.path.join(out_dir, f"{gene_id}.subjects.csv")
    if not force and os.path.exists(out_path) and os.path.exists(marker):
        return {"gene_id": gene_id, "status": "exists", "path": out_path}

    data = torch.load(src, map_location="cpu", weights_only=False)
    subjects = data.get(SUBJECT_KEY)
    if subjects is None:
        raise KeyError(f"{gene_id}: embedding has no {SUBJECT_KEY}")
    subjects = np.asarray([str(s) for s in subjects])

    views = build_views(data)
    for view, values in views.items():
        if values.shape[0] != subjects.shape[0]:
            raise ValueError(
                f"{gene_id}: {view} has {values.shape[0]} rows for "
                f"{subjects.shape[0]} subjects"
            )

    os.makedirs(out_dir, exist_ok=True)
    if fmt == "npz":
        # Uncompressed on purpose: assoc reads these repeatedly, once per
        # trait/sex cell, and decompression would cost more than the disk.
        np.savez(out_path, **{SUBJECT_KEY: subjects}, **views)
    else:
        # One CSV per view keeps each file a plain rectangle of numbers, with
        # the subject order in its own file beside them.
        np.savetxt(marker, subjects, fmt="%s", header=SUBJECT_KEY, comments="")
        for view, values in views.items():
            np.savetxt(os.path.join(out_dir, f"{gene_id}.{view}.csv"),
                       values.reshape(values.shape[0], -1), delimiter=",")
        with open(out_path, "w") as fh:
            fh.write("\n".join(sorted(views)) + "\n")

    return {"gene_id": gene_id, "status": "written", "path": out_path,
            "subjects": int(subjects.shape[0]),
            "views": {view: list(values.shape) for view, values in views.items()}}


def genes_with_embeddings(feature_root: str) -> list[str]:
    if not os.path.isdir(feature_root):
        return []
    return sorted(gene for gene in os.listdir(feature_root)
                  if os.path.exists(os.path.join(feature_root, gene, f"{gene}.pt")))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature_root", default=config.FEATURE_ROOT)
    parser.add_argument("--out_root", default=config.FEATURE_TABLE_ROOT)
    parser.add_argument("--genes", nargs="*", default=None)
    parser.add_argument("--format", choices=("npz", "csv"), default="npz")
    parser.add_argument("--force", action="store_true",
                        help="re-export genes that already have a table")
    args = parser.parse_args()

    genes = args.genes or genes_with_embeddings(args.feature_root)
    if not genes:
        print(f"no embeddings under {args.feature_root}", file=sys.stderr)
        return 3

    failed = 0
    for gene_id in genes:
        try:
            result = export_gene(gene_id, args.feature_root, args.out_root,
                                 fmt=args.format, force=args.force)
        except Exception as exc:  # one bad gene must not sink the rest
            print(f"{gene_id}: FAILED {type(exc).__name__}: {exc}", file=sys.stderr)
            failed += 1
            continue
        print(result)

    print(f"exported {len(genes) - failed}/{len(genes)} gene(s) -> {args.out_root}",
          file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

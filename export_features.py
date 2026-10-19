"""Turn the torch .pt embeddings into a torch-free feature table per gene.

Why: assoc does statistics, not tensor work, but it used to torch.load() the
.pt files -- which pinned it to the GPU cluster, because deep_learning is the
only env with a torch compiled against that cluster's glibc. Writing the same
arrays as plain numpy once, on the GPU node, lets the association tests run
anywhere numpy runs.

The table holds the .pt's own fields under their own names: subject_id and the
eight per-track means (concat_<track>_mean, concat_<track>_tss500_mean),
converted to numpy by the same rule fmgwas/insample_hpp_AG.py applies when it
reads a .pt (bfloat16 -> float32, other dtypes kept). That script accepts <gene>/<gene>.npz in place of
<gene>/<gene>.pt and builds every concat/add view from these fields itself, so
the table changes nothing it computes. Tables from before 2026-10-19 held the
old script's derived views instead; they lack these fields and are rewritten.

Runs in an env with torch (the GPU node, chained off run_embeddings.sh):
    python export_features.py                  # every gene with a .pt
    python export_features.py --genes ENSG...  # a subset

A table is rewritten when it lacks a raw field or is older than its .pt.
"""
import argparse
import os
import sys

import numpy as np
import torch

import config

SUBJECT_KEY = "subject_id"

# The .pt fields FM-GWAS reads (insample_hpp_AG.track_field): four tracks, each
# as a gene-body mean and a TSS-500 mean.
RAW_FIELDS = tuple(
    f"concat_{track}_{window}"
    for window in ("mean", "tss500_mean")
    for track in ("cage", "dnase", "atac", "rna_seq")
)


def as_numpy(value) -> np.ndarray:
    """insample_hpp_AG.numpy_value, verbatim in effect: tensors narrower than
    32 bits (the bfloat16 TSS tracks) go to float32, anything else keeps its
    dtype."""
    if hasattr(value, "detach"):
        value = value.detach().cpu()
        if value.is_floating_point() and value.element_size() < 4:
            value = value.float()
        value = value.numpy()
    return np.asarray(value)


def raw_fields(data: dict) -> dict[str, np.ndarray]:
    """The per-track fields as numpy, exactly as FM-GWAS's .pt reader sees them."""
    missing = [key for key in RAW_FIELDS if key not in data]
    if missing:
        raise KeyError(f"embedding is missing {missing}; has {sorted(data)}")
    return {key: as_numpy(data[key]) for key in RAW_FIELDS}


def is_current(path: str) -> bool:
    """An .npz table already holds every raw field (not an old views table)."""
    with np.load(path, allow_pickle=False) as handle:
        return SUBJECT_KEY in handle.files and all(k in handle.files for k in RAW_FIELDS)


def export_gene(gene_id: str, feature_root: str, out_root: str,
                force: bool = False) -> dict:
    src = os.path.join(feature_root, gene_id, f"{gene_id}.pt")
    if not os.path.exists(src):
        raise FileNotFoundError(src)

    out_dir = os.path.join(out_root, gene_id)
    out_path = os.path.join(out_dir, f"{gene_id}.npz")
    if (not force and os.path.exists(out_path) and is_current(out_path)
            and os.path.getmtime(src) <= os.path.getmtime(out_path)):
        return {"gene_id": gene_id, "status": "exists", "path": out_path}

    data = torch.load(src, map_location="cpu", weights_only=False)
    subjects = data.get(SUBJECT_KEY)
    if subjects is None:
        raise KeyError(f"{gene_id}: embedding has no {SUBJECT_KEY}")
    subjects = np.asarray([str(s) for s in subjects])

    fields = raw_fields(data)
    for key, values in fields.items():
        if values.ndim != 2 or values.shape[0] != subjects.shape[0]:
            raise ValueError(
                f"{gene_id}: {key} is {values.shape} for "
                f"{subjects.shape[0]} subjects"
            )

    os.makedirs(out_dir, exist_ok=True)
    # Uncompressed on purpose: assoc reads these repeatedly, once per
    # trait/sex cell, and decompression would cost more than the disk.
    # Temp + rename, so a killed export never leaves a half-written table.
    tmp_path = os.path.join(out_dir, f".{gene_id}.partial.npz")
    np.savez(tmp_path, **{SUBJECT_KEY: subjects}, **fields)
    os.replace(tmp_path, out_path)

    return {"gene_id": gene_id, "status": "written", "path": out_path,
            "subjects": int(subjects.shape[0]),
            "fields": {key: [list(v.shape), str(v.dtype)] for key, v in fields.items()}}


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
                                 force=args.force)
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

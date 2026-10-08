"""Read the feature tables export_features.py wrote. numpy only, no torch.

This is the assoc side of the split: the GPU node turns each gene's .pt into
one table of the views the association tests use, and this module hands those
views back in the same shape insample_hpp_AG.py built them in. Importing this
must not pull torch in -- that is the whole point of the split.
"""
import os

import numpy as np

SUBJECT_KEY = "subject_id"


def table_path(gene_id: str, root: str, fmt: str = "npz") -> str:
    return os.path.join(root, gene_id, f"{gene_id}.{'npz' if fmt == 'npz' else 'csv'}")


def has_table(gene_id: str, root: str) -> bool:
    return os.path.exists(table_path(gene_id, root))


def load_gene(gene_id: str, root: str) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """(subject ids, {view name: array with subjects along dim 0})."""
    npz_path = table_path(gene_id, root, "npz")
    if os.path.exists(npz_path):
        with np.load(npz_path, allow_pickle=False) as handle:
            subjects = handle[SUBJECT_KEY].astype(str)
            views = {key: handle[key] for key in handle.files if key != SUBJECT_KEY}
        return subjects, views

    # CSV export: one rectangle per view, the subject order in its own file,
    # and a manifest naming the views so a half-written directory is visible
    # rather than silently short.
    manifest = table_path(gene_id, root, "csv")
    if not os.path.exists(manifest):
        raise FileNotFoundError(
            f"{gene_id}: no feature table under {root} -- run export_features.py "
            f"on the GPU node"
        )
    out_dir = os.path.dirname(manifest)
    with open(manifest) as fh:
        names = [line.strip() for line in fh if line.strip()]
    subjects = np.loadtxt(os.path.join(out_dir, f"{gene_id}.subjects.csv"),
                          dtype=str, skiprows=1, ndmin=1)
    views = {name: np.loadtxt(os.path.join(out_dir, f"{gene_id}.{name}.csv"),
                              delimiter=",", ndmin=2)
             for name in names}
    return subjects, views

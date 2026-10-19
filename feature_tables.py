"""Read the feature tables export_features.py wrote. numpy only, no torch.

The association stage itself does not use this module: fmgwas/insample_hpp_AG.py
reads <gene>/<gene>.npz directly. This is for the diagnostics
(diagnose_assoc.py, compare_feature_tables.py), which work on the 16
single-track views the July/d17eae23a6 script tested. Importing this must not
pull torch in -- that is the whole point of the split.
"""
import os

import numpy as np

SUBJECT_KEY = "subject_id"

# view name -> raw .pt field (export_features.RAW_FIELDS)
CONCAT_VIEWS = {
    f"concat_{prefix}{track}": f"concat_{source}_{window}"
    for prefix, window in (("", "mean"), ("tss_", "tss500_mean"))
    for track, source in (("cage", "cage"), ("rnaseq", "rna_seq"),
                          ("atac", "atac"), ("dnase", "dnase"))
}


def table_path(gene_id: str, root: str) -> str:
    return os.path.join(root, gene_id, f"{gene_id}.npz")


def has_table(gene_id: str, root: str) -> bool:
    return os.path.exists(table_path(gene_id, root))


def fold_halves(values: np.ndarray) -> np.ndarray:
    """add_* view: hap1 half + hap2 half of a concat track."""
    half = values.shape[-1] // 2
    if values.shape[-1] != 2 * half:
        raise ValueError(f"{values.shape[-1]} columns cannot be two haplotype halves")
    return values[..., :half] + values[..., half:]


def load_gene(gene_id: str, root: str) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """(subject ids, {view name: array with subjects along dim 0})."""
    path = table_path(gene_id, root)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{gene_id}: no feature table under {root} -- run export_features.py "
            f"on the GPU node"
        )
    with np.load(path, allow_pickle=False) as handle:
        subjects = handle[SUBJECT_KEY].astype(str)
        missing = [f for f in CONCAT_VIEWS.values() if f not in handle.files]
        if missing:
            raise KeyError(f"{path} lacks {missing}: an old-format table, "
                           f"re-run export_features.py")
        concat = {view: handle[field] for view, field in CONCAT_VIEWS.items()}
    views = dict(concat)
    views.update({"add_" + view.removeprefix("concat_"): fold_halves(values)
                  for view, values in concat.items()})
    return subjects, views

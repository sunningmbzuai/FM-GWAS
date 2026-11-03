"""Each gene's PCA-reduced embeddings, computed once and saved (Ning, 2026-11-03).

For every gene and every one of the 60 embedding types, the vendored FM-GWAS
script's own PCA (fmgwas/insample_hpp_AG.py: standardize, exact Gram
eigendecomposition, keep components up to 95% of the variance) is fitted once
per sex sample: All, F only and M only, each from trait_scan.py's
participants.tsv intersected with the gene's feature table. gene_scan.py then
reuses each sample's scores for every trait in that sex, instead of refitting
the PCA in each trait cell as run_assoc.sh does.

Array genotypes leave each gene with few distinct embedding rows, so the scores
are stored as their distinct rows plus a per-participant index; scores =
rows[index] exactly.

    <cache>/<gene>.npz   <sex>__participant_id, <sex>__<embed>__rows, <sex>__<embed>__index
    <cache>/<gene>.json  per sex: components and explained variance per embedding

    python gene_pca_cache.py --genes genes.tsv --participants participants.tsv \
        --feature-root feature_tables --cache DIR [--rank 0 --world-size 8]
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd

import assoc_settings

SEXES = (("All", None), ("F", 0), ("M", 1))   # FM-GWAS coding: F=0, M=1
HERE = Path(__file__).resolve().parent
FMGWAS_SCRIPT = HERE / "fmgwas" / "insample_hpp_AG.py"


def fmgwas() -> ModuleType:
    """The vendored association script as a module (run unmodified)."""
    spec = importlib.util.spec_from_file_location("insample_hpp_AG", FMGWAS_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("insample_hpp_AG", module)
    spec.loader.exec_module(module)
    return module


def log(message: str) -> None:
    print(f"[{time.strftime('%F %T')}] [gene_pca_cache] {message}", file=sys.stderr, flush=True)


def cache_paths(cache: Path, gene: str) -> tuple[Path, Path]:
    return cache / f"{gene}.npz", cache / f"{gene}.json"


def compress_rows(scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rows, index = np.unique(scores, axis=0, return_inverse=True)
    return rows, index.reshape(-1).astype(np.int32)


def load_cache(cache: Path, gene: str) -> tuple[dict[str, tuple], dict]:
    """({sex: (participant ids, {embed: N x k scores})}, meta) of one cached gene."""
    npz_path, json_path = cache_paths(cache, gene)
    meta = json.loads(json_path.read_text())
    by_sex = {}
    with np.load(npz_path) as data:
        for sex, info in meta.get("sexes", {}).items():
            scores = {e: data[f"{sex}__{e}__rows"][data[f"{sex}__{e}__index"]]
                      for e in info["embeddings"]}
            by_sex[sex] = (data[f"{sex}__participant_id"], scores)
    return by_sex, meta


def is_current(cache: Path, gene: str) -> bool:
    """Done in the per-sex layout. A file from the earlier all-participant
    layout (no "sexes" key) is rebuilt, never silently skipped."""
    json_path = cache_paths(cache, gene)[1]
    if not json_path.exists():
        return False
    try:
        return "sexes" in json.loads(json_path.read_text())
    except (OSError, ValueError):
        return False


def feature_dirs(feature_root: Path) -> dict[str, str]:
    """Unversioned gene id -> feature directory name (versions differ by release)."""
    return {entry.name.split(".")[0]: entry.name
            for entry in os.scandir(feature_root) if entry.is_dir()}


def fit_sample(fm: ModuleType, matrices: dict[str, np.ndarray], rows: np.ndarray,
               config: dict) -> tuple[dict[str, np.ndarray], dict]:
    """PCA of every embedding on one participant sample (rows of the feature table)."""
    arrays: dict[str, np.ndarray] = {}
    info: dict = dict(n_samples=int(len(rows)), embeddings=[], components={},
                      explained_variance={}, input_features={}, failed={})
    for embed, matrix in matrices.items():
        try:
            scores, fit = fm.fit_pca(matrix[rows], config)
        except (ValueError, np.linalg.LinAlgError) as error:
            info["failed"][embed] = str(error)
            continue
        arrays[f"{embed}__rows"], arrays[f"{embed}__index"] = compress_rows(scores)
        info["embeddings"].append(embed)
        info["components"][embed] = int(fit["n_components"])
        info["explained_variance"][embed] = float(fit["explained_variance"])
        info["input_features"][embed] = int(fit["input_features"])
    return arrays, info


def build_gene(fm: ModuleType, gene: str, feature_root: Path, people: pd.DataFrame,
               embeds: list[str], pca_spec: str,
               dirs: dict[str, str]) -> tuple[dict[str, np.ndarray], dict]:
    """One PCA per sex sample (All, F, M), each on that sample's participants only."""
    folder = dirs.get(gene.split(".")[0])
    if folder is None:
        raise ValueError("no feature table (untestable in UKB or not embedded)")
    ids, matrices = fm.load_gene(fm.locate_features(feature_root, folder), embeds)
    sex_of = dict(zip(people.participant_id.astype(np.int64), people.sex.astype(int)))
    config = fm.parse_pca(pca_spec)
    arrays: dict[str, np.ndarray] = {}
    meta: dict = dict(gene_id=gene, feature_dir=folder, gene_pca=pca_spec, sexes={})
    for sex, code in SEXES:
        rows = np.array([i for i, p in enumerate(ids)
                         if int(p) in sex_of and (code is None or sex_of[int(p)] == code)],
                        dtype=np.int64)
        if len(rows) < 3:
            log(f"[warn] {gene} {sex}: {len(rows)} participants with features, no PCA")
            continue
        sex_arrays, info = fit_sample(fm, matrices, rows, config)
        arrays[f"{sex}__participant_id"] = ids[rows].astype(np.int64)
        arrays.update({f"{sex}__{k}": v for k, v in sex_arrays.items()})
        meta["sexes"][sex] = info
    if not meta["sexes"]:
        raise ValueError("fewer than 3 scan participants have features")
    return arrays, meta


def write_gene(cache: Path, gene: str, arrays: dict, meta: dict) -> None:
    npz_path, json_path = cache_paths(cache, gene)
    tmp = npz_path.with_suffix(".tmp.npz")
    np.savez(tmp, **arrays)
    os.replace(tmp, npz_path)
    tmp_json = json_path.with_suffix(".tmp")
    tmp_json.write_text(json.dumps(meta, indent=1))
    os.replace(tmp_json, json_path)  # the json marks the gene done


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--genes", required=True)
    ap.add_argument("--participants", required=True)
    ap.add_argument("--feature-root", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--gene-pca", default=assoc_settings.GENE_PCA)
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--world-size", type=int, default=1)
    args = ap.parse_args(argv)

    fm = fmgwas()
    genes = fm.read_genes(args.genes)[args.rank::args.world_size]
    people = pd.read_csv(args.participants, sep="\t")
    cache = Path(args.cache)
    cache.mkdir(parents=True, exist_ok=True)
    embeds = list(assoc_settings.ALL_VIEWS)
    dirs = feature_dirs(Path(args.feature_root))
    start, done = time.monotonic(), 0
    for number, gene in enumerate(genes, 1):
        if is_current(cache, gene):
            continue
        t0 = time.monotonic()
        try:
            arrays, meta = build_gene(fm, gene, Path(args.feature_root), people,
                                      embeds, args.gene_pca, dirs)
        except (ValueError, KeyError, OSError) as error:
            log(f"[FAILED] {gene}: {error}")
            write_gene(cache, gene, {}, dict(gene_id=gene, sexes={}, error=str(error)))
            continue
        done += 1
        eta = (time.monotonic() - start) / done * (len(genes) - number)
        ks = [k for info in meta["sexes"].values() for k in info["components"].values()]
        log(f"[{number}/{len(genes)}] {gene}: PCA for {', '.join(meta['sexes'])}, "
            f"k {min(ks, default=0)}-{max(ks, default=0)}, "
            f"{time.monotonic() - t0:.0f}s, ETA {eta / 60:.1f} min")
        write_gene(cache, gene, arrays, meta)
    return 0


if __name__ == "__main__":
    sys.exit(main())

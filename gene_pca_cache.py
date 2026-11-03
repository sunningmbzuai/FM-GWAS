"""Each gene's PCA-reduced embeddings, computed once and saved (Ning, 2026-11-03).

For every gene and every one of the 60 embedding types, the vendored FM-GWAS
script's own PCA (fmgwas/insample_hpp_AG.py: standardize, exact Gram
eigendecomposition, keep components up to 95% of the variance) is fitted once
on the scan participants (trait_scan.py's participants.tsv, intersected with
the gene's feature table). gene_scan.py then reuses these scores for every
trait and sex, instead of refitting the PCA in each cell as run_assoc.sh does.

Array genotypes leave each gene with few distinct embedding rows, so the scores
are stored as their distinct rows plus a per-participant index; scores =
rows[index] exactly.

    <cache>/<gene>.npz   participant_id, <embed>__rows, <embed>__index
    <cache>/<gene>.json  components and explained variance per embedding

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


def load_cache(cache: Path, gene: str) -> tuple[np.ndarray, dict[str, np.ndarray], dict]:
    """(participant ids, {embed: N x k scores}, meta) of one cached gene."""
    npz_path, json_path = cache_paths(cache, gene)
    meta = json.loads(json_path.read_text())
    with np.load(npz_path) as data:
        ids = data["participant_id"]
        scores = {e: data[f"{e}__rows"][data[f"{e}__index"]] for e in meta["embeddings"]}
    return ids, scores, meta


def feature_dirs(feature_root: Path) -> dict[str, str]:
    """Unversioned gene id -> feature directory name (versions differ by release)."""
    return {entry.name.split(".")[0]: entry.name
            for entry in os.scandir(feature_root) if entry.is_dir()}


def build_gene(fm: ModuleType, gene: str, feature_root: Path, wanted: np.ndarray,
               embeds: list[str], pca_spec: str,
               dirs: dict[str, str]) -> tuple[dict[str, np.ndarray], dict]:
    folder = dirs.get(gene.split(".")[0])
    if folder is None:
        raise ValueError("no feature table (untestable in UKB or not embedded)")
    ids, matrices = fm.load_gene(fm.locate_features(feature_root, folder), embeds)
    keep = np.isin(ids, wanted)
    if keep.sum() < 3:
        raise ValueError(f"{int(keep.sum())} scan participants have features")
    config = fm.parse_pca(pca_spec)
    arrays: dict[str, np.ndarray] = {"participant_id": ids[keep].astype(np.int64)}
    meta: dict = dict(gene_id=gene, feature_dir=folder, gene_pca=pca_spec, n_samples=int(keep.sum()),
                      embeddings=[], components={}, explained_variance={},
                      input_features={}, failed={})
    for embed in embeds:
        try:
            scores, info = fm.fit_pca(matrices[embed][keep], config)
        except (ValueError, np.linalg.LinAlgError) as error:
            meta["failed"][embed] = str(error)
            continue
        rows, index = compress_rows(scores)
        arrays[f"{embed}__rows"], arrays[f"{embed}__index"] = rows, index
        meta["embeddings"].append(embed)
        meta["components"][embed] = int(info["n_components"])
        meta["explained_variance"][embed] = float(info["explained_variance"])
        meta["input_features"][embed] = int(info["input_features"])
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
    wanted = pd.read_csv(args.participants, sep="\t").participant_id.to_numpy(np.int64)
    cache = Path(args.cache)
    cache.mkdir(parents=True, exist_ok=True)
    embeds = list(assoc_settings.ALL_VIEWS)
    dirs = feature_dirs(Path(args.feature_root))
    start, done = time.monotonic(), 0
    for number, gene in enumerate(genes, 1):
        if cache_paths(cache, gene)[1].exists():
            continue
        t0 = time.monotonic()
        try:
            arrays, meta = build_gene(fm, gene, Path(args.feature_root), wanted,
                                      embeds, args.gene_pca, dirs)
        except (ValueError, KeyError, OSError) as error:
            log(f"[FAILED] {gene}: {error}")
            write_gene(cache, gene, {"participant_id": np.empty(0, np.int64)},
                       dict(gene_id=gene, embeddings=[], error=str(error)))
            continue
        done += 1
        eta = (time.monotonic() - start) / done * (len(genes) - number)
        log(f"[{number}/{len(genes)}] {gene}: {len(meta['embeddings'])} embeddings, "
            f"k {min(meta['components'].values(), default=0)}-"
            f"{max(meta['components'].values(), default=0)}, "
            f"{time.monotonic() - t0:.0f}s, ETA {eta / 60:.1f} min")
        write_gene(cache, gene, arrays, meta)
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Gene association with the analytic partial F test and (binary traits) the logistic LRT.

Nested models, fitted on one shared cohort per gene:
    reduced: y ~ 1 + covariates [+ cell PCs]
    full:    y ~ 1 + covariates [+ cell PCs] + gene PCs
F test   (--tests F):   OLS partial F; asks whether the gene PCs explain additional variance.
LRT      (--tests LRT): logistic regression, 2*(loglik_full - loglik_reduced) ~ chi2(df),
                        df = number of added gene-PC dimensions. Binary 0/1 labels only;
                        quantitative traits get lrt_status=not_binary.
Both tests use the same cohort, design matrices and rank cutoff, so df_lrt == df1.
Permutation p values: see gene_assoc_permutation.py (imports this module).

Gene and cell embeddings each get their own PCA setting, chosen on the command line:
    --gene_pca variance:0.90        keep PCs up to 90% explained variance
    --gene_pca fixed:50             keep exactly 50 PCs (capped at min(N-1, D))
    --cell_pca variance:0.80 fixed:20   several settings -> every combination is run
    --cell_pca none                 DNA-only model (no cell covariates)
Optional caps for variance mode: --gene_pca_max_components / --cell_pca_max_components.

PCA whitening is not offered: rescaling PC columns does not change the column space,
so the F and LRT statistics are identical with or without whitening.

Typical usage
    python gene_assoc_ftest.py run  TRAIT All --gene_tsv genes.tsv --feature_root feats/ \
        --cohort_file cohort.tsv --embed_type all \
        --gene_pca variance:0.9 --cell_embedding cells.npz --cell_pca variance:0.8 \
        --out_dir out/  [--tests F,LRT] [--rank 0 --world_size 8]
    python gene_assoc_ftest.py merge --out_dir out/
"""
import os
for _name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
              'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_name, '2')
import argparse
import hashlib
import json
import sys
import time
from collections import OrderedDict
from pathlib import Path
from urllib.parse import quote

try:  # Load PyTorch before SciPy/sklearn when installed (macOS OpenMP ordering issue).
    import torch
except ImportError:
    torch = None

import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import chi2, f as fdist, norm
from scipy.linalg import eigh as scipy_eigh
from sklearn.decomposition import PCA

VERSION = 'ftest-lrt-1.1'  # unchanged: the speedups give the same results, so old out_dirs resume
KEYS = ['gene_id', 'embed_type', 'gene_pca_id', 'cell_pca_id']


try:
    from tqdm.auto import tqdm
except ImportError:  # progress bars are optional
    tqdm = None


def log(message):
    # tqdm.write keeps log lines from breaking an active progress bar.
    if tqdm is not None:
        tqdm.write(str(message))
    else:
        print(message, flush=True)


def progress(iterable=None, enabled=True, **kwargs):
    """tqdm bar when available and enabled; otherwise the plain iterable."""
    if tqdm is None or not enabled:
        return _NoBar(iterable)
    # In a non-interactive log file, refresh rarely so the log stays readable.
    if not sys.stderr.isatty():
        kwargs.setdefault('mininterval', 30)
    return tqdm(iterable, dynamic_ncols=True, **kwargs)


class _NoBar:
    def __init__(self, iterable=None):
        self.iterable = iterable

    def __iter__(self):
        return iter(self.iterable)

    def update(self, n=1):
        pass

    def set_postfix_str(self, s, refresh=True):
        pass

    def close(self):
        pass


def fmt_seconds(seconds):
    seconds = int(round(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f'{h:d}:{m:02d}:{s:02d}'


# ------------------------------- small utilities -------------------------------
def digest_arrays(*arrays):
    h = hashlib.sha256()
    for array in arrays:
        a = np.ascontiguousarray(array)
        h.update(str((a.shape, a.dtype.str)).encode())
        if a.size:
            h.update(memoryview(a).cast('B'))
    return h.hexdigest()


def atomic_tsv(path, frame):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.tmp.{os.getpid()}')
    frame.to_csv(tmp, sep='\t', index=False)
    os.replace(tmp, path)


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.tmp.{os.getpid()}')
    tmp.write_text(json.dumps(data, indent=2) + '\n')
    os.replace(tmp, path)


def numpy_value(value):
    """Tensor/array -> NumPy. bfloat16/float16 tensors are upcast (NumPy has no bfloat16)."""
    if hasattr(value, 'detach'):
        value = value.detach().cpu()
        if value.is_floating_point() and value.element_size() < 4:
            value = value.float()
        value = value.numpy()
    return np.asarray(value)


def shape_of(value):
    """Shape without converting (and copying) a large tensor."""
    return tuple(value.shape) if hasattr(value, 'shape') else np.shape(value)


def is_numeric(value):
    if hasattr(value, 'detach'):  # torch tensor, including bfloat16
        return not value.is_complex() and value.dtype != torch.bool
    return np.issubdtype(np.asarray(value).dtype, np.number)


def load_dict(path):
    path = Path(path)
    if path.suffix == '.npz':
        with np.load(path, allow_pickle=True) as stored:
            return {k: stored[k] for k in stored.files}
    if path.suffix in ('.pt', '.pth'):
        if torch is None:
            raise ImportError('Reading .pt/.pth files requires PyTorch')
        import inspect
        kwargs = {'map_location': 'cpu'}
        if 'weights_only' in inspect.signature(torch.load).parameters:
            kwargs['weights_only'] = False
        return torch.load(path, **kwargs)
    raise ValueError(f'{path}: expected .npz, .pt or .pth')


def normalize_ids(values, subject=False, allow_duplicates=False):
    values = np.asarray(values).reshape(-1)
    if subject:
        values = [str(v.decode() if isinstance(v, bytes) else v).split('_')[0] for v in values]
    numbers = pd.to_numeric(pd.Series(values), errors='raise')
    if numbers.isna().any() or (numbers % 1 != 0).any():
        raise ValueError('Participant IDs must be integers')
    ids = numbers.to_numpy(dtype=np.int64)
    if not len(ids) or (not allow_duplicates and len(np.unique(ids)) != len(ids)):
        raise ValueError('Empty or duplicated participant IDs')
    return ids


def find_id_key(data):
    key = next((k for k in ('participant_id', 'participant_ids', 'subject_id', 'subject_ids')
                if k in data), None)
    if key is None:
        raise ValueError('File has no participant/subject ID field')
    return key


# ------------------------------------ inputs -----------------------------------
def read_genes(path):
    df = pd.read_csv(path, sep='\t', header=None, dtype=str, keep_default_na=False, comment='#')
    if df.empty:
        raise ValueError('The gene TSV is empty')
    headers = {'gene_id', 'gene_ids', 'geneid', 'gene', 'ensembl_id', 'ensembl_gene_id'}
    first = [str(x).strip().lower() for x in df.iloc[0]]
    hit = [i for i, x in enumerate(first) if x in headers]
    values = df.iloc[1:, hit[0]] if hit else df.iloc[:, 0]
    genes = list(dict.fromkeys(str(x).strip() for x in values if str(x).strip()))
    if not genes or any('/' in g or '\\' in g for g in genes):
        raise ValueError('Expected nonempty gene IDs')
    return genes


def read_phenotype(args):
    frame = pd.read_csv(args.cohort_file, sep='\t')
    cov_names = [c.strip() for c in args.covariates.split(',') if c.strip()]
    if args.label_column in cov_names or len(set(cov_names)) != len(cov_names):
        raise ValueError('Covariates must be unique and must not include the label')
    if args.sex != 'All':
        cov_names = [c for c in cov_names if c != 'sex']
    needed = list(dict.fromkeys(['participant_id', args.label_column, 'sex'] + cov_names))
    missing = set(needed) - set(frame.columns)
    if missing:
        raise ValueError(f'Missing phenotype columns: {sorted(missing)}')
    frame['participant_id'] = normalize_ids(frame.participant_id)
    for c in needed[1:]:
        frame[c] = pd.to_numeric(frame[c], errors='raise')
    keep = frame.sex.isin([0, 1]) if args.sex == 'All' else frame.sex == {'F': 0, 'M': 1}[args.sex]
    frame = frame.loc[keep, needed]
    complete = np.isfinite(frame[needed[1:]].to_numpy(dtype=float)).all(axis=1)
    log(f'[cohort] phenotype rows kept={int(complete.sum())} (dropped {int((~complete).sum())} '
        'with missing label/covariates)')
    frame = frame.loc[complete].set_index('participant_id', drop=False)
    if args.participant_ids_tsv:
        wanted = normalize_ids(pd.read_csv(args.participant_ids_tsv, sep='\t').participant_id)
        absent = set(wanted) - set(frame.index)
        if absent:
            raise ValueError(f'{len(absent)} requested participant IDs lack phenotype/covariates')
        frame = frame.loc[frame.index.isin(wanted)]
    return frame, cov_names


def locate_features(root, gene):
    root = Path(root)
    found = [p for p in (root/gene/f'{gene}.pt', root/gene/f'{gene}.npz',
                         root/f'{gene}.pt', root/f'{gene}.npz') if p.is_file()]
    if len(found) != 1:
        raise ValueError(f'{gene}: expected exactly one .pt/.npz feature file, found {found}')
    return found[0]


TRACKS = ('cage', 'dnase', 'atac', 'rnaseq')


def track_field(track, tss):
    source = 'rna_seq' if track == 'rnaseq' else track
    return f'concat_{source}_' + ('tss500_mean' if tss else 'mean')


def available_embeddings(data, n):
    """Every embedding this file supports.

    Derived: {concat, add} x {gene-body mean, tss} x every nonempty subset of the tracks
    present, e.g. concat_cage, add_tss_cage_dnase_atac_rnaseq. 'add' is included only when
    every chosen track has an even width (two allele blocks).
    Stored: any other participant x feature matrix in the file that is not a raw track field.
    """
    from itertools import combinations
    names = []
    raw_fields = set()
    for tss in (False, True):
        present = []
        for track in TRACKS:
            field = track_field(track, tss)
            if field in data:
                shape = shape_of(data[field])
                if len(shape) == 2 and shape[0] == n and shape[1] > 0:
                    present.append((track, shape[1]))
                    raw_fields.add(field)
        for size in range(1, len(present) + 1):
            for subset in combinations(present, size):
                tracks = '_'.join(t for t, _ in subset)
                prefix = 'tss_' if tss else ''
                names.append(f'concat_{prefix}{tracks}')
                if all(width % 2 == 0 for _, width in subset):
                    names.append(f'add_{prefix}{tracks}')
    for key, value in data.items():
        if key in raw_fields or key in names:
            continue
        if not hasattr(value, 'shape'):
            continue  # lists, strings, metadata
        shape = shape_of(value)
        if len(shape) == 2 and shape[0] == n and shape[1] > 0 and is_numeric(value):
            names.append(key)
    return names


def load_gene(path, embed_types):
    data = load_dict(path)
    id_key = find_id_key(data)
    ids = normalize_ids(numpy_value(data[id_key]), subject=id_key.startswith('subject'))
    if list(embed_types) == ['all']:
        embed_types = available_embeddings(data, len(ids))
        if not embed_types:
            raise ValueError('No usable embeddings in this feature file')
    matrices = {}
    for name in embed_types:
        if name in data:
            matrix = numpy_value(data[name])
        else:  # e.g. concat_cage_dnase, add_tss_atac_rnaseq
            parts = name.split('_')
            mode, tracks = parts[0], parts[1:]
            tss = bool(tracks and tracks[0] == 'tss')
            tracks = tracks[1:] if tss else tracks
            if mode not in ('add', 'concat') or not tracks:
                raise ValueError(f'Unknown embedding {name}')
            blocks = []
            for track in tracks:
                if track not in ('cage', 'dnase', 'atac', 'rnaseq'):
                    raise ValueError(f'Unknown track {track} in {name}')
                source = 'rna_seq' if track == 'rnaseq' else track
                field = f'concat_{source}_' + ('tss500_mean' if tss else 'mean')
                if field not in data:
                    raise ValueError(f'{name} requires feature field {field}')
                block = np.asarray(numpy_value(data[field]), dtype=np.float64)
                if mode == 'add':
                    if block.shape[1] % 2:
                        raise ValueError(f'{field}: cannot split two equal allele blocks')
                    half = block.shape[1] // 2
                    block = block[:, :half] + block[:, half:]
                blocks.append(block)
            matrix = np.concatenate(blocks, axis=1)
        matrix = np.asarray(matrix, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[0] != len(ids) or matrix.shape[1] == 0:
            raise ValueError(f'{name}: expected {len(ids)} x D matrix, got {matrix.shape}')
        matrices[name] = matrix
    return ids, matrices


def load_cell_embedding(path, key=None):
    data = load_dict(path)
    id_key = find_id_key(data)
    key = key or next((k for k in ('cell_embedding', 'cell_embeddings', 'embedding',
                                   'embeddings', 'X') if k in data), None)
    if key not in data:
        raise ValueError('Cell file needs an embedding matrix; use --cell_embedding_key')
    ids = normalize_ids(numpy_value(data[id_key]), subject=True, allow_duplicates=True)
    matrix = np.asarray(numpy_value(data[key]), dtype=np.float64)
    matrix = matrix[:, None] if matrix.ndim == 1 else matrix
    if matrix.ndim != 2 or matrix.shape[0] != len(ids):
        raise ValueError('Cell embedding must be a rows x features matrix aligned to IDs')
    unique, inverse, counts = np.unique(ids, return_inverse=True, return_counts=True)
    if len(unique) != len(ids):  # repeated samples per participant -> mean
        means = np.zeros((len(unique), matrix.shape[1]))
        np.add.at(means, inverse, matrix)
        log(f'[cell] {len(ids)} rows -> {len(unique)} participants (mean of repeats)')
        matrix, ids = means / counts[:, None], unique
    finite = np.isfinite(matrix).all(axis=1)
    if not finite.all():
        log(f'[cell] dropping {int((~finite).sum())} participants with nonfinite cell features')
        matrix, ids = matrix[finite], ids[finite]
    return ids, matrix


# -------------------------------------- PCA ------------------------------------
def parse_pca(spec, max_components=None):
    """'variance:0.9' | 'fixed:50' | 'none' -> config dict (None for 'none')."""
    if spec.lower() == 'none':
        return None
    try:
        mode, value = spec.split(':')
    except ValueError:
        raise ValueError(f'PCA setting "{spec}": use variance:<0-1>, fixed:<int> or none') from None
    if mode == 'variance':
        target = float(value)
        if not 0 < target < 1:
            raise ValueError(f'{spec}: variance target must lie in (0, 1)')
    elif mode == 'fixed':
        target = int(value)
        if target < 1:
            raise ValueError(f'{spec}: need at least one component')
    else:
        raise ValueError(f'{spec}: mode must be variance or fixed')
    ident = f'{mode}_{target:g}' + (f'_max{max_components}' if max_components else '')
    return dict(id=ident, mode=mode, target=target, max_components=max_components)


def standardize(matrix):
    matrix = np.asarray(matrix, dtype=np.float64)
    if not np.isfinite(matrix).all():
        raise ValueError('Nonfinite values in the final cohort')
    scale = matrix.std(axis=0, ddof=1) if len(matrix) > 1 else np.zeros(matrix.shape[1])
    mask = np.isfinite(scale) & (scale > 0) & np.any(matrix != matrix[:1], axis=0)
    return (matrix[:, mask] - matrix[:, mask].mean(axis=0)) / scale[mask], mask


class Spectrum:
    """Standardized matrix and its leading principal components, from the small Gram matrix.

    Equivalent to an exact (full) PCA: the top eigenpairs of Z'Z (or ZZ' when features
    outnumber participants) give the principal axes and squared singular values, and the
    explained-variance ratios use the exact total ||Z||_F^2. Eigenpairs are computed once
    and reused by every PCA setting of the same matrix.
    """

    def __init__(self, matrix):
        self.input_features = matrix.shape[1]
        self.z, self.mask = standardize(matrix)
        n, d = self.z.shape
        self.dual = d > n
        self.gram = self.z @ self.z.T if self.dual else self.z.T @ self.z
        self.total = float(np.einsum('ij,ij->', self.z, self.z))
        self.limit = min(n - 1, d)
        self.values = self.vectors = None  # descending

    def top(self, k):
        m = len(self.gram)
        k = max(1, min(k, m))
        if self.values is None or len(self.values) < k:
            w, v = scipy_eigh(self.gram, subset_by_index=[m - k, m - 1], check_finite=False)
            self.values, self.vectors = np.clip(w[::-1], 0., None), v[:, ::-1]
        return self.values[:k], self.vectors[:, :k]

    def n_for_variance(self, target):
        """Smallest k with cumulative ratio > target (sklearn's rule for float n_components)."""
        guess = 64
        while True:
            values, _ = self.top(guess)
            cumulative = np.cumsum(values) / self.total
            if cumulative[-1] > target or len(values) >= min(len(self.gram), self.limit + 1):
                return int(min(np.searchsorted(cumulative, target, side='right') + 1,
                               len(values)))
            guess *= 4

    def scores(self, k):
        values, vectors = self.top(k)
        if self.dual:
            return vectors * np.sqrt(values)
        return self.z @ vectors


def fit_pca(matrix, config, seed=0, spectrum=None):
    """Standardize features, then PCA by explained-variance target or fixed k.

    variance:<t> uses the exact Gram eigendecomposition (Spectrum), which gives the same
    components as sklearn's svd_solver='full' at a fraction of the cost. fixed:<k> keeps
    the original sklearn call (svd_solver='auto', randomized for large matrices) so its
    results are unchanged.
    """
    spec = spectrum if spectrum is not None else Spectrum(matrix)
    if spec.limit < 1:
        raise ValueError('No nonconstant features in the final cohort')
    if config['mode'] == 'variance':
        k = spec.n_for_variance(config['target'])
        if config['max_components']:
            k = min(k, config['max_components'])
        scores = spec.scores(k)
        explained = float(spec.top(k)[0].sum() / spec.total)
    else:
        k = min(config['target'], spec.limit)
        if config['max_components']:
            k = min(k, config['max_components'])
        if k < config['target']:
            log(f'[WARN PCA] requested {config["target"]} PCs; capped at {k}')
        pca = PCA(n_components=k, svd_solver='auto', random_state=seed)
        scores = pca.fit_transform(spec.z)
        k = scores.shape[1]
        explained = float(pca.explained_variance_ratio_[:k].sum())
    info = dict(n_components=k, explained_variance=explained,
                input_features=spec.input_features, nonconstant_features=int(spec.mask.sum()))
    return scores, info


# -------------------------------- OLS partial F --------------------------------
def fit_ols(design, y, rank_tol=None):
    """SVD least squares with a shared absolute singular-value cutoff."""
    u, s, _ = np.linalg.svd(design, full_matrices=False)
    if rank_tol is None:
        rank_tol = np.finfo(np.float64).eps * max(design.shape) * s[0]
    keep = s > rank_tol
    fitted = u[:, keep] @ (u[:, keep].T @ y)
    residual = y - fitted
    tss = float(np.sum((y - y.mean()) ** 2))
    rss = float(residual @ residual)
    return dict(rank=int(keep.sum()), rss=rss, tss=tss, rank_tol=float(rank_tol),
                r2=1 - rss/tss if tss > 0 else np.nan, y_ss=float(y @ y))


def partial_f(y, reduced_design, full_design):
    full = fit_ols(full_design, y)
    red = fit_ols(reduced_design, y, rank_tol=full['rank_tol'])
    n = len(y)
    df1, df2 = full['rank'] - red['rank'], n - full['rank']
    gain = red['rss'] - full['rss']
    eps = np.finfo(np.float64).eps
    scale = max(full['y_ss'], red['rss'], np.finfo(np.float64).tiny)
    if gain < -128 * eps * scale:
        raise ValueError('Numerical nesting failed')
    gain = max(0., gain)
    tiny = (64 * eps) ** 2 * scale
    F, p, status = np.nan, np.nan, 'ok'
    if full['tss'] == 0:
        status = 'constant_outcome'
    elif df1 <= 0:
        status = 'no_added_rank'
    elif df2 <= 0:
        status = 'no_residual_df'
    elif full['rss'] <= tiny:
        F, p, status = np.inf, 0., 'perfect_full_fit'
    else:
        F = (gain / df1) / (full['rss'] / df2)
        p = float(fdist.sf(F, df1, df2))
    tss = full['tss']
    return dict(F_obs=float(F), p_F=p, z_score=float(norm.isf(p)) if np.isfinite(p) else np.nan,
                df1=df1, df2=df2, rank_red=red['rank'], rank_full=full['rank'],
                r2_red=red['r2'], r2_full=full['r2'],
                delta_r2=gain/tss if tss > 0 else np.nan,
                partial_r2=gain/red['rss'] if red['rss'] > tiny else np.nan,
                cohen_f2=gain/full['rss'] if full['rss'] > tiny else np.nan,
                test_status=status)


# ------------------------------- logistic LRT ----------------------------------
SEPARATION_ETA = 30.  # |linear predictor| beyond this means fitted probability ~0 or ~1


def top_singular_value(matrix):
    """Largest singular value from the small Gram matrix (accurate to ~eps relative)."""
    if matrix.size == 0:
        return 0.
    return float(np.sqrt(max(np.linalg.eigvalsh(matrix.T @ matrix)[-1], 0.)))


def shared_rank_tol(full_design):
    """The absolute singular-value cutoff of fit_ols, without a full SVD of the design."""
    return np.finfo(np.float64).eps * max(full_design.shape) * top_singular_value(full_design)


def reduced_basis(reduced_design, rank_tol):
    u, s, _ = np.linalg.svd(reduced_design, full_matrices=False)
    return u[:, s > rank_tol]


def nested_bases(reduced_design, full_design):
    """Orthonormal bases with one shared absolute cutoff (the same rule as fit_ols).

    q_red spans the reduced design; q_add spans what the full design adds beyond it.
    [q_red, q_add] spans the full design, so its likelihood equals that of the raw design.
    """
    rank_tol = shared_rank_tol(full_design)
    q_red = reduced_basis(reduced_design, rank_tol)
    q_add = added_basis(added_columns(reduced_design, full_design), q_red, rank_tol)
    return q_red, q_add, float(rank_tol)


def added_columns(reduced_design, full_design):
    """The columns of full_design beyond reduced_design when full = [reduced, extra].

    Only a shortcut: the reduced columns have no residual and would be dropped anyway.
    """
    k = reduced_design.shape[1]
    if full_design.shape[1] >= k and np.array_equal(full_design[:, :k], reduced_design):
        return full_design[:, k:]
    return full_design


# A residual block whose singular values all lie within this ratio of the largest is
# orthonormalized from its Gram matrix (two Cholesky-QR-style passes, orthonormal to ~eps);
# anything closer to rank deficiency falls back to the SVD so the rank decision is unchanged.
GRAM_MIN_RATIO = 1e-4


def added_basis(matrix, q_red, rank_tol):
    """Orthonormal basis of the part of `matrix` orthogonal to span(q_red)."""
    residual = matrix - q_red @ (q_red.T @ matrix)
    if residual.shape[1] == 0:
        return np.empty((len(matrix), 0))
    # Columns already inside span(q_red) (e.g. the reduced columns of a full design) leave
    # residuals far below rank_tol. Their total norm is < rank_tol, so dropping them changes
    # no singular value by more than rank_tol and adds none that the SVD would keep.
    norms = np.sqrt(np.einsum('ij,ij->j', residual, residual))
    live = residual[:, norms > rank_tol / np.sqrt(len(norms))]
    if live.shape[1] == 0:
        return np.empty((len(matrix), 0))
    lam, vec = np.linalg.eigh(live.T @ live)                  # ascending
    s = np.sqrt(np.clip(lam, 0., None))
    if s[0] > max(GRAM_MIN_RATIO * s[-1], 1e3 * rank_tol):
        # Clearly full rank (far above rank_tol), so the SVD would keep every column too.
        q = (live @ vec[:, ::-1]) / s[::-1]
        try:  # second pass removes the O(eps * cond^2) loss of orthogonality
            chol = np.linalg.cholesky(q.T @ q)                 # ~ identity
            return q @ np.linalg.inv(chol).T
        except np.linalg.LinAlgError:
            pass
    u, s, _ = np.linalg.svd(residual, full_matrices=False)
    return u[:, s > rank_tol]


def _loglik(y, eta):
    return float(y @ eta - np.logaddexp(0., eta).sum())


def fit_logistic(design, y, start=None, max_iter=100, tol=1e-10):
    """Newton-Raphson with step halving. Converged when the Newton decrement^2/2 < tol."""
    n, p = design.shape
    beta = np.zeros(p) if start is None else np.asarray(start, dtype=np.float64).copy()
    eta = design @ beta
    ll = _loglik(y, eta)
    converged, iterations, decrement = False, 0, np.nan
    for iterations in range(1, max_iter + 1):
        mu = expit(eta)
        grad = design.T @ (y - mu)
        weighted = design * np.sqrt(mu * (1. - mu))[:, None]
        hess = weighted.T @ weighted  # symmetric rank-k update: half the work of X'WX
        try:
            step = np.linalg.solve(hess, grad)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(hess, grad, rcond=None)[0]
        decrement = float(grad @ step)
        if not np.isfinite(decrement):
            break
        if decrement / 2. < tol:
            converged = True
            break
        t, accepted = 1., False
        while t > 1e-10:
            candidate = beta + t * step
            eta_c = design @ candidate
            ll_c = _loglik(y, eta_c)
            if np.isfinite(ll_c) and ll_c >= ll - 1e-12 * max(1., abs(ll)):
                accepted = True
                break
            t /= 2.
        if not accepted:
            break
        beta, eta, ll = candidate, eta_c, ll_c
    separated = bool(np.max(np.abs(eta)) > SEPARATION_ETA) if n else False
    return dict(beta=beta, loglik=ll, converged=converged, iterations=iterations,
                newton_decrement=decrement, separated=separated)


def lrt_from_bases(y, q_red, q_add, reduced_fit=None, max_iter=100, tol=1e-10):
    """Logistic LRT of span(q_red) vs span([q_red, q_add]).

    Columns are scaled by sqrt(n) so X'X = n I (well conditioned Newton steps); this is a
    reparameterization and leaves both log-likelihoods unchanged.
    """
    scale = np.sqrt(len(y))
    if reduced_fit is None:
        reduced_fit = fit_logistic(q_red * scale, y, max_iter=max_iter, tol=tol)
    df = q_add.shape[1]
    out = dict(df_lrt=df, loglik_red=reduced_fit['loglik'],
               lrt_converged_red=reduced_fit['converged'], lrt_iter_red=reduced_fit['iterations'])
    if df == 0:
        return dict(out, LRT_obs=np.nan, p_LRT=np.nan, z_score_LRT=np.nan, loglik_full=np.nan,
                    lrt_converged_full=None, lrt_iter_full=0, mcfadden_r2_increment=np.nan,
                    lrt_status='no_added_rank')
    full_fit = fit_logistic(np.column_stack([q_red, q_add]) * scale, y,
                            start=np.r_[reduced_fit['beta'], np.zeros(df)],
                            max_iter=max_iter, tol=tol)
    statistic = 2. * (full_fit['loglik'] - reduced_fit['loglik'])
    if statistic < -1e-6 * max(1., abs(reduced_fit['loglik'])):
        raise ValueError('Logistic nesting failed: full log-likelihood below reduced')
    statistic = max(0., statistic)
    p = float(chi2.sf(statistic, df))
    if reduced_fit['separated']:
        status = 'reduced_separation'
    elif full_fit['separated']:
        status = 'full_separation'
    elif not (reduced_fit['converged'] and full_fit['converged']):
        status = 'not_converged'
    else:
        status = 'ok'
    ll_red = reduced_fit['loglik']
    return dict(out, LRT_obs=float(statistic), p_LRT=p,
                z_score_LRT=float(norm.isf(p)) if np.isfinite(p) else np.nan,
                loglik_full=full_fit['loglik'], lrt_converged_full=full_fit['converged'],
                lrt_iter_full=full_fit['iterations'],
                mcfadden_r2_increment=1. - full_fit['loglik'] / ll_red if ll_red < 0 else np.nan,
                lrt_status=status)


_REDUCED_CACHE = OrderedDict()  # key -> dict(u, s, fits={(rank, max_iter, tol): fit})
_REDUCED_CACHE_SIZE = 16


def _reduced_entry(key, reduced_design):
    entry = _REDUCED_CACHE.get(key)
    if entry is None:
        u, s, _ = np.linalg.svd(reduced_design, full_matrices=False)
        entry = _REDUCED_CACHE[key] = dict(u=u, s=s, fits={})
        while len(_REDUCED_CACHE) > _REDUCED_CACHE_SIZE:
            _REDUCED_CACHE.popitem(last=False)
    else:
        _REDUCED_CACHE.move_to_end(key)
    return entry


def partial_lrt(y, reduced_design, full_design, max_iter=100, tol=1e-10, cache_key=None):
    """Logistic LRT of the reduced vs full design.

    cache_key: optional hashable that identifies (y, reduced_design). Designs sharing it
    reuse the reduced SVD and the reduced logistic fit, which are identical for every
    embedding and gene-PCA setting of a gene and, with the same cohort, across genes.
    """
    if cache_key is None:
        q_red, q_add, _ = nested_bases(reduced_design, full_design)
        return lrt_from_bases(y, q_red, q_add, max_iter=max_iter, tol=tol)
    rank_tol = shared_rank_tol(full_design)
    entry = _reduced_entry(cache_key, reduced_design)
    keep = entry['s'] > rank_tol
    q_red = entry['u'][:, keep]
    fit_key = (int(keep.sum()), max_iter, tol)
    if fit_key not in entry['fits']:
        entry['fits'][fit_key] = fit_logistic(q_red * np.sqrt(len(y)), y,
                                              max_iter=max_iter, tol=tol)
    q_add = added_basis(added_columns(reduced_design, full_design), q_red, rank_tol)
    return lrt_from_bases(y, q_red, q_add, entry['fits'][fit_key], max_iter, tol)


# ---------------------------------- run / merge --------------------------------
def gene_result_path(root, gene):
    return Path(root)/'per_gene'/f'{quote(gene, safe="")}.tsv'


MANIFEST_KEYS = ('trait', 'sex', 'label_column', 'covariates', 'embed_type', 'gene_pca',
                 'cell_pca', 'gene_pca_max_components', 'cell_pca_max_components',
                 'cell_embedding', 'cell_embedding_key', 'cohort_file', 'participant_ids_tsv',
                 'feature_root', 'seed', 'tests', 'lrt_max_iter', 'lrt_tol')


def manifest_for(args, genes, version=VERSION, extra_keys=()):
    keys = MANIFEST_KEYS + tuple(extra_keys)
    return dict(version=version, genes=genes, **{k: getattr(args, k, None) for k in keys})


def ensure_manifest(root, manifest):
    path = Path(root)/'run_manifest.json'
    if path.exists():
        saved = json.loads(path.read_text())
        if saved != manifest:
            changed = sorted(k for k in set(saved) | set(manifest) if saved.get(k) != manifest.get(k))
            raise ValueError(f'out_dir holds a different run (changed: {changed}); use a new --out_dir')
    else:
        atomic_json(path, manifest)


def parse_tests(spec):
    tests = [t.strip().upper() for t in spec.split(',') if t.strip()]
    if not tests or len(set(tests)) != len(tests) or any(t not in ('F', 'LRT') for t in tests):
        raise ValueError('--tests must be a comma-separated subset of F,LRT')
    return tests


# ------------------------- shared cohort / design builders ----------------------
# Used by cmd_run below and by gene_assoc_permutation.py, so both scripts test exactly
# the same cohort, covariates, PCA scores and nested designs.
def build_context(args):
    """Parse the PCA settings and load inputs shared by every gene."""
    embeds = list(dict.fromkeys(e.strip() for e in args.embed_type.split(',') if e.strip()))
    gene_cfgs = [parse_pca(s, args.gene_pca_max_components) for s in args.gene_pca]
    if None in gene_cfgs:
        raise ValueError('--gene_pca cannot be none')
    cell_cfgs = [parse_pca(s, args.cell_pca_max_components) for s in args.cell_pca]
    if any(c is not None for c in cell_cfgs) and not args.cell_embedding:
        raise ValueError('--cell_pca needs --cell_embedding (or use --cell_pca none)')
    if args.cell_embedding and all(c is None for c in cell_cfgs):
        log('[WARN] --cell_embedding given but --cell_pca none: running DNA-only')
    for group in (gene_cfgs, [c for c in cell_cfgs if c]):
        if len({c['id'] for c in group}) != len(group):
            raise ValueError('Duplicate PCA settings')
    phenotype, cov_names = read_phenotype(args)
    cell_ids = cell_matrix = None
    if args.cell_embedding and any(cell_cfgs):
        cell_ids, cell_matrix = load_cell_embedding(args.cell_embedding, args.cell_embedding_key)
        phenotype = phenotype.loc[phenotype.index.isin(set(cell_ids.tolist()))]
        log(f'[cohort] phenotype ∩ cell = {len(phenotype)}')
    cell_pos = {int(p): i for i, p in enumerate(cell_ids)} if cell_ids is not None else None
    return dict(embeds=embeds, gene_cfgs=gene_cfgs, cell_cfgs=cell_cfgs, phenotype=phenotype,
                cov_names=cov_names, cell_matrix=cell_matrix, cell_pos=cell_pos,
                cell_cache={})  # (cohort hash, cell cfg id) -> (scores, info)


def prepare_gene(ctx, args, gene):
    """Load one gene's embeddings and build its cohort, outcome and base design."""
    ids, matrices = load_gene(locate_features(args.feature_root, gene), ctx['embeds'])
    frame = ctx['phenotype'].loc[ctx['phenotype'].index.isin(set(ids.tolist()))]
    if len(frame) < 3:
        raise ValueError('Fewer than 3 participants after intersection')
    if args.expected_n_samples is not None and len(frame) != args.expected_n_samples:
        raise ValueError(f'Final N={len(frame)} != expected {args.expected_n_samples}')
    keep_ids = frame.participant_id.to_numpy(np.int64)
    y = frame[args.label_column].to_numpy(np.float64)
    cov_names = ctx['cov_names']
    cov = standardize(frame[cov_names].to_numpy(np.float64))[0] if cov_names else \
        np.empty((len(frame), 0))
    gpos = {int(p): i for i, p in enumerate(ids)}
    return dict(matrices=matrices, n=len(frame), keep_ids=keep_ids,
                cohort_hash=digest_arrays(keep_ids), y=y,
                base=np.column_stack([np.ones(len(frame)), cov]),
                gindex=[gpos[int(p)] for p in keep_ids],
                is_binary=bool(np.array_equal(np.unique(y), [0., 1.])))


def failed_rows(ctx, gene, error):
    return [dict(gene_id=gene, embed_type=e, gene_pca_id=g['id'],
                 cell_pca_id=c['id'] if c else 'none', status='input_failed', error=str(error))
            for e in ctx['embeds'] for g in ctx['gene_cfgs'] for c in ctx['cell_cfgs']]


def iter_designs(ctx, args, gene, prep, wanted=None):
    """Yield (row, reduced, scores, error) for every cell PCA x embedding x gene PCA setting.

    wanted: optional set of (embed_type, gene_pca_id, cell_pca_id) to restrict the output.
    scores is None (and error set) when the gene PCA fails.
    """
    gene_pca_cache = {}  # (embed, gene cfg) -> (scores, info); shared by all cell settings
    keep_ids = prep['keep_ids']
    for cell_cfg in ctx['cell_cfgs']:
        cell_id = cell_cfg['id'] if cell_cfg else 'none'
        if wanted is not None and not any(w[2] == cell_id for w in wanted):
            continue
        if cell_cfg is None:
            reduced, cell_info = prep['base'], dict(n_components=0, explained_variance=None)
        else:
            key = (prep['cohort_hash'], cell_cfg['id'])
            if key not in ctx['cell_cache']:
                selected = ctx['cell_matrix'][[ctx['cell_pos'][int(p)] for p in keep_ids]]
                ctx['cell_cache'][key] = fit_pca(selected, cell_cfg, args.seed)
                log(f'[cell PCA] {cell_cfg["id"]}: N={len(keep_ids)}, '
                    f'k={ctx["cell_cache"][key][1]["n_components"]}, '
                    f'var={ctx["cell_cache"][key][1]["explained_variance"]:.4f}')
            cell_scores, cell_info = ctx['cell_cache'][key]
            reduced = np.column_stack([prep['base'], cell_scores])
        for embed, full_matrix in prep['matrices'].items():
            spectrum = None  # shared by every gene PCA setting of this embedding
            for gene_cfg in ctx['gene_cfgs']:
                if wanted is not None and (embed, gene_cfg['id'], cell_id) not in wanted:
                    continue
                y = prep['y']
                row = dict(gene_id=gene, embed_type=embed, gene_pca_id=gene_cfg['id'],
                           cell_pca_id=cell_id, model='dna_cell' if cell_cfg else 'dna_only',
                           trait=args.trait, sex=args.sex, n_samples=prep['n'],
                           is_binary=prep['is_binary'],
                           n_cases=int((y == 1).sum()) if prep['is_binary'] else None,
                           participant_ids_sha256=prep['cohort_hash'],
                           cell_pca_components=cell_info['n_components'],
                           cell_pca_explained_variance=cell_info['explained_variance'])
                try:
                    pca_key = (embed, gene_cfg['id'])
                    if pca_key not in gene_pca_cache:
                        if spectrum is None:
                            spectrum = Spectrum(full_matrix[prep['gindex']])
                        gene_pca_cache[pca_key] = fit_pca(None, gene_cfg,
                                                          args.seed, spectrum)
                    scores, info = gene_pca_cache[pca_key]
                except (ValueError, np.linalg.LinAlgError) as error:
                    yield row, reduced, None, error
                    continue
                row.update(gene_pca_components=info['n_components'],
                           gene_pca_explained_variance=info['explained_variance'],
                           gene_input_features=info['input_features'])
                yield row, reduced, scores, None


def analytic_tests(row, y, reduced, scores, tests, is_binary, args):
    """Fill one result row with the requested analytic F and/or LRT results."""
    full = np.column_stack([reduced, scores])
    if 'F' in tests:
        try:
            row.update(partial_f(y, reduced, full))
            row['status'] = row['test_status']
        except (ValueError, np.linalg.LinAlgError) as error:
            row.update(status='failed', error=str(error))
    if 'LRT' in tests:
        if not is_binary:
            row['lrt_status'] = 'not_binary'
        else:
            try:
                # y and the reduced design are fixed by the cohort and the cell PCA setting.
                cache_key = (row.get('participant_ids_sha256'), row.get('cell_pca_id'),
                             reduced.shape, digest_arrays(y))
                row.update(partial_lrt(y, reduced, full, args.lrt_max_iter, args.lrt_tol,
                                       cache_key=cache_key))
            except (ValueError, np.linalg.LinAlgError) as error:
                row.update(lrt_status='failed', lrt_error=str(error))
        if 'F' not in tests:
            row['status'] = row['lrt_status']
    return row


def _fmt(value):
    return f'{value:.3g}' if value is not None and np.isfinite(value) else 'nan'


def cmd_run(args):
    if not 0 <= args.rank < args.world_size:
        raise ValueError('Require 0 <= rank < world_size')
    tests = parse_tests(args.tests)
    genes = read_genes(args.gene_tsv)
    ctx = build_context(args)
    ensure_manifest(args.out_dir, manifest_for(args, genes))
    if 'LRT' in tests:
        labels = np.unique(ctx['phenotype'][args.label_column].to_numpy(np.float64))
        if not np.array_equal(labels, [0., 1.]):
            log(f'[WARN] label is not binary 0/1 (values like {labels[:5]}); '
                'LRT rows will have lrt_status=not_binary')

    assigned = genes[args.rank::args.world_size]
    log(f'[rank {args.rank}/{args.world_size}] {len(assigned)} genes; tests={tests}; '
        f'embeddings={ctx["embeds"]}; gene PCA={[c["id"] for c in ctx["gene_cfgs"]]}; '
        f'cell PCA={[c["id"] if c else "none" for c in ctx["cell_cfgs"]]}')
    run_start = time.monotonic()
    n_done = n_skipped = 0
    gene_bar = progress(assigned, enabled=not args.no_progress, desc=f'rank {args.rank} genes',
                        unit='gene', position=0)
    for number, gene in enumerate(gene_bar, 1):
        gene_bar.set_postfix_str(gene)
        out = gene_result_path(args.out_dir, gene)
        if out.exists() and not args.overwrite:
            n_skipped += 1
            if args.verbose:
                log(f'[skip] {gene}: {out.name} exists')
            continue
        gene_start = time.monotonic()
        try:
            prep = prepare_gene(ctx, args, gene)
        except (ValueError, KeyError) as error:
            log(f'[FAILED] {gene}: {error}')
            atomic_tsv(out, pd.DataFrame(failed_rows(ctx, gene, error)))
            continue
        if ctx['embeds'] == ['all'] and (args.verbose or number == 1):
            log(f'[embeddings] {gene}: {len(prep["matrices"])} types -> '
                f'{", ".join(prep["matrices"])}')

        rows = []
        test_bar = progress(total=len(prep['matrices']) * len(ctx['gene_cfgs']) *
                            len(ctx['cell_cfgs']), enabled=not args.no_progress,
                            desc=f'  {gene} tests', unit='test', leave=False, position=1)
        for row, reduced, scores, error in iter_designs(ctx, args, gene, prep):
            if error is not None:
                row.update(status='failed', error=str(error))
            else:
                analytic_tests(row, prep['y'], reduced, scores, tests, prep['is_binary'], args)
            rows.append(row)
            test_bar.update(1)
            if args.verbose:
                log(f'[{number}/{len(assigned)}] {gene} {row["embed_type"]} '
                    f'gene={row["gene_pca_id"]} cell={row["cell_pca_id"]}: '
                    f'k={row.get("gene_pca_components")} F={_fmt(row.get("F_obs"))} '
                    f'p_F={_fmt(row.get("p_F"))} LRT={_fmt(row.get("LRT_obs"))} '
                    f'p_LRT={_fmt(row.get("p_LRT"))} status={row.get("status")} '
                    f'lrt_status={row.get("lrt_status")}')
        test_bar.close()
        atomic_tsv(out, pd.DataFrame(rows))
        n_done += 1
        frame = pd.DataFrame(rows)
        mins = ', '.join(f'min {c}={_fmt(pd.to_numeric(frame[c], errors="coerce").min())}'
                         for c in ('p_F', 'p_LRT') if c in frame)
        n_ok = sum(r.get('status') in ('ok', 'perfect_full_fit') for r in rows)
        elapsed = time.monotonic() - run_start
        eta = elapsed / n_done * (len(assigned) - number)  # skipped genes cost ~0
        log(f'[{number}/{len(assigned)}] {gene}: {len(rows)} tests ({n_ok} ok), {mins}, '
            f'gene time {fmt_seconds(time.monotonic() - gene_start)}, '
            f'elapsed {fmt_seconds(elapsed)}, ETA {fmt_seconds(eta)}')
    gene_bar.close()
    log(f'Rank {args.rank} finished in {fmt_seconds(time.monotonic() - run_start)}: '
        f'{n_done} genes computed, {n_skipped} skipped (already done). '
        'Run "merge" after all ranks finish.')


def bh_fdr(p):
    p = np.asarray(p, dtype=float)
    q = np.full(len(p), np.nan)
    ok = np.isfinite(p)
    if ok.any():
        order = np.argsort(p[ok])
        ranked = p[ok][order] * ok.sum() / np.arange(1, ok.sum() + 1)
        ranked = np.minimum.accumulate(ranked[::-1])[::-1]
        values = np.empty(ok.sum())
        values[order] = np.minimum(ranked, 1)
        q[ok] = values
    return q


def merge_results(root, out_name, p_columns, allow_partial=False):
    """Concatenate per-gene TSVs and add BH q-values for each p column, within each
    (embedding, gene PCA, cell PCA) configuration across genes."""
    root = Path(root)
    manifest = json.loads((root/'run_manifest.json').read_text())
    frames, missing = [], []
    for gene in manifest['genes']:
        path = gene_result_path(root, gene)
        if path.exists():
            frames.append(pd.read_csv(path, sep='\t'))
        else:
            missing.append(gene)
    if missing and not allow_partial:
        raise ValueError(f'{len(missing)} gene(s) unfinished, e.g. {missing[:5]}; '
                         'finish all ranks or pass --allow_partial_merge')
    if not frames:
        raise ValueError('No per-gene results found')
    frame = pd.concat(frames, ignore_index=True)
    if frame.duplicated(KEYS).any():
        raise ValueError('Duplicated result keys')
    for column in p_columns:
        if column not in frame:
            continue
        q_column = 'q_' + column[2:] + '_bh' if column.startswith('p_') else column + '_q_bh'
        frame[q_column] = np.nan
        for _, idx in frame.groupby(['embed_type', 'gene_pca_id', 'cell_pca_id']).groups.items():
            frame.loc[idx, q_column] = bh_fdr(pd.to_numeric(frame.loc[idx, column], errors='coerce'))
    atomic_tsv(root/out_name, frame)
    atomic_tsv(root/'missing_genes.tsv', pd.DataFrame({'gene_id': missing}))
    return frame, frames, missing


def cmd_merge(args):
    frame, frames, missing = merge_results(args.out_dir, 'ftest_results_merged.tsv',
                                           ('p_F', 'p_LRT'), args.allow_partial_merge)
    ok = frame.status.isin(['ok', 'perfect_full_fit']).sum() if 'status' in frame else 0
    lrt_ok = (frame.lrt_status == 'ok').sum() if 'lrt_status' in frame else 0
    log(f'Merged {len(frame)} rows from {len(frames)} genes ({ok} F testable, {lrt_ok} LRT ok); '
        f'missing genes={len(missing)} -> {Path(args.out_dir)/"ftest_results_merged.tsv"}')


def add_data_args(run):
    """Cohort, embedding and PCA arguments shared with gene_assoc_permutation.py."""
    run.add_argument('trait')
    run.add_argument('sex', choices=['M', 'F', 'All'], help='Cohort coding F=0, M=1')
    run.add_argument('--gene_tsv', '--sanity_check_gene_tsv', dest='gene_tsv', required=True)
    run.add_argument('--feature_root', required=True)
    run.add_argument('--cohort_file', required=True)
    run.add_argument('--participant_ids_tsv')
    run.add_argument('--label_column', default='label')
    run.add_argument('--covariates', default='age,sex')
    run.add_argument('--embed_type', default='all',
                     help='"all" (default): every stored matrix and every concat/add x mean/tss x '
                          'track-subset combination the feature file supports; or comma-separated keys')
    run.add_argument('--gene_pca', nargs='+', default=['variance:0.9'],
                     help='One or more gene-embedding PCA settings: variance:0.9 fixed:50 ...')
    run.add_argument('--gene_pca_max_components', type=int,
                     help='Optional cap on gene PCs (applies to every --gene_pca setting)')
    run.add_argument('--cell_embedding', '--cell1', help='Optional .npz/.pt cell embeddings')
    run.add_argument('--cell_embedding_key')
    run.add_argument('--cell_pca', nargs='+', default=['variance:0.9'],
                     help='One or more cell PCA settings: variance:0.8 fixed:20 none')
    run.add_argument('--cell_pca_max_components', type=int,
                     help='Optional cap on cell PCs (applies to every --cell_pca setting)')
    run.add_argument('--expected_n_samples', type=int)
    run.add_argument('--seed', type=int, default=0, help='Randomized PCA solvers (and permutations)')
    run.add_argument('--tests', default='F,LRT',
                     help='Comma-separated subset of F,LRT (LRT only applies to binary 0/1 labels)')
    run.add_argument('--lrt_max_iter', type=int, default=100, help='Newton iterations per logistic fit')
    run.add_argument('--lrt_tol', type=float, default=1e-10,
                     help='Logistic convergence: Newton decrement^2/2 below this value')
    run.add_argument('--out_dir', required=True)
    run.add_argument('--overwrite', action='store_true', help='Recompute genes that already have results')
    run.add_argument('--rank', type=int, default=0, help='Split genes across workers')
    run.add_argument('--world_size', type=int, default=1)
    run.add_argument('--no_progress', action='store_true', help='Disable tqdm progress bars')
    run.add_argument('--verbose', action='store_true',
                     help='Log every single test and skipped gene (default: one summary line per gene)')


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    run = sub.add_parser('run', help='PCA + analytic partial F test and logistic LRT per gene')
    add_data_args(run)
    run.set_defaults(func=cmd_run)
    merge = sub.add_parser('merge', help='Combine per-gene results and add BH q-values')
    merge.add_argument('--out_dir', required=True)
    merge.add_argument('--allow_partial_merge', action='store_true')
    merge.set_defaults(func=cmd_merge)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.command == 'run' and not args.cell_embedding:
        args.cell_pca = ['none']
    args.func(args)


if __name__ == '__main__':
    main()

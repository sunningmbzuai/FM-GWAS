#!/usr/bin/env python
r"""Bring the staged insample_hpp_AG.py up to FM-GWAS commit d17eae23a6.

Ning's 2026-10-03 commit ("Refactor covariate handling and PCA usage") is the
only change upstream since the July copy we stage, and it touches this one
file. Rather than re-staging her repo over our patched copy, this replays the
same five hunks, line for line:

  1. Covariates go into OLS raw. The July code ran PCA(0.99) on [age, sex],
     which kept only PC1 (~age, var(age) >> var(sex)) and so dropped sex from
     the sex=All cells. Shape and constant-column checks are added.
  2. Gene PCA keeps 95% of the variance for every embedding type, replacing
     the fixed 5 (add_*) / 50 (others) components. Zero-variance directions
     never count toward 95%, so this also subsumes our rank cap
     (rank_capped_pca.py passes float requests through untouched).
  3. A gene is skipped only when every feature is constant, not when it has
     fewer features than the old fixed component count.
  4. svd_solver='auto' is explicit, and a PCA failure is logged, not swallowed.
  5. The F-test uses the raw covariates.

Result rows are labelled <embed_type>_pca0.95 (was _pca50 / _pca5).

Our dedup does not change what PCA sees: expand_embeddings.py gathers every
gene back to one row per participant before export_features.py writes the
table, so the 95% is taken over the same person-level matrix as hers.

Idempotent; fails loudly if the staged file is not the July text. Backs the
original up to <name>.orig on first run (kept if an earlier patch made it).

Usage:
    python fix_assoc_pca95.py <FM-GWAS checkout>/insample_hpp_AG.py
"""
import argparse
import pathlib

REPLACEMENTS = [
    # 1. Raw covariates instead of PCA(0.99) on them.
    (
        "        Z = data['covariants']\n"
        "        labels = data['labels']\n"
        "        # association test\n"
        "        pca = PCA(n_components=0.99, svd_solver='full', whiten=True)\n"
        "        Z_pca = pca.fit_transform(Z)\n"
        "        # >> Ning: fix sign\n"
        "        Z_pca = fix_signs(Z_pca)\n",
        "        Z = np.asarray(data['covariants'], dtype=np.float64)\n"
        "        labels = data['labels']\n"
        "        # Covariates enter OLS directly (no PCA). Previously PCA(0.99) on raw [age, sex]\n"
        "        # kept only PC1 (~age, since var(age) >> var(sex)), silently dropping sex for sex=All.\n"
        "        expected_cov = ['age', 'sex'] if args.sex.upper() == 'ALL' else ['age']\n"
        "        if Z.ndim != 2 or Z.shape[1] != len(expected_cov):\n"
        "            raise ValueError(f'Expected covariates {expected_cov}, got shape {Z.shape}')\n"
        "        cov_std = Z.std(axis=0)\n"
        "        if np.any(cov_std == 0):\n"
        "            raise ValueError(f'Constant covariate(s) {[c for c, s in zip(expected_cov, cov_std) if s == 0]} '\n"
        "                             f'for sex={args.sex}')\n",
    ),
    # 2. 95% variance for every embedding type.
    (
        "            pca_component_list = [5] if 'add' in embed_type else [50]\n",
        "            # Gene PCA: keep 95% variance, whitened, sklearn 'auto' solver (same config for all embed types)\n"
        "            pca_component_list = [0.95]\n",
    ),
    # 3 + 4a. Skip only all-constant genes; explicit solver.
    (
        "                if X.shape[-1] < pca_component:\n"
        "                    break\n"
        "                pca_final = PCA(n_components=pca_component, whiten=True)\n",
        "                if X.shape[-1] == 0:\n"
        "                    print(f'{embed_type}: all features constant, skip')\n"
        "                    break\n"
        "                pca_final = PCA(n_components=pca_component, whiten=True, svd_solver='auto')\n",
    ),
    # 4b. Say why a PCA failed.
    (
        "                except:\n"
        "                    continue\n",
        "                except Exception as e:\n"
        "                    print(f'{gene_id} {embed_type}: PCA failed ({e}), skip')\n"
        "                    continue\n",
    ),
    # 5. F-test on the raw covariates.
    (
        "ret = LinearRegression_Ftest(X, Z_pca, Y)",
        "ret = LinearRegression_Ftest(X, Z, Y)",
    ),
]


def apply_replacements(source: str, replacements) -> tuple[str, int]:
    """(patched source, hunks applied). Already-applied hunks are skipped."""
    patched = source
    applied = 0
    for old, new in replacements:
        if new in patched:
            continue
        if old not in patched:
            raise SystemExit(
                "expected code not found -- inspect by hand:\n  %s" % old)
        patched = patched.replace(old, new, 1)
        applied += 1
    return patched, applied


def patch_file(target: pathlib.Path, replacements, dry_run: bool = False) -> int:
    source = target.read_text()
    patched, applied = apply_replacements(source, replacements)
    if applied == 0:
        print("already patched: %s" % target.name)
        return 0
    if dry_run:
        print("would apply %d replacement(s) to %s" % (applied, target.name))
        return applied
    backup = target.with_suffix(target.suffix + ".orig")
    if not backup.exists():
        backup.write_text(source)
        print("backup -> %s" % backup.name)
    target.write_text(patched)
    print("patched %s (%d replacement(s))" % (target.name, applied))
    return applied


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("script")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    target = pathlib.Path(args.script)
    if not target.is_file():
        raise SystemExit("not a file: %s" % target)
    patch_file(target, REPLACEMENTS, dry_run=args.dry_run)


if __name__ == "__main__":
    main()

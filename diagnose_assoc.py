#!/usr/bin/env python
"""Rank and permutation checks for one gene's association, on real data.

Answers whether a significant gene at the thin tail of the variant
distribution (few distinct sequences) is a real hit. Either the thin genes
are genuinely easier to test -- two haplotypes reduce the 50-component test to
a single-variant contrast, which is the regime GWAS was designed for -- or
their p-values are not calibrated. This prints both diagnostics per view so the
answer comes from the data.

    python diagnose_assoc.py --gene ENSG00000174327.6 --trait Cholesterol --sex All
    python diagnose_assoc.py --gene ENSG00000161681.11 --trait VAT --sex M --permutations 200

Read the output this way: when p_perm is within an order of magnitude or two of
p_analytic, the analytic test is calibrated and the hit is real. When p_analytic
is tiny and p_perm sits near 1/(B+1) that is agreement too, since permutation
cannot resolve below that floor. Only p_analytic tiny with p_perm large is a
broken test.

numpy only, so it runs in the microbiome env alongside the assoc stage.
"""
import argparse
import os

import numpy as np

import assoc_diagnostics as D
import config
import feature_tables

LABEL_COLUMN = "label"
ID_COLUMN = "participant_id"
COVARIATE_COLUMNS = ("sex", "age")


def read_cohort(path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(participant ids, covariates, label) from a build_cohorts.py TSV."""
    with open(path) as handle:
        header = handle.readline().rstrip("\n").split("\t")
        rows = [line.rstrip("\n").split("\t") for line in handle if line.strip()]
    index = {name: i for i, name in enumerate(header)}
    for needed in (ID_COLUMN, LABEL_COLUMN) + COVARIATE_COLUMNS:
        if needed not in index:
            raise KeyError(f"{path} has no {needed} column: {header}")

    def column(name):
        return np.array([row[index[name]] for row in rows])

    ids = column(ID_COLUMN)
    label = column(LABEL_COLUMN).astype(np.float64)
    covariates = np.column_stack([column(c).astype(np.float64)
                                  for c in COVARIATE_COLUMNS])
    usable = np.isfinite(label) & np.all(np.isfinite(covariates), axis=1)
    return ids[usable], covariates[usable], label[usable]


def align(subject_ids: np.ndarray, cohort_ids: np.ndarray) -> tuple:
    """Row indices matching a gene table's subjects to the cohort's.

    The table's subject ids carry the "<participant>_<...>" spelling the .pt
    files used, so the participant is everything before the first underscore.
    """
    table_participants = np.array([s.split("_")[0] for s in subject_ids])
    position = {p: i for i, p in enumerate(table_participants)}
    keep_cohort, keep_table = [], []
    for i, participant in enumerate(cohort_ids):
        row = position.get(str(participant))
        if row is not None:
            keep_cohort.append(i)
            keep_table.append(row)
    return np.array(keep_cohort, dtype=int), np.array(keep_table, dtype=int)


def diagnose_view(name: str, matrix: np.ndarray, covariates: np.ndarray,
                  label: np.ndarray, n_permutations: int, seed: int) -> dict:
    components = D.pca_features(matrix)
    result = D.permutation_p(label, covariates, components,
                             n_permutations=n_permutations, seed=seed)
    result.update({
        "view": name,
        "n_columns": int(matrix.shape[1]),
        "distinct_rows": D.distinct_rows(matrix),
        "rank": D.design_rank(matrix),
        "components_kept": int(components.shape[1]),
    })
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gene", required=True, help="versioned gene id")
    parser.add_argument("--trait", required=True)
    parser.add_argument("--sex", required=True, choices=("All", "F", "M"))
    parser.add_argument("--permutations", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--views", nargs="*", default=None,
                        help="default: every view in the table")
    parser.add_argument("--row_order", action="store_true",
                        help="also rerun the pipeline's own recipe "
                             "(normalise, PCA whiten=True, QR F-test) in the "
                             "original and in shuffled participant orders")
    parser.add_argument("--orders", type=int, default=5,
                        help="shuffled orders to try with --row_order")
    parser.add_argument("--feature_root", default=None)
    parser.add_argument("--cohort_file", default=None)
    args = parser.parse_args()

    feature_root = args.feature_root or config.FEATURE_TABLE_ROOT
    cohort_file = args.cohort_file or os.path.join(
        config.COHORT_DIR, f"{args.trait}_{args.sex}.tsv")

    subject_ids, views = feature_tables.load_gene(args.gene, feature_root)
    cohort_ids, covariates, label = read_cohort(cohort_file)
    keep_cohort, keep_table = align(subject_ids, cohort_ids)
    if len(keep_cohort) == 0:
        raise SystemExit(f"{args.gene}: no participants shared with {cohort_file}")

    covariates, label = covariates[keep_cohort], label[keep_cohort]
    print(f"{args.gene}  {args.trait} {args.sex}")
    print(f"  participants matched : {len(keep_cohort):,}")
    print(f"  permutations         : {args.permutations}"
          f"  (smallest reportable p = {1 / (args.permutations + 1):.2G})")
    print()
    header = (f"{'view':<20}{'cols':>6}{'distinct':>10}{'rank':>6}{'PCs':>5}"
              f"{'F':>12}{'p_analytic':>13}{'p_perm':>10}")
    print(header)
    print("-" * len(header))

    for name in sorted(args.views or views):
        matrix = views[name][keep_table]
        result = diagnose_view(name, matrix, covariates, label,
                               args.permutations, args.seed)
        print(f"{result['view']:<20}{result['n_columns']:>6}"
              f"{result['distinct_rows']:>10}{result['rank']:>6}"
              f"{result['components_kept']:>5}{result['f_observed']:>12.3f}"
              f"{result['p_analytic']:>13.3G}{result['p_permutation']:>10.4f}")

    if args.row_order:
        print_row_order(views, keep_table, covariates, label, args)


def print_row_order(views: dict, keep_table: np.ndarray,
                    covariates: np.ndarray, label: np.ndarray,
                    args: argparse.Namespace) -> None:
    """The pipeline's recipe, original participant order against shuffled.

    Components follow insample_hpp_AG.py: 5 for add_* views, 50 otherwise.
    A valid test gives the same p in every order; the pipeline's moving by
    orders of magnitude is the defect, shown on this gene's own data.
    """
    print()
    print("Pipeline recipe (normalise, PCA whiten=True, QR F-test) by row order")
    header = f"{'view':<20}{'PCs':>5}{'p_original':>13}  p_shuffled_orders"
    print(header)
    print("-" * (len(header) + 30))
    y = (label - label.mean()) / label.std()
    for name in sorted(args.views or views):
        n_components = 5 if name.startswith("add") else 50
        matrix = views[name][keep_table]
        if matrix.shape[1] < n_components:
            continue
        result = D.row_order_sensitivity(matrix, covariates, y, n_components,
                                         n_orders=args.orders, seed=args.seed)
        shuffled = "  ".join(f"{p:.2G}" for p in result["p_shuffled_orders"])
        print(f"{name:<20}{n_components:>5}"
              f"{result['p_original_order']:>13.3G}  {shuffled}")


if __name__ == "__main__":
    main()

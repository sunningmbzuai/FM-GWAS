"""Build the replication target list from the HPP novel-association exports.

Ning's HPP exports are the upstream source of truth. The current one is
novel_genes_1006.tsv (2026-10-06, novel Height and Cholesterol hits from the
full HPP cohort, pooled sexes). Earlier exports are kept as named lists:
novel_genes_1003.tsv (the hits that survive the 95%-variance PCA update) and
the July pair, novel_genes.tsv and novel_genes_Cholesterol.tsv.
All are already thresholded upstream -- nothing here re-tests significance.

Only gene_id, trait and sex are required. The p-value columns are optional:
the 1006 export carries none, so its rows are unranked and keep file order,
after any ranked rows.

hpp_top_novel_genes.tsv, which config.TARGET_GENES_TSV points at, is the list
the pipeline actually runs: every source association, ranked by
p_F_analytic_min. --max-per-cell and --max-rows cut it down to a shortlist
instead -- the caps exist because one saturated cell would otherwise crowd out every other trait in a short list. Uncapped is
the default: the whole point of the exports is to replicate all of them, and
the per-gene cost is one embedding pass.

The exports and the target list are data and are not committed; place them
next to this script. Rerunning it reproduces the target list exactly, so a
refreshed export is a one-command update.
"""
import argparse
import csv
import os

GENE_ID_COLUMN = "gene_id"
TRAIT_COLUMN = "trait"
SEX_COLUMN = "sex"
P_MIN_COLUMN = "p_F_analytic_min"
P_MAX_COLUMN = "p_F_analytic_max"
COLUMNS = [GENE_ID_COLUMN, P_MIN_COLUMN, P_MAX_COLUMN, "n_rows",
           TRAIT_COLUMN, SEX_COLUMN]
REQUIRED_COLUMNS = [GENE_ID_COLUMN, TRAIT_COLUMN, SEX_COLUMN]
P_COLUMNS = (P_MIN_COLUMN, P_MAX_COLUMN)

# None = no cap. Shortlist runs pass --max-per-cell 4 --max-rows 18, which is
# what the first (pre-full-export) target list was.
MAX_PER_CELL = None   # per (trait, sex)
MAX_ROWS = None       # total genes the replication run covers

REPO_DIR = os.path.dirname(os.path.abspath(__file__))
# Current HPP export (2026-10-06): novel Height/Cholesterol hits in the full
# HPP cohort, sex "All", no p-values. Supersedes the 2026-10-03 list.
SOURCE_LISTS = [os.path.join(REPO_DIR, "novel_genes_1006.tsv")]
# 2026-10-03 export: the associations still significant in HPP after
# FM-GWAS d17eae23a6 (95%-variance PCA, raw covariates).
OCT03_SOURCE_LISTS = [os.path.join(REPO_DIR, "novel_genes_1003.tsv")]
JULY_SOURCE_LISTS = [os.path.join(REPO_DIR, "novel_genes.tsv"),
                     os.path.join(REPO_DIR, "novel_genes_Cholesterol.tsv")]
OUTPUT_TSV = os.path.join(REPO_DIR, "hpp_top_novel_genes.tsv")


def read_source(path: str) -> list[dict]:
    """One export file as rows. Delimiter is sniffed: the exports mix TSV/CSV.

    utf-8-sig because at least one export carries a BOM, which would otherwise
    ride along in the first column name and break the column check.
    """
    with open(path, newline="", encoding="utf-8-sig") as handle:
        sample = handle.readline()
        handle.seek(0)
        delimiter = "," if sample.count(",") > sample.count("\t") else "\t"
        rows = list(csv.DictReader(handle, delimiter=delimiter))
    if rows:
        missing = set(REQUIRED_COLUMNS) - set(rows[0])
        if missing:
            raise KeyError(f"{path} is missing column(s) {sorted(missing)}: "
                           f"{list(rows[0])}")
    return rows


def union_sources(paths: list[str]) -> list[dict]:
    """Every source row, deduplicated on (gene_id, trait, sex).

    The same association appears in more than one export with p-values that
    differ in the last digits (one export was rounded on the way out). The
    first source listed wins, so the union never depends on float equality.
    """
    merged: dict[tuple[str, str, str], dict] = {}
    for path in paths:
        for row in read_source(path):
            key = (row[GENE_ID_COLUMN], row[TRAIT_COLUMN], row[SEX_COLUMN])
            merged.setdefault(key, row)
    return list(merged.values())


def select_targets(rows: list[dict], max_per_cell: int | None = MAX_PER_CELL,
                   max_rows: int | None = MAX_ROWS) -> list[dict]:
    """Rank by p_F_analytic_min, cap each (trait, sex) cell, then truncate.

    Either cap may be None, meaning no cap; with both None this is a sort.
    Rows without a p-value sort last, in input order (the sort is stable).
    """
    ranked = sorted(rows, key=p_min_rank)
    per_cell: dict[tuple[str, str], int] = {}
    kept = []
    for row in ranked:
        cell = (row[TRAIT_COLUMN], row[SEX_COLUMN])
        if max_per_cell is not None and per_cell.get(cell, 0) >= max_per_cell:
            continue
        per_cell[cell] = per_cell.get(cell, 0) + 1
        kept.append(row)
        if max_rows is not None and len(kept) == max_rows:
            break
    return kept


def p_min_rank(row: dict) -> float:
    """p_F_analytic_min as a sort key; absent or blank ranks last."""
    value = str(row.get(P_MIN_COLUMN) or "").strip()
    return float(value) if value else float("inf")


def format_p(value: str | None) -> str:
    """2-decimal scientific, or blank when the export had no p-value."""
    value = str(value or "").strip()
    return f"{float(value):.2E}" if value else ""


def format_row(row: dict) -> dict:
    """p-values to 2-decimal scientific; the rest verbatim, blank if absent.

    The full float repr from the export is noise at this precision and makes
    the committed table unreadable, and nothing downstream reads the p-values
    (target_genes.py uses gene_id/trait/sex only).
    """
    return {column: (format_p(row.get(column)) if column in P_COLUMNS
                     else row.get(column, ""))
            for column in COLUMNS}


def write_targets(rows: list[dict], path: str) -> None:
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS, delimiter="\t",
                                lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(format_row(row))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-per-cell", type=int, default=MAX_PER_CELL,
                        help="keep at most N genes per (trait, sex); "
                             "default: no cap")
    parser.add_argument("--max-rows", type=int, default=MAX_ROWS,
                        help="keep at most N genes overall; default: no cap")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    missing = [p for p in SOURCE_LISTS if not os.path.exists(p)]
    if missing:
        raise SystemExit(f"Source list(s) not found: {missing}")

    rows = union_sources(SOURCE_LISTS)
    targets = select_targets(rows, args.max_per_cell, args.max_rows)
    write_targets(targets, OUTPUT_TSV)

    print(f"{len(rows)} novel associations in "
          f"{len(SOURCE_LISTS)} source list(s)")
    for trait, sex in sorted({(r[TRAIT_COLUMN], r[SEX_COLUMN])
                              for r in targets}):
        n = sum(1 for r in targets
                if (r[TRAIT_COLUMN], r[SEX_COLUMN]) == (trait, sex))
        print(f"  {trait} {sex}: {n}")
    print(f"Wrote {len(targets)} targets to {OUTPUT_TSV}")


if __name__ == "__main__":
    main()

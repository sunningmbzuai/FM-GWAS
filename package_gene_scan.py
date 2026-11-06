"""Package a gene_scan summary for Ning: result tables + README, zipped.

Reads the summarize_gene_scan.py tables in --results, adds gene symbols from a
GTF when one is given, writes a README with the method, thresholds,
exclusions and counts, and zips the folder to --zip.

    python package_gene_scan.py --results DIR --zip OUT.zip [--gtf GTF] \
        [--title "..."]
"""
from __future__ import annotations

import argparse
import gzip
import re
import shutil
import sys
import zipfile
from datetime import date
from pathlib import Path

import pandas as pd

import top_hits
from trait_qc import MAX_COVARIATE_R

STEM = "gene_trait_scan.by_field"
GENE_NAME = re.compile(r'gene_id "([^"]+)".*?gene_name "([^"]+)"')
TOP_N = 50


def gene_symbols(gtf: Path | None) -> dict[str, str]:
    """Unversioned Ensembl id -> symbol from the gene lines of a GTF."""
    if gtf is None or not gtf.is_file():
        return {}
    opener = gzip.open if gtf.suffix == ".gz" else open
    symbols = {}
    with opener(gtf, "rt") as handle:
        for line in handle:
            if "\tgene\t" not in line:
                continue
            match = GENE_NAME.search(line)
            if match:
                symbols[match.group(1).split(".")[0]] = match.group(2)
    return symbols


def with_symbols(table: pd.DataFrame, symbols: dict[str, str]) -> pd.DataFrame:
    symbol = table.gene_id.str.split(".").str[0].map(symbols).fillna("")
    cols = ["gene_id", "gene_symbol"] + [c for c in table.columns if c != "gene_id"]
    return table.assign(gene_symbol=symbol)[cols]


def readme(title: str, rows: int, n_genes: int, hits: pd.DataFrame, strict: pd.DataFrame,
           excluded: pd.DataFrame, strict_p: float, genetic_pcs: int = 0) -> str:
    anc = hits[hits.ancestry_linked.astype(bool)] if "ancestry_linked" in hits else hits[:0]
    reasons = excluded.excluded.str.split(" \\(").str[0].value_counts()
    reason_lines = "\n".join(f"    {n:>5}  {r}" for r, n in reasons.items())
    if genetic_pcs:
        covariates = (f"intercept + age + sex + genetic PCs 1-{genetic_pcs} (age + PCs\n"
                      f"  in F and M). Sensitivity analysis: FM-GWAS itself uses age and sex\n"
                      f"  only")
        caveat = (f"  Genetic PCs 1-{genetic_pcs} are covariates here. Hits that pass in the\n"
                  f"  age + sex scan but not here are likely driven by ancestry.\n"
                  f"  Ancestry-linked fields at p < 2.5e-6: {len(anc):,} rows.")
    else:
        covariates = "intercept + age + sex (age only in F and M)"
        caveat = (f"  The gene embeddings predict UKB genetic PCs strongly, and some hits are on\n"
                  f"  ancestry-linked fields (birthplace, ethnicity, country of birth:\n"
                  f"  ancestry_linked = True, {len(anc):,} of the p < 2.5e-6 rows, "
                  f"{anc.gene_id.nunique()} genes).\n"
                  f"  Hits on other ancestry-structured traits may also be partly ancestry.\n"
                  f"  These results have no genetic-PC adjustment, as in FM-GWAS; a run with\n"
                  f"  genetic PCs 1-10 as extra covariates can be made on request.")
    return f"""{title}
{"=" * len(title)}
Packaged {date.today():%d %B %Y}.

Model (FM-GWAS, as in the paper and insample_hpp_AG.py)
  Per gene, the gene-embedding PCA (variance 0.95) is fitted once per sample
  (All, F, M) and reused for every trait of that sample. Reduced model:
  {covariates}. Full model: reduced + gene PCs.
  Quantitative traits: partial F test. Binary traits: logistic LRT.
  All 60 embedding types; p_min is the lowest p over them.
  Genes tested: {n_genes} (genes with no UKB feature table cannot be tested).

Traits
  Every quantitative or binary field of every table in the UKB phenotype
  directory, plus the 8 FM-GWAS traits with their curated labels. Many tables
  repeat the same UKB field (derived train/test splits), so results are given
  per UKB field: for each gene x field x sex, the table with the lowest p;
  n_tables and tables list every table that carried the field.

  Not tested or not reported:
    genetic PCs (ancestry axes, not phenotypes)
    traits with |r| >= {MAX_COVARIATE_R} with sex or age (already covariates),
    '_nan' missingness indicators, and id / bookkeeping columns:
{reason_lines}
    full list with reasons: excluded_traits.tsv

Thresholds
  FM-GWAS genome-wide: p < 2.5e-6.
  Strict: 0.05 / ({rows:,} gene x field x sex rows x 60 embeddings)
          = {strict_p:.2e}, which also corrects for taking the minimum
          over embeddings.

Results
  {rows:,} gene x field x sex rows.
  p < 2.5e-6: {len(hits):,} rows, {hits.gene_id.nunique()} genes, {hits.field.nunique()} fields.
  p < {strict_p:.2e}: {len(strict):,} rows, {strict.gene_id.nunique()} genes, {strict.field.nunique()} fields.
  p = 0 means below the double-precision floor (1e-308); rank those by stat.

Caveat: ancestry
{caveat}

Files
  hits.tsv            gene x field x sex rows at p < 2.5e-6, by p
  all.tsv.gz          every gene x field x sex row
  excluded_traits.tsv traits left out, with the reason
  top_genes.tsv       per gene: fields at p < 2.5e-6 and the best hit

Columns: gene_id, gene_symbol, sex (All/F/M), field (UKB field id, or the
column name for non-UKB tables), title, p_min, best_embedding, best_stat,
best_df, best_components (gene PCs), n_samples, n_cases (binary only),
n_embeddings_tested, genome_wide, bonferroni (0.05 / rows), ancestry_linked,
trait_id / path (the table the best p came from), n_tables, tables.
"""


def top_genes(hits: pd.DataFrame) -> pd.DataFrame:
    best = hits.sort_values(["p_min", "best_stat"], ascending=[True, False]) \
        .drop_duplicates("gene_id").set_index("gene_id")
    fields = hits.groupby("gene_id").field.nunique().rename("n_fields")
    cols = ["gene_symbol", "field", "title", "sex", "p_min", "best_embedding"]
    out = best[cols].rename(columns={c: f"best_{c}" for c in cols[1:]})
    return out.join(fields).sort_values("n_fields", ascending=False).reset_index()


def package(results: Path, zip_path: Path, title: str, gtf: Path | None,
            genetic_pcs: int = 0) -> Path:
    symbols = gene_symbols(gtf)
    rows = top_hits.count_rows(results / f"{STEM}.tsv")
    dtypes = {"trait_id": str, "field": str}
    hits = with_symbols(pd.read_csv(results / f"{STEM}.hits.tsv", sep="\t", dtype=dtypes),
                        symbols)
    strict_p = top_hits.ALPHA / (rows * top_hits.N_EMBEDDINGS)
    excluded_path = results / "gene_trait_scan.excluded_traits.tsv"
    excluded = (pd.read_csv(excluded_path, sep="\t") if excluded_path.is_file()
                else pd.DataFrame(columns=["trait_id", "excluded"]))

    stage = zip_path.with_suffix("")
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)
    hits.to_csv(stage / "hits.tsv", sep="\t", index=False)
    top_genes(hits).to_csv(stage / "top_genes.tsv", sep="\t", index=False)
    excluded.to_csv(stage / "excluded_traits.tsv", sep="\t", index=False)
    every = with_symbols(pd.read_csv(results / f"{STEM}.tsv", sep="\t", dtype=dtypes), symbols)
    every.to_csv(stage / "all.tsv.gz", sep="\t", index=False, compression="gzip")
    (stage / "README.txt").write_text(
        readme(title, rows, every.gene_id.nunique(), hits, hits[hits.p_min < strict_p],
               excluded, strict_p, genetic_pcs))

    tmp = zip_path.with_name(zip_path.name + ".tmp")
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(stage.iterdir()):
            archive.write(path, f"{stage.name}/{path.name}")
    tmp.replace(zip_path)
    print(f"wrote {zip_path} ({zip_path.stat().st_size / 1e6:.1f} MB): "
          f"{', '.join(p.name for p in sorted(stage.iterdir()))}"
          + ("" if symbols else "; no GTF, gene_symbol left empty"))
    return zip_path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--results", required=True)
    ap.add_argument("--zip", required=True)
    ap.add_argument("--gtf")
    ap.add_argument("--title", default="UKB gene x trait scan")
    ap.add_argument("--genetic-pcs", type=int, default=0,
                    help="genetic PCs the scan adjusted for (README wording only)")
    args = ap.parse_args(argv)
    package(Path(args.results), Path(args.zip), args.title,
            Path(args.gtf) if args.gtf else None, args.genetic_pcs)
    return 0


if __name__ == "__main__":
    sys.exit(main())

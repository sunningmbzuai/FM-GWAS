"""package_gene_scan.py: zip of the summary tables with a README."""
import sys
import zipfile
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import package_gene_scan as P  # noqa: E402
import summarize_gene_scan as sgs  # noqa: E402
from test_summarize_gene_scan import write_part  # noqa: E402


@pytest.mark.unit
def test_package_has_tables_symbols_and_readme(tmp_path: Path) -> None:
    results = tmp_path / "results"
    write_part(results / "per_gene/G/part0.tsv.gz", "ENSG00000185038.10", "ukbA:30840",
               [0.0, 0.5])
    write_part(results / "per_gene/G/part1.tsv.gz", "ENSG00000185038.10", "ukbA:50", [0.3])
    catalog = tmp_path / "catalog.tsv"
    pd.DataFrame({"trait_id": ["ukbA:30840", "ukbA:50"], "title": ["", ""],
                  "path": ["a", "a"], "field": ["30840", "50"]}).to_csv(
        catalog, sep="\t", index=False)
    sgs.summarize(results, catalog)
    gtf = tmp_path / "genes.gtf"
    gtf.write_text('2\tHAVANA\tgene\t1\t9\t.\t+\t.\tgene_id "ENSG00000185038.6"; '
                   'gene_type "protein_coding"; gene_name "MROH2A";\n')

    zip_path = P.package(results, tmp_path / "out" / "scan_pkg.zip", "Test scan", gtf)
    names = set(zipfile.ZipFile(zip_path).namelist())
    assert names == {f"scan_pkg/{f}" for f in ("README.txt", "hits.tsv", "all.tsv.gz",
                                                "excluded_traits.tsv", "top_genes.tsv")}
    stage = tmp_path / "out" / "scan_pkg"
    hits = pd.read_csv(stage / "hits.tsv", sep="\t")
    assert list(hits.columns[:2]) == ["gene_id", "gene_symbol"]
    assert set(hits.gene_symbol) == {"MROH2A"} and set(hits.field) == {30840}
    top = pd.read_csv(stage / "top_genes.tsv", sep="\t")
    assert top.n_fields.tolist() == [1] and top.best_field.tolist() == [30840]
    text = (stage / "README.txt").read_text()
    assert "Test scan" in text and "p < 2.5e-6: 2 rows, 1 genes, 1 fields" in text
    assert len(pd.read_csv(stage / "all.tsv.gz", sep="\t")) == 4

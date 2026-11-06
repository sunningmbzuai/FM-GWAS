"""summarize_gene_scan.py: per-cell minimum p, parallel read == serial read."""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import summarize_gene_scan as sgs  # noqa: E402

COLUMNS = ["gene_id", "trait_id", "sex", "kind", "embed_type", "n_samples", "n_cases",
           "gene_pca_components", "df", "stat", "p", "status"]


def write_part(path: Path, gene: str, trait: str, ps: list) -> None:
    rows = [dict(gene_id=gene, trait_id=trait, sex=sex, kind="quant", embed_type=f"e{i}",
                 n_samples=100, n_cases=None, gene_pca_components=3, df=3, stat=1.0,
                 p=p, status="ok") for sex in ("All", "F") for i, p in enumerate(ps)]
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=COLUMNS).to_csv(path, sep="\t", index=False)


@pytest.fixture
def results(tmp_path: Path) -> Path:
    # trait_id "0123" checks ids stay strings (no leading-zero loss).
    write_part(tmp_path / "per_gene/G1/part000.tsv.gz", "G1", "0123", [0.5, 1e-9, None])
    write_part(tmp_path / "per_gene/G2/part000.tsv.gz", "G2", "T2", [0.2, 0.3, 0.04])
    return tmp_path


@pytest.mark.unit
def test_min_p_per_cell(results: Path) -> None:
    table = sgs.summarize(results, None, workers=1).set_index(["gene_id", "sex"])
    assert table.loc[("G1", "All"), "p_min"] == pytest.approx(1e-9)
    assert table.loc[("G1", "All"), "best_embedding"] == "e1"
    assert table.loc[("G1", "All"), "n_embeddings_tested"] == 2
    assert table.loc[("G1", "All"), "trait_id"] == "0123"
    assert table.loc[("G2", "F"), "p_min"] == pytest.approx(0.04)
    assert len(table) == 4


@pytest.mark.unit
def test_parallel_matches_serial(results: Path) -> None:
    serial = sgs.summarize(results, None, workers=1)
    parallel = sgs.summarize(results, None, workers=2)
    pd.testing.assert_frame_equal(serial.reset_index(drop=True),
                                  parallel.reset_index(drop=True))


@pytest.fixture
def messy(tmp_path: Path) -> tuple[Path, Path]:
    """Same UKB field 30840 in two tables, plus a genetic PC 'trait'."""
    write_part(tmp_path / "per_gene/G1/part000.tsv.gz", "G1", "ukbA:30840", [1e-20, 0.5])
    write_part(tmp_path / "per_gene/G1/part001.tsv.gz", "G1", "ukbB_train:30840", [1e-30, 0.5])
    write_part(tmp_path / "per_gene/G1/part002.tsv.gz", "G1", "ukbA_genetic_pcs:pc1", [0.0])
    catalog = tmp_path / "trait_catalog.tsv"
    pd.DataFrame({"trait_id": ["ukbA:30840", "ukbB_train:30840", "ukbA_genetic_pcs:pc1"],
                  "title": ["", "", ""], "path": ["a", "b", "c"],
                  "field": ["30840", "30840", "pc1"]}).to_csv(catalog, sep="\t", index=False)
    return tmp_path, catalog


@pytest.mark.unit
def test_genetic_pcs_dropped_and_fields_collapsed(messy) -> None:
    out, catalog = messy
    table = sgs.summarize(out, catalog, workers=1)
    assert not table.trait_id.str.contains("genetic_pcs").any()
    fields = pd.read_csv(out / "gene_trait_scan.by_field.tsv", sep="\t", dtype={"field": str})
    assert len(fields) == 2                       # G1 x 30840 x {All, F}
    row = fields[fields.sex == "All"].iloc[0]
    assert row.field == "30840" and row.n_tables == 2
    assert row.trait_id == "ukbB_train:30840" and row.p_min == pytest.approx(1e-30)
    assert row.tables == "ukbA:30840;ukbB_train:30840"
    assert bool(row.bonferroni)                   # 0.05 / 2 rows


@pytest.mark.unit
def test_from_summary_rebuilds_the_same_tables(messy) -> None:
    out, catalog = messy
    first = sgs.summarize(out, catalog, workers=1)
    fields = pd.read_csv(out / "gene_trait_scan.by_field.tsv", sep="\t")
    for part in (out / "per_gene").rglob("*.tsv.gz"):
        part.unlink()                             # parts no longer needed
    again = sgs.summarize(out, catalog, from_summary=True)
    pd.testing.assert_frame_equal(first.reset_index(drop=True), again.reset_index(drop=True),
                                  check_dtype=False)
    pd.testing.assert_frame_equal(fields, pd.read_csv(out / "gene_trait_scan.by_field.tsv",
                                                      sep="\t"))

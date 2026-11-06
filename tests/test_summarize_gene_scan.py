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

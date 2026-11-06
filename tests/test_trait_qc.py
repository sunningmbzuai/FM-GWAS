"""trait_qc.py: covariate correlations and which traits are not phenotypes."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import summarize_gene_scan as sgs  # noqa: E402
import trait_qc as Q  # noqa: E402
from test_summarize_gene_scan import write_part  # noqa: E402

N = 1000


def make_scan(root: Path) -> Path:
    rng = np.random.default_rng(0)
    sex = rng.integers(0, 2, N)
    age = rng.normal(55, 8, N)
    real = rng.normal(size=N)
    real[rng.random(N) < 0.3] = np.nan                    # partly measured
    sex_woe = np.where(sex == 1, 0.7, -0.6) + rng.normal(0, 0.05, N)
    women_only = np.where(sex == 0, rng.normal(size=N), np.nan)   # r undefined
    columns = {"ukbA:30840": real, "ukbB:Sex_woe": sex_woe, "ukbB:6153-0.0_nan": sex * 1.0,
               "ukbB:code": rng.integers(0, 5, N) * 1.0, "ukbA:2724": women_only,
               "ukbA:129": rng.normal(size=N), "ukbB:age_copy": age + rng.normal(0, 1, N)}
    scan = root / "traits"
    scan.mkdir()
    pd.DataFrame({"participant_id": np.arange(N), "sex": sex, "age": age}).to_csv(
        scan / "participants.tsv", sep="\t", index=False)
    np.save(scan / "trait_values.npy", np.column_stack(list(columns.values())))
    pd.DataFrame({"trait_id": list(columns), "column": range(len(columns)),
                  "field": [c.split(":")[1] for c in columns],
                  "title": "", "path": "p"}).to_csv(scan / "trait_catalog.tsv",
                                                    sep="\t", index=False)
    return scan


@pytest.mark.unit
def test_chunked_r_matches_numpy(tmp_path, monkeypatch):
    monkeypatch.setattr(Q, "ROWS_PER_CHUNK", 77)          # many chunk boundaries
    scan = make_scan(tmp_path)
    values = np.load(scan / "trait_values.npy")
    people = pd.read_csv(scan / "participants.tsv", sep="\t")
    got = Q.covariate_r(values, people)
    for j in (0, 1, 6):
        ok = np.isfinite(values[:, j])
        for cov in ("sex", "age"):
            want = np.corrcoef(values[ok, j], people[cov].to_numpy()[ok])[0, 1]
            assert got[f"r_{cov}"][j] == pytest.approx(want, abs=1e-9)
    assert np.isnan(got.r_sex[4])                         # women only: sex constant


@pytest.mark.unit
def test_exclusions_and_cache(tmp_path):
    scan = make_scan(tmp_path)
    catalog = pd.read_csv(scan / "trait_catalog.tsv", sep="\t", dtype=str)
    qc = Q.classify(catalog, Q.load_covariate_r(scan)).set_index("trait_id")
    assert (scan / Q.CACHE_NAME).is_file()
    assert qc.excluded["ukbB:Sex_woe"].startswith("covariate proxy")
    assert qc.excluded["ukbB:age_copy"].startswith("covariate proxy")
    assert qc.excluded["ukbB:6153-0.0_nan"] == "missingness indicator"
    assert qc.excluded["ukbB:code"].startswith("not a phenotype")
    for kept in ("ukbA:30840", "ukbA:2724", "ukbA:129"):
        assert qc.excluded[kept] == ""
    assert qc.ancestry_linked["ukbA:129"] and not qc.ancestry_linked["ukbA:30840"]


@pytest.mark.unit
def test_summary_drops_non_phenotypes(tmp_path):
    scan = make_scan(tmp_path)
    out = tmp_path / "results"
    catalog = pd.read_csv(scan / "trait_catalog.tsv", sep="\t", dtype=str)
    for k, trait in enumerate(catalog.trait_id):
        write_part(out / f"per_gene/G1/part{k}.tsv.gz", "G1", trait, [1e-9])
    table = sgs.summarize(out, scan / "trait_catalog.tsv", scan=scan)
    assert set(table.trait_id) == {"ukbA:30840", "ukbA:2724", "ukbA:129"}
    assert table.set_index("trait_id").ancestry_linked.groupby(level=0).first().to_dict() \
        == {"ukbA:129": True, "ukbA:2724": False, "ukbA:30840": False}
    excluded = pd.read_csv(out / "gene_trait_scan.excluded_traits.tsv", sep="\t")
    assert len(excluded) == 4

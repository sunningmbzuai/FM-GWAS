"""Column-selective, memory-bounded readers for the UKB parquet sources.

ukb676772.parquet is 502,244 rows x 23,550 columns. Nothing here ever reads it
whole: single columns are pulled by name, and the wide array fields (ICD-10 has
259 array columns) are read in batches and reduced to a boolean mask per batch,
so peak memory is one batch, not the whole field.

Needs pyarrow -- run under the "microbiome" env, NOT wasp_env (which has no
pyarrow) and never the precompiled deep_learning env.
"""
from typing import Callable, Iterable, Iterator, Sequence

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

import ukb_fields as F

# ICD-10 read width. 32 string columns x 502k rows is a few hundred MB at most.
ARRAY_BATCH = 32


def column_names(path: str) -> list[str]:
    """Every column name in a parquet file, read from the schema only."""
    return list(pq.read_schema(path).names)


def array_columns(path: str, field: str, instance: int = 0) -> list[str]:
    """The "<field>-<instance>.<array>" columns of one multi-array UKB field."""
    prefix = f"{field}-{instance}."
    return [c for c in column_names(path) if c.startswith(prefix)]


def read_columns(path: str, columns: Sequence[str], index: str = F.EID) -> pd.DataFrame:
    """Read named columns (plus the id column) into a DataFrame indexed by id."""
    wanted = [index, *(c for c in columns if c != index)]
    missing = set(wanted) - set(column_names(path))
    if missing:
        raise KeyError(f"{path} has no column(s): {sorted(missing)}")
    table = pq.read_table(path, columns=wanted)
    frame = table.to_pandas()
    return frame.set_index(index)


def _batches(items: Sequence[str], size: int) -> Iterator[Sequence[str]]:
    for start in range(0, len(items), size):
        yield items[start:start + size]


def match_many(
    path: str,
    columns: Sequence[str],
    predicates: dict[str, Callable[[pd.Series], pd.Series]],
    index: str = F.EID,
) -> pd.DataFrame:
    """OR each predicate across many columns in ONE pass over the data.

    Returns a boolean DataFrame indexed by `index`, one column per predicate
    name, True where ANY of `columns` satisfies that predicate. Read in
    ARRAY_BATCH-column chunks; every predicate is evaluated against a chunk
    before it is dropped, so a 259-column field is traversed once no matter how
    many disease definitions are being tested against it.
    """
    if not columns:
        raise ValueError("match_many called with no columns")
    if not predicates:
        raise ValueError("match_many called with no predicates")

    hits: dict[str, pd.Series] | None = None
    for batch in _batches(list(columns), ARRAY_BATCH):
        frame = read_columns(path, batch, index=index)
        batch_hits = {
            name: _any_match(frame, predicate)
            for name, predicate in predicates.items()
        }
        hits = batch_hits if hits is None else {
            name: hits[name] | batch_hits[name] for name in predicates
        }
        del frame
    return pd.DataFrame(hits)


def _any_match(frame: pd.DataFrame,
               predicate: Callable[[pd.Series], pd.Series]) -> pd.Series:
    hit = pd.Series(False, index=frame.index)
    for col in frame.columns:
        hit = hit | predicate(frame[col]).fillna(False)
    return hit


def icd_prefix_predicate(prefixes: Iterable[str]) -> Callable[[pd.Series], pd.Series]:
    """Predicate matching ICD-10 strings ("E119") by prefix ("E11")."""
    wanted = tuple(prefixes)

    def match(series: pd.Series) -> pd.Series:
        return series.astype("string").str.startswith(wanted, na=False)

    return match


def code_set_predicate(codes: Iterable[float]) -> Callable[[pd.Series], pd.Series]:
    """Predicate matching a numeric coding column against a set of codes."""
    wanted = set(codes)

    def match(series: pd.Series) -> pd.Series:
        return pd.to_numeric(series, errors="coerce").isin(wanted)

    return match


def mask_zeros_as_missing(frame: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    """Replace exact zeros with NaN in the named columns.

    UK Biobank writes some DXA measures as an exact 0 where the value is
    actually absent (see the phenotypes export README). Left in place, a
    "0 cm3 of visceral fat" sits at the bottom of a continuous trait's
    distribution as if it were the leanest participant in the cohort.
    """
    present = [c for c in columns if c in frame.columns]
    if not present:
        return frame
    replacements = {
        c: pd.to_numeric(frame[c], errors="coerce").replace(0.0, np.nan)
        for c in present
    }
    return frame.assign(**replacements)


def lowest_across_sites(frame: pd.DataFrame, columns: Sequence[str]) -> pd.Series:
    """Minimum across several measurement sites, ignoring sites not measured.

    Bone-density classification uses the lowest T-score among the measured
    skeletal sites, so a participant osteoporotic at one site is not scored
    normal because another site is healthy.
    """
    missing = set(columns) - set(frame.columns)
    if missing:
        raise KeyError(f"frame has no column(s): {sorted(missing)}")
    return frame[list(columns)].min(axis=1, skipna=True)


def mean_of_readings(frame: pd.DataFrame, primary: Sequence[str],
                     fallback: Sequence[str]) -> pd.Series:
    """Mean of repeated readings, falling back to a second device where absent.

    UKB takes two automated BP readings; participants the automated device
    failed on have only the manual sphygmomanometer columns (93/94). Averaging
    the two automated reads and filling from the manual mean recovers those.
    """
    primary_mean = frame[list(primary)].mean(axis=1)
    fallback_mean = frame[list(fallback)].mean(axis=1)
    return primary_mean.fillna(fallback_mean)

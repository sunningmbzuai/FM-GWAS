"""Deep scan of the UKB phenotype directory: every testable trait, one table.

Walks UKBV_PHENO_DIR recursively, opens every tabular file that carries a
participant id column (parquet, csv, tsv), and runs each field through
trait_rules.py. Accepted fields become traits: quantitative ones get the F
test, binary ones the logistic LRT, exactly as the 8 FM-GWAS traits do. The 8
FM-GWAS traits themselves are added from the cohort tables
(UKBV_COHORT_DIR/<trait>_All.tsv), so their curated labels are used, not a
raw-field guess.

Values are restricted to QC-passed participants (config.QC_FAM) with age and
sex. Fields whose values are identical to a trait already kept (the same
measure in the dump and in the export) are dropped as duplicates.

Writes to --out (default UKBV_OUT_ROOT/trait_scan):
  trait_catalog.tsv   one row per trait: id, source, field, instance, kind, N, ...
  trait_skipped.tsv   every field not kept, with the reason
  trait_values.npy    participants x traits float64 (NaN = not measured)
  participants.tsv    participant_id, sex, age -- the row order of the .npy

Run under the microbiome env (pyarrow):
    python trait_scan.py [--workers 16] [--out DIR]
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

import config
import trait_rules as R
import ukb_fields as F
from build_cohorts import load_qc_ids

TABULAR = (".parquet", ".csv", ".tsv", ".txt", ".csv.gz", ".tsv.gz")
DICTIONARY_HINTS = ("dictionary", "showcase")
MAX_TEXT_BYTES = 4 << 30   # text tables larger than this are not parsed
COLUMNS_PER_BATCH = 64     # parquet: columnar reads, so many small batches
HEARTBEAT_SECONDS = 60
FMGWAS_TRAITS = tuple(config.TRAITS)

_QC_IDS: np.ndarray | None = None
_DICTIONARY: dict[str, tuple[str, str]] = {}


def log(message: str) -> None:
    print(f"[{time.strftime('%F %T')}] [trait_scan] {message}", file=sys.stderr, flush=True)


# ------------------------------------------------------------------ discovery
def tabular_files(root: Path) -> list[Path]:
    found = []
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        dirnames[:] = sorted(d for d in dirnames if not d.startswith("."))
        for name in sorted(filenames):
            if name.startswith(".") or not name.lower().endswith(TABULAR):
                continue
            found.append(Path(dirpath) / name)
    return found


def header(path: Path) -> list[str]:
    if path.suffix == ".parquet":
        return list(pq.read_schema(path).names)
    sep = "," if ".csv" in path.name.lower() else "\t"
    return list(pd.read_csv(path, sep=sep, nrows=0).columns)


def load_dictionary(files: list[Path]) -> dict[str, tuple[str, str]]:
    """FieldID -> (ValueType, Field title) from a UKB Showcase dictionary, if any."""
    for path in files:
        if not any(h in path.name.lower() for h in DICTIONARY_HINTS):
            continue
        try:
            sep = "," if ".csv" in path.name.lower() else "\t"
            table = pd.read_csv(path, sep=sep, dtype=str)
        except (OSError, ValueError, pd.errors.ParserError):
            continue
        cols = {c.lower(): c for c in table.columns}
        if "fieldid" in cols and "valuetype" in cols:
            title = cols.get("field", cols["fieldid"])
            log(f"data dictionary: {path} ({len(table):,} fields)")
            return {str(r[cols["fieldid"]]): (str(r[cols["valuetype"]]), str(r[title]))
                    for _, r in table.iterrows()}
    log("no data dictionary found: value types guessed from the values")
    return {}


# ------------------------------------------------------------------ workers
def _init(qc_ids: np.ndarray, dictionary: dict[str, tuple[str, str]]) -> None:
    global _QC_IDS, _DICTIONARY
    _QC_IDS, _DICTIONARY = qc_ids, dictionary


def _read(path: str, id_column: str, names: list[str]) -> pd.DataFrame:
    if path.endswith(".parquet"):
        frame = pq.read_table(path, columns=[id_column, *names]).to_pandas()
    else:
        sep = "," if ".csv" in os.path.basename(path).lower() else "\t"
        frame = pd.read_csv(path, sep=sep, usecols=[id_column, *names], low_memory=False)
    frame[id_column] = pd.to_numeric(frame[id_column], errors="coerce")
    frame = frame.dropna(subset=[id_column]).drop_duplicates(id_column)
    frame = frame.set_index(frame[id_column].astype(np.int64)).drop(columns=id_column)
    return frame.reindex(_QC_IDS)


def scan_batch(task: tuple[str, str, list[tuple[str, list[R.Column]]]]) -> list[dict]:
    """Classify a batch of fields of one file; accepted ones carry their values."""
    path, id_column, fields = task
    names = [c.name for _, cols in fields for c in cols]
    try:
        frame = _read(path, id_column, names)
    except (OSError, ValueError, KeyError) as error:
        return [dict(path=path, field=f, kind="skip", reason=f"read failed: {error}")
                for f, _ in fields]
    out = []
    for field, columns in fields:
        value_type, title = _DICTIONARY.get(field, (None, ""))
        record = dict(path=path, field=field, title=title, value_type=value_type)
        numeric = [c for c in columns
                   if pd.api.types.is_numeric_dtype(frame[c.name])
                   or pd.api.types.is_bool_dtype(frame[c.name])]
        if not numeric:
            out.append(dict(record, kind="skip", reason="not numeric"))
            continue
        instance, merged = R.best_instance(frame, numeric)
        verdict = R.classify(merged, value_type)
        record.update(instance=instance, kind=verdict.kind, reason=verdict.reason,
                      n=verdict.n, n_minor=verdict.n_minor)
        if verdict.kind != "skip":
            record["values"] = R.encode(merged, verdict).to_numpy(np.float64)
        out.append(record)
    return out


def plan_batches(path: Path, names: list[str], id_column: str,
                 skipped: list[dict]) -> list[tuple]:
    tasks, batch, width = [], [], 0
    for field, columns in R.group_fields(names, id_column).items():
        if R.is_excluded(field):
            skipped.append(dict(path=str(path), field=field, kind="skip",
                                reason="administrative/genetic/registry field"))
            continue
        if R.too_many_arrays(columns):
            skipped.append(dict(path=str(path), field=field, kind="skip",
                                reason=f"{len(columns)} array columns (multi-select codes)"))
            continue
        batch.append((field, columns))
        width += len(columns)
        # A text table is parsed whole on every read, so it is one batch:
        # one read per file, not one per 64 columns.
        if width >= COLUMNS_PER_BATCH and path.suffix == ".parquet":
            tasks.append((str(path), id_column, batch))
            batch, width = [], 0
    if batch:
        tasks.append((str(path), id_column, batch))
    return tasks


# ------------------------------------------------------------------ assembly
def participants(qc_ids: set[int]) -> pd.DataFrame:
    cov = pq.read_table(config.UKB_MAIN_PARQUET, columns=[F.EID, F.SEX, F.AGE]).to_pandas()
    cov = cov.rename(columns={F.EID: "participant_id", F.SEX: "sex", F.AGE: "age"})
    cov = cov[cov.participant_id.isin(qc_ids)].dropna()
    cov = cov[cov.sex.isin([0, 1])].sort_values("participant_id")
    return cov.astype({"participant_id": np.int64, "sex": np.int64, "age": np.float64})


def fmgwas_records(cohort_dir: Path, ids: np.ndarray) -> list[dict]:
    """The 8 FM-GWAS traits, with their curated labels from the cohort tables."""
    records = []
    for trait in FMGWAS_TRAITS:
        path = cohort_dir / f"{trait}_All.tsv"
        if not path.is_file():
            log(f"[warn] no cohort table {path}: {trait} left out")
            continue
        table = pd.read_csv(path, sep="\t").set_index("participant_id")["label"]
        values = table.reindex(ids).to_numpy(np.float64)
        binary = set(np.unique(values[np.isfinite(values)])) <= {0.0, 1.0}
        present = values[np.isfinite(values)]
        records.append(dict(path=str(path), field=trait, title=trait, instance=0,
                            kind="binary" if binary else "quantitative",
                            reason="FM-GWAS trait (cohort table)", n=len(present),
                            n_minor=int(min(present.sum(), len(present) - present.sum()))
                            if binary else 0, values=values))
    return records


def fingerprint(values: np.ndarray) -> str:
    return hashlib.sha256(np.nan_to_num(values, nan=-1.234567e300).tobytes()).hexdigest()


def trait_id(record: dict) -> str:
    stem = Path(record["path"]).name.split(".")[0]
    if record.get("reason", "").startswith("FM-GWAS"):
        return record["field"]
    return f"{stem}:{record['field']}"


def assemble(records: list[dict], out: Path, people: pd.DataFrame) -> pd.DataFrame:
    kept, seen, columns, skipped = [], {}, [], []
    for rec in records:
        if rec.get("kind") == "skip" or "values" not in rec:
            skipped.append({k: v for k, v in rec.items() if k != "values"})
            continue
        key = fingerprint(rec["values"])
        if key in seen:
            skipped.append(dict({k: v for k, v in rec.items() if k != "values"},
                                kind="skip", reason=f"duplicate of {seen[key]}"))
            continue
        tid = trait_id(rec)
        seen[key] = tid
        columns.append(rec["values"])
        kept.append(dict(trait_id=tid, column=len(columns) - 1,
                         **{k: rec.get(k) for k in ("path", "field", "title", "instance",
                                                    "kind", "n", "n_minor", "value_type")}))
    catalog = pd.DataFrame(kept)
    log(f"writing {len(columns):,} traits x {len(people):,} participants")
    np.save(out / "trait_values.npy", np.column_stack(columns) if columns
            else np.empty((len(people), 0)))
    catalog.to_csv(out / "trait_catalog.tsv", sep="\t", index=False)
    pd.DataFrame(skipped).to_csv(out / "trait_skipped.tsv", sep="\t", index=False)
    people.to_csv(out / "participants.tsv", sep="\t", index=False)
    return catalog


def run_tasks(tasks: list[tuple], ids: np.ndarray, dictionary: dict,
              workers: int) -> list[dict]:
    """Scan every batch in parallel. Progress counts batches as they finish, in
    any order, and a heartbeat names the files still being read, so one slow
    table never looks like a hang."""
    records: list[dict] = []
    start, done = time.monotonic(), 0
    with ProcessPoolExecutor(workers, initializer=_init,
                             initargs=(ids, dictionary)) as pool:
        running = {pool.submit(scan_batch, task): task[0] for task in tasks}
        while running:
            finished, _ = wait(running, timeout=HEARTBEAT_SECONDS,
                               return_when=FIRST_COMPLETED)
            for future in finished:
                running.pop(future)
                records.extend(future.result())
                done += 1
            elapsed = time.monotonic() - start
            eta = elapsed / done * (len(tasks) - done) if done else float("nan")
            if finished and done % 10 and done != len(tasks):
                continue
            busy = sorted({Path(p).name for p in list(running.values())[:workers]})
            log(f"[ETA] traits: batch {done}/{len(tasks)}, elapsed {elapsed / 60:.1f} min, "
                f"ETA {eta / 60:.1f} min" + (f"; reading {', '.join(busy[:4])}" if busy else ""))
    return records


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pheno-dir", default=config._setting("UKBV_PHENO_DIR"))
    ap.add_argument("--cohort-dir", default=config.COHORT_DIR)
    ap.add_argument("--out", default=os.path.join(config.OUT_ROOT, "trait_scan"))
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    args = ap.parse_args(argv)
    if not args.pheno_dir or not os.path.isdir(args.pheno_dir):
        raise SystemExit(f"phenotype directory not found: {args.pheno_dir!r}")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    people = participants(load_qc_ids())
    ids = people.participant_id.to_numpy(np.int64)
    log(f"{len(ids):,} QC-passed participants with age and sex")

    files = tabular_files(Path(args.pheno_dir))
    log(f"{len(files)} tabular file(s) under {args.pheno_dir}")
    dictionary = load_dictionary(files)
    skipped: list[dict] = []
    tasks: list[tuple] = []
    for path in files:
        if any(h in path.name.lower() for h in DICTIONARY_HINTS):
            continue
        if path.suffix != ".parquet" and path.stat().st_size > MAX_TEXT_BYTES:
            skipped.append(dict(path=str(path), field="*", kind="skip",
                                reason="text table over 4 GB"))
            continue
        try:
            names = header(path)
        except (OSError, ValueError, pd.errors.ParserError) as error:
            skipped.append(dict(path=str(path), field="*", kind="skip",
                                reason=f"unreadable: {error}"))
            continue
        id_column = R.find_id_column(names)
        if id_column is None:
            skipped.append(dict(path=str(path), field="*", kind="skip",
                                reason="no participant id column"))
            continue
        file_tasks = plan_batches(path, names, id_column, skipped)
        log(f"  {path.name}: {len(names):,} columns -> {len(file_tasks)} batch(es)")
        tasks.extend(file_tasks)

    records = fmgwas_records(Path(args.cohort_dir), ids)
    records.extend(run_tasks(tasks, ids, dictionary, args.workers))
    log(f"assembling {len(records):,} field records (dedup + writing trait_values.npy)")
    catalog = assemble(records + skipped, out, people)
    counts = catalog.kind.value_counts().to_dict() if len(catalog) else {}
    log(f"{len(catalog)} traits kept {counts}; catalog -> {out / 'trait_catalog.tsv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""How each UKB disease case group is made up, for Ning's 2026-10-23 question.

UKB case rates run 2.5-3x HPP's for Hypertension and Osteoporosis. Ning asked
whether several phenotype classes were merged into one case group, and for the
class sizes. They were: a case is the union of ICD-10, self-report, medication
and a measurement cutoff (ukb_traits.py), and the cutoffs follow the paper's
broad definitions (osteopenia counts as osteoporosis, prediabetes as T2D).

Each case is put in exactly one class: the first one in the listed order that
it matches, so a diagnosed participant is never also counted under a
measurement. The classes therefore sum to the case count, and the labels are
the ones build_cohorts.py writes (same inputs, same labellers, same cohort
rows). Writes, under OUT_ROOT/case_breakdown/:

    case_classes.tsv   trait, sex, class, n, % of cases, % of cohort
    case_rates.tsv     UKB vs HPP sample and case counts (hpp_cohort_sizes.tsv)

Run with the microbiome env, like build_cohorts.py:
    python case_breakdown.py
"""
import os

import pandas as pd

import build_cohorts as B
import config
import ukb_fields as F
import ukb_source as S

CONTROL = "control"
UNCLASSIFIED = "case, no class matched"
SEXES = (("All", None), ("F", 0), ("M", 1))
HPP_SIZES_TSV = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "hpp_cohort_sizes.tsv")

Tiers = list[tuple[str, pd.Series]]


def assign_classes(labels: pd.Series, tiers: Tiers) -> pd.Series:
    """Each case -> its first matching class; each control -> CONTROL."""
    classes = pd.Series(CONTROL, index=labels.index, dtype=object)
    unassigned = labels == 1
    classes[unassigned] = UNCLASSIFIED
    for name, mask in tiers:
        hit = unassigned & mask.reindex(labels.index, fill_value=False).astype(bool)
        classes[hit] = name
        unassigned &= ~hit
    return classes


def class_rows(trait: str, sex: str, classes: pd.Series,
               names: list[str]) -> list[dict]:
    """One row per class (listed order, then unclassified, then controls)."""
    n_cohort = len(classes)
    n_cases = int((classes != CONTROL).sum())
    counts = classes.value_counts()
    order = names + ([UNCLASSIFIED] if UNCLASSIFIED in counts else []) + [CONTROL]
    return [{
        "trait": trait, "sex": sex, "class": name, "n": int(counts.get(name, 0)),
        "pct_of_cases": (100 * counts.get(name, 0) / n_cases
                         if name != CONTROL and n_cases else float("nan")),
        "pct_of_cohort": 100 * counts.get(name, 0) / n_cohort if n_cohort else float("nan"),
    } for name in order]


def hypertension_tiers(sbp: pd.Series, dbp: pd.Series, icd: pd.Series,
                       selfrep: pd.Series, bp_med: pd.Series) -> Tiers:
    return [
        ("ICD-10 I10-I15", icd),
        ("self-reported hypertension", selfrep),
        ("blood-pressure medication", bp_med),
        ("SBP >= 160 or DBP >= 100, no diagnosis",
         (sbp >= F.SBP_STAGE2) | (dbp >= F.DBP_STAGE2)),
        ("SBP 140-159 or DBP 90-99, no diagnosis",
         (sbp >= F.SBP_HIGH) | (dbp >= F.DBP_HIGH)),
    ]


def osteoporosis_tiers(tscore: pd.Series, icd: pd.Series,
                       selfrep: pd.Series) -> Tiers:
    return [
        ("ICD-10 M80-M81", icd),
        ("self-reported osteoporosis", selfrep),
        ("T-score <= -2.5 (osteoporosis), no diagnosis",
         tscore <= F.TSCORE_OSTEOPOROSIS),
        ("T-score -2.5 to -1.0 (osteopenia), no diagnosis",
         tscore <= F.TSCORE_LOW_BONE_MASS),
    ]


def t2d_tiers(inputs: B.CohortInputs) -> Tiers:
    icd, selfrep, singles = inputs.icd, inputs.selfrep, inputs.singles
    hba1c = singles[F.HBA1C]
    return [
        ("ICD-10 E11 (type 2)", icd["t2d"]),
        ("ICD-10 E10 (type 1)", icd["t1d"]),
        ("self-reported diabetes (any type)",
         selfrep["t2d"] | selfrep["t1d"] | selfrep["diabetes_unspec"]),
        ("doctor-diagnosed diabetes (field 2443)", singles[F.DIABETES_SELFREP] == 1),
        ("insulin", inputs.meds["insulin"]),
        ("HbA1c >= 48 (diabetes), no diagnosis", hba1c >= F.HBA1C_DIABETES),
        ("HbA1c 39-47 (prediabetes), no diagnosis", hba1c >= F.HBA1C_PREDIABETES),
    ]


def hyperlipidemia_tiers(inputs: B.CohortInputs) -> Tiers:
    return [
        ("ICD-10 E78", inputs.icd["hyperlipidemia"]),
        ("self-reported high cholesterol", inputs.selfrep["high_cholesterol"]),
        ("cholesterol medication", inputs.meds["cholesterol"]),
        ("total cholesterol >= 6.2 mmol/L, no diagnosis", inputs.chol >= F.CHOL_HIGH),
    ]


def obesity_tiers(bmi: pd.Series) -> Tiers:
    return [
        ("BMI >= 40 (class III)", bmi >= F.BMI_OBESE_III),
        ("BMI 35-39.9 (class II)", bmi >= F.BMI_OBESE_II),
        ("BMI 30-34.9 (class I)", bmi >= F.BMI_OBESE),
    ]


def disease_tiers(inputs: B.CohortInputs) -> dict[str, Tiers]:
    """The class lists per disease trait, from the same inputs as the labels."""
    singles = inputs.singles
    sbp = S.mean_of_readings(singles, F.SBP_AUTO, F.SBP_MANUAL)
    dbp = S.mean_of_readings(singles, F.DBP_AUTO, F.DBP_MANUAL)
    return {
        "Hyperlipidemia": hyperlipidemia_tiers(inputs),
        "Hypertension": hypertension_tiers(
            sbp, dbp, inputs.icd["hypertension"], inputs.selfrep["hypertension"],
            inputs.meds["blood_pressure"]),
        "Obesity": obesity_tiers(singles[F.BMI]),
        "Osteoporosis": osteoporosis_tiers(
            inputs.tscore, inputs.icd["osteoporosis"], inputs.selfrep["osteoporosis"]),
        "T2D": t2d_tiers(inputs),
    }


def cohort_labels(labels: pd.Series, covariates: pd.DataFrame,
                  sex: int | None) -> pd.Series:
    """The label column of the cohort file build_cohorts.write_cohorts writes."""
    base = covariates.join(labels.rename("label"), how="inner").dropna()
    if sex is not None:
        base = base[base["sex"] == sex]
    return base["label"].astype(int)


def case_rates(ukb: pd.DataFrame, hpp: pd.DataFrame) -> pd.DataFrame:
    """UKB and HPP case fractions side by side, with their ratio."""
    hpp = hpp.rename(columns={"n_samples": "hpp_n", "n_cases": "hpp_cases"})
    out = ukb.merge(hpp, on=["trait", "sex"], how="left")
    out["ukb_case_pct"] = 100 * out["ukb_cases"] / out["ukb_n"]
    out["hpp_case_pct"] = 100 * out["hpp_cases"] / out["hpp_n"]
    out["ukb_over_hpp"] = out["ukb_case_pct"] / out["hpp_case_pct"]
    return out[["trait", "sex", "ukb_n", "ukb_cases", "ukb_case_pct",
                "hpp_n", "hpp_cases", "hpp_case_pct", "ukb_over_hpp"]]


def main() -> None:
    out_dir = os.path.join(config.OUT_ROOT, "case_breakdown")
    os.makedirs(out_dir, exist_ok=True)
    inputs = B.load_inputs()
    labels = B.labels_for(inputs)

    rows, totals = [], []
    for trait, tiers in disease_tiers(inputs).items():
        names = [name for name, _ in tiers]
        for sex_name, sex_val in SEXES:
            cohort = cohort_labels(labels[trait], inputs.covariates, sex_val)
            rows += class_rows(trait, sex_name, assign_classes(cohort, tiers), names)
            totals.append({"trait": trait, "sex": sex_name, "ukb_n": len(cohort),
                           "ukb_cases": int(cohort.sum())})

    classes = pd.DataFrame(rows)
    rates = case_rates(pd.DataFrame(totals), pd.read_csv(HPP_SIZES_TSV, sep="\t"))
    classes.to_csv(os.path.join(out_dir, "case_classes.tsv"), sep="\t",
                   index=False, float_format="%.1f")
    rates.to_csv(os.path.join(out_dir, "case_rates.tsv"), sep="\t",
                 index=False, float_format="%.2f")
    unclassified = classes[classes["class"] == UNCLASSIFIED]
    if not unclassified.empty:
        print(f"[warn] {int(unclassified['n'].sum())} case(s) matched no class:")
        print(unclassified.to_string(index=False))
    print(rates.to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    print(f"Wrote case_classes.tsv and case_rates.tsv to {out_dir}")


if __name__ == "__main__":
    main()

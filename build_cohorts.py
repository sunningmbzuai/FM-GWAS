"""Build FM-GWAS cohort TSVs (participant_id/sex/age/label) for all 8 traits,
each in All/F/M variants, restricted to samples that passed genotype QC (config.QC_FAM).

FM-GWAS expects: tab-separated, columns participant_id, sex (0=F,1=M), age, label.

Phenotypes come from UKB release 676772 -- the raw dump for height, BMI, blood
pressure, ICD-10, self-reported illness, medication, sex and age; the tidied
phenotypes export for the named DXA and blood-biomarker columns. Trait
definitions follow the FM-GWAS paper's Methods (continuous
traits as measured values; disease traits as presence/absence, with osteopenia
counted as osteoporosis and prediabetes as T2D) and live in ukb_fields.py /
ukb_traits.py; the readers that keep the 23,550-column dump out of memory live in ukb_source.py.

Run with the "microbiome" conda env (pyarrow + pandas), NOT wasp_env.
"""
import os
from typing import Callable, Iterable, Sequence

import pandas as pd

import config
import ukb_fields as F
import ukb_source as S
import ukb_traits as T

SINGLE_COLUMNS = (
    F.SEX, F.AGE, F.HEIGHT, F.BMI, F.DIABETES_SELFREP, F.HBA1C,
    *F.SBP_AUTO, *F.DBP_AUTO, *F.SBP_MANUAL, *F.DBP_MANUAL,
)

# One pass over the 259-column ICD-10 field answers all five disease questions.
ICD_PREDICATES = {
    "hyperlipidemia": F.ICD_HYPERLIPIDEMIA,
    "hypertension": F.ICD_HYPERTENSION,
    "t2d": F.ICD_T2D,
    "t1d": F.ICD_T1D,
    "osteoporosis": F.ICD_OSTEOPOROSIS,
}
SELFREP_PREDICATES = {
    "high_cholesterol": (F.SELFREP_HIGH_CHOLESTEROL,),
    "hypertension": F.SELFREP_HYPERTENSION,
    "t2d": (F.SELFREP_T2D,),
    "t1d": (F.SELFREP_T1D,),
    "diabetes_unspec": (F.SELFREP_DIABETES_UNSPEC,),
    "osteoporosis": (F.SELFREP_OSTEOPOROSIS,),
}
MED_PREDICATES = {
    "cholesterol": (F.MED_CHOLESTEROL,),
    "blood_pressure": (F.MED_BLOOD_PRESSURE,),
    "insulin": (F.MED_INSULIN,),
}


def load_qc_ids() -> set[int]:
    """eids of samples that passed genotype QC (config.QC_FAM)."""
    fam = pd.read_csv(
        config.QC_FAM, sep=r"\s+", header=None,
        names=["FID", "IID", "PAT", "MAT", "SEX_FAM", "PHENO"],
    )
    return set(pd.to_numeric(fam["IID"], errors="coerce").dropna().astype(int))


def masks_over_array_field(
    field: str,
    predicates: dict[str, Sequence],
    kind: Callable[[Iterable], Callable[[pd.Series], pd.Series]],
) -> pd.DataFrame:
    """Reduce one wide UKB array field to one boolean column per definition."""
    columns = S.array_columns(config.UKB_MAIN_PARQUET, field)
    if not columns:
        raise KeyError(f"no instance-0 array columns for field {field} in "
                       f"{config.UKB_MAIN_PARQUET}")
    print(f"  field {field}: {len(columns)} array columns, "
          f"{len(predicates)} definitions")
    return S.match_many(
        config.UKB_MAIN_PARQUET, columns,
        {name: kind(codes) for name, codes in predicates.items()},
    )


def medication_masks() -> pd.DataFrame:
    """Touchscreen medication answers, men's and women's fields OR'd together.

    6177 is asked of men, 6153 of women, with codes 1/2/3 (cholesterol, blood
    pressure, insulin) shared. Every participant answers exactly one of the two,
    so the union is the cohort-wide answer.
    """
    men = masks_over_array_field(F.MEDS_MEN, MED_PREDICATES, S.code_set_predicate)
    women = masks_over_array_field(F.MEDS_WOMEN, MED_PREDICATES, S.code_set_predicate)
    both = men.index.union(women.index)
    return men.reindex(both, fill_value=False) | women.reindex(both, fill_value=False)


def align(frame: pd.DataFrame, index: pd.Index) -> pd.DataFrame:
    """Reindex boolean masks onto the cohort index; absent evidence is False."""
    return frame.reindex(index, fill_value=False).fillna(False)


def build_labels(singles: pd.DataFrame, icd: pd.DataFrame, selfrep: pd.DataFrame,
                 meds: pd.DataFrame, chol: pd.Series,
                 vat: pd.Series, tscore: pd.Series) -> dict[str, pd.Series]:
    mode, tail = config.QUANT_LABEL_MODE, F.EXTREME_TAIL_FRAC
    sbp = S.mean_of_readings(singles, F.SBP_AUTO, F.SBP_MANUAL)
    dbp = S.mean_of_readings(singles, F.DBP_AUTO, F.DBP_MANUAL)
    return {
        "Cholesterol": T.quantitative(chol, mode, tail),
        "Height": T.quantitative(singles[F.HEIGHT], mode, tail),
        "VAT": T.quantitative(vat, mode, tail),
        "Obesity": T.obesity(singles[F.BMI]),
        "Hyperlipidemia": T.hyperlipidemia(
            chol, icd["hyperlipidemia"], selfrep["high_cholesterol"],
            meds["cholesterol"],
        ),
        "Hypertension": T.hypertension(
            sbp, dbp, icd["hypertension"], selfrep["hypertension"],
            meds["blood_pressure"],
        ),
        "T2D": T.type_2_diabetes(
            icd["t2d"], icd["t1d"], selfrep["t2d"], selfrep["t1d"],
            selfrep["diabetes_unspec"], singles[F.DIABETES_SELFREP],
            meds["insulin"], singles[F.HBA1C],
        ),
        "Osteoporosis": T.osteoporosis(
            tscore, icd["osteoporosis"], selfrep["osteoporosis"],
        ),
    }


def describe_labels(labels: pd.Series) -> str:
    """One-line label summary: case fraction if binary, distribution if not.

    Continuous traits have tens of thousands of distinct values, so a
    value_counts dump is unreadable -- report quantiles instead.
    """
    if labels.empty:
        return "no labels"
    values = set(labels.unique().tolist())
    if values <= {0, 1}:
        cases = int((labels == 1).sum())
        return f"binary, {cases:,} cases ({cases / len(labels):.1%})"
    q = labels.quantile([0.0, 0.25, 0.5, 0.75, 1.0])
    return (f"continuous, median {q[0.5]:.3g} "
            f"[min {q[0.0]:.3g}, IQR {q[0.25]:.3g}-{q[0.75]:.3g}, max {q[1.0]:.3g}]")


def write_cohorts(trait: str, labels: pd.Series, covariates: pd.DataFrame) -> None:
    base = covariates.join(labels.rename("label"), how="inner").reset_index()
    base = base.rename(columns={F.EID: "participant_id"})[
        ["participant_id", "sex", "age", "label"]
    ].dropna()

    for sex_name, sex_val in (("All", None), ("F", 0), ("M", 1)):
        suffix = config.SEX_SUFFIXES[sex_name]
        gene_file = os.path.join(
            config.HPP_UNION_GENE_DIR, f"{trait}_extreme{suffix}.tsv"
        )
        if not os.path.exists(gene_file):
            print(f"[skip] {trait} {sex_name}: no gene file {gene_file}")
            continue

        cohort = base if sex_val is None else base[base["sex"] == sex_val]
        out_path = os.path.join(config.COHORT_DIR, f"{trait}_{sex_name}.tsv")
        cohort.to_csv(out_path, sep="\t", index=False)
        print(f"{trait} {sex_name}: {len(cohort):,} samples "
              f"({describe_labels(cohort['label'])}) -> {out_path}")


def main() -> None:
    os.makedirs(config.COHORT_DIR, exist_ok=True)

    qc_ids = load_qc_ids()
    print(f"QC-passed sample count: {len(qc_ids):,}")

    print("reading single-column phenotypes")
    singles = S.read_columns(config.UKB_MAIN_PARQUET, SINGLE_COLUMNS)
    singles = singles[singles.index.isin(qc_ids)]
    print(f"  QC-passed rows present in the dump: {len(singles):,}")

    print("reducing ICD-10 diagnoses")
    icd = align(masks_over_array_field(F.ICD10_ALL, ICD_PREDICATES,
                                       S.icd_prefix_predicate), singles.index)
    print("reducing self-reported illness")
    selfrep = align(masks_over_array_field(F.SELFREP_ILLNESS, SELFREP_PREDICATES,
                                           S.code_set_predicate), singles.index)
    print("reducing touchscreen medication")
    meds = align(medication_masks(), singles.index)

    print("reading export biomarker/DXA columns")
    chol = S.read_columns(config.UKB_BLOOD_PARQUET,
                          [F.EXPORT_CHOLESTEROL])[F.EXPORT_CHOLESTEROL]
    dxa = S.read_columns(config.UKB_DXA_PARQUET,
                         [F.EXPORT_VAT_VOLUME, *F.EXPORT_BMD_TSCORE_SITES])
    dxa = S.mask_zeros_as_missing(dxa, F.DXA_ZERO_IS_MISSING)
    chol = chol.reindex(singles.index)
    vat = dxa[F.EXPORT_VAT_VOLUME].reindex(singles.index)
    tscore = S.lowest_across_sites(
        dxa, F.EXPORT_BMD_TSCORE_SITES
    ).reindex(singles.index)

    covariates = pd.DataFrame({
        "sex": singles[F.SEX].astype("Int64"),
        "age": singles[F.AGE],
    })
    unexpected = set(covariates["sex"].dropna().unique()) - {0, 1}
    if unexpected:
        raise ValueError(f"unexpected sex coding in {F.SEX}: {unexpected}")

    print(f"labelling traits (quantitative mode: {config.QUANT_LABEL_MODE})")
    labels = build_labels(singles, icd, selfrep, meds, chol, vat, tscore)
    for trait in config.TRAITS:
        write_cohorts(trait, labels[trait], covariates)


if __name__ == "__main__":
    main()

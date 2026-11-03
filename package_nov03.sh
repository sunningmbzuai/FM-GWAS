#!/usr/bin/env bash
# Package the ning_nov03.sh results for Ning: three folders plus a README,
# zipped to ukbb_replication_hpp1103.zip under UKBV_OUT_ROOT.
#
#   bash package_nov03.sh
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${HERE}/paths.sh"
PHENO_PY="${CONDA_ROOT}/envs/microbiome/bin/python3"

NAME="ukbb_replication_hpp1103"
STAGE="${OUT_ROOT}/${NAME}"
ZIP="${OUT_ROOT}/${NAME}.zip"
ASSOC="${OUT_ROOT}/assoc_results_fmgwas_89aba56_all60_hpp1103"
CASES="${OUT_ROOT}/case_breakdown_hpp_1103"
RESULT_FILES=(
    ukbb_replication.tsv ukbb_replication.rates.tsv ukbb_replication.per_embedding.tsv
    ukbb_replication_bonferroni.tsv ukbb_replication_bonferroni.rates.tsv
    ukbb_replication_bonferroni.per_embedding.tsv
    ukbb_replication_fdr_only.tsv ukbb_replication_fdr_only.rates.tsv
    ukbb_replication_fdr_only.per_embedding.tsv
    paper_vs_hpp1103.tsv hpp1102_vs_hpp1103.tsv
)

copy() { [[ -f "$1" ]] || { echo "missing: $1" >&2; exit 1; }; cp "$1" "$2"; }

rm -rf "$STAGE" "$ZIP"
mkdir -p "$STAGE/1_scheme1_icd_selfreport" "$STAGE/2_scheme2_icd_only" "$STAGE/3_case_rates"

for f in "${RESULT_FILES[@]}"; do
    copy "${ASSOC}_icd_selfrep/$f" "$STAGE/1_scheme1_icd_selfreport/$f"
    copy "${ASSOC}_icd_only/$f" "$STAGE/2_scheme2_icd_only/$f"
done
for s in icd_selfrep icd_only; do
    tag="scheme1"; [[ "$s" == icd_only ]] && tag="scheme2"
    for f in case_rates case_classes excluded_classes; do
        copy "${CASES}_${s}/$f.tsv" "$STAGE/3_case_rates/${f}_${tag}.tsv"
    done
done

cat > "$STAGE/README.txt" <<'EOF'
UKB replication of the HPP disease genes, rerun 3 November 2026
================================================================

Case and control definitions
  Controls are the "control" category only: participants with no diagnosis,
  no relevant medication and a measurement below the cutoff (blood pressure,
  total cholesterol, HbA1c or field 2443 = no, T-score > -1.0). Every other
  non-case (medication only, lab or blood-pressure cutoff, T-score) is
  excluded. Cases:

  trait           scheme 1                     scheme 2
  Hypertension    ICD-10 I10-I15 + self-report ICD-10 I10-I15
  Hyperlipidemia  ICD-10 E78 + self-report     ICD-10 E78
  Osteoporosis    ICD-10 M80-M81 + self-report ICD-10 M80-M81
  T2D             ICD-10 E11                   ICD-10 E11

  Targets: the 67 hypertension, hyperlipidemia, osteoporosis and T2D rows of
  novel_gene_FDR_1026_no_height.csv (3 Bonferroni, 64 FDR-only). Obesity was
  not rerun.

Thresholds
  Genome-wide: p < 2.5e-6.
  Targeted: 0.05 / number of targets in the table. ukbb_replication.tsv uses
  0.05/67 = 7.5e-4 (all rows). The Bonferroni and FDR-only tables are scored
  separately, at 0.05/3 and 0.05/64.
  ukbb_p_min is the lowest p-value over the 60 embedding types.

1_scheme1_icd_selfreport/ and 2_scheme2_icd_only/ (same files in each)
  ukbb_replication.tsv                  per-gene results, all 67 rows
  ukbb_replication_bonferroni.tsv       the 3 Bonferroni rows
  ukbb_replication_fdr_only.tsv         the 64 FDR-only rows
  *.rates.tsv                           replication counts and rates, with the
                                        thresholds used
  *.per_embedding.tsv                   p-value for every embedding type
  paper_vs_hpp1103.tsv                  each gene under the original paper
                                        definitions and under this scheme
  hpp1102_vs_hpp1103.tsv                each gene in the 2 November run and
                                        under this scheme (osteoporosis from
                                        the diagnosis-only run)

3_case_rates/
  case_rates_scheme{1,2}.tsv            UKB vs HPP sample sizes and case rates
  case_classes_scheme{1,2}.tsv          how each UKB case group is made up
  excluded_classes_scheme{1,2}.tsv      participants excluded relative to the
                                        original definitions, by category
EOF

(cd "$OUT_ROOT" && "$PHENO_PY" -m zipfile -c "$ZIP" "$NAME")
echo "wrote $ZIP"
"$PHENO_PY" -m zipfile -l "$ZIP"

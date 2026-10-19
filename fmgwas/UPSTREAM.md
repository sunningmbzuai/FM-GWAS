# Vendored FM-GWAS association script

`insample_hpp_AG.py` is a byte-for-byte copy of Ning Sun's script from
https://github.com/sunningmbzuai/FM-GWAS, branch `main`, commit `89aba56`
(2026-10-19, "Update command syntax in README"; the script itself last changed
in `17d451b`, "update association test with LRT score for binary labels").

    sha256  7a47698ae697beaf37409b35eb7d5f9b54fd42d962bf1db48983f1c0fc562e19

Do not edit it. Everything this pipeline does differently (deduplicated
embeddings, torch-free feature tables, parallel ranks, target-gene lists) sits
outside it, so a UKBB p-value is exactly what Ning's code returns for the same
cohort and features. `tests/test_fmgwas_vendored.py` fails if the file drifts
from the hash above.

It replaces the copy in the FM-GWAS checkout that `run_assoc.sh` used to patch
in place (`fix_assoc_feature_tables.py`, `fix_assoc_pca95.py`,
`assoc_rank_capped.py`, all removed). The PCA rank problem those patches
guarded against cannot occur here: gene PCA is `variance:0.95` on an exact
eigendecomposition, which stops at the matrix rank, and both tests drop added
directions below the shared singular-value cutoff.

To update: copy the new upstream file over this one, update the commit and
hash here and in the test, and give `config.ASSOC_DIR` a new name if the
results change. The script refuses an `--out_dir` written with different
arguments or a different `VERSION` string, but upstream keeps `VERSION` when it
believes results are unchanged, so it does not catch every code change.

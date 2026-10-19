"""What one assoc cell asks FM-GWAS for: the settings Ning specified.

fmgwas/insample_hpp_AG.py is run unmodified; this module is the single place
that turns a trait into its command-line settings, so run_assoc.sh and the
summaries agree on which test and which embeddings a cell used.

Ning, 2026-10-19 (README of FM-GWAS 89aba56 and her email):
  * gene PCA      variance:0.95
  * covariates    age,sex for All; age for F and M. The script drops sex by
                  itself when the cohort is single-sex, so age,sex is passed
                  for every cell.
  * disease       --tests LRT on the concat_tss* embeddings
  * quantitative  the partial F test (LRT needs a 0/1 label) on every
                  embedding the feature file supports, as before

    python assoc_settings.py args <trait>     one CLI token per line
    python assoc_settings.py p_column <trait>
"""
import sys
from itertools import combinations

import config

GENE_PCA = "variance:0.95"
COVARIATES = "age,sex"

# Labelled by build_cohorts.py: continuous for these three (config.QUANT_LABEL_MODE),
# 0/1 for every other trait.
QUANTITATIVE_TRAITS = ("Cholesterol", "Height", "VAT")

# The script's own track order (insample_hpp_AG.TRACKS). Embedding names are
# built from it, so they match the names Ning's HPP results carry.
TRACKS = ("cage", "dnase", "atac", "rnaseq")


def is_quantitative(trait: str) -> bool:
    if trait not in config.TRAITS:
        raise KeyError(f"unknown trait {trait!r}; expected one of {config.TRAITS}")
    return trait in QUANTITATIVE_TRAITS


def embeddings(modes: tuple[str, ...], tss_options: tuple[bool, ...]) -> list[str]:
    """Names in the order insample_hpp_AG.available_embeddings lists them."""
    names = []
    for tss in tss_options:
        prefix = "tss_" if tss else ""
        for size in range(1, len(TRACKS) + 1):
            for subset in combinations(TRACKS, size):
                names.extend(f"{mode}_{prefix}{'_'.join(subset)}" for mode in modes)
    return names


# Every track subset, TSS-500 window, haplotype halves concatenated: 15 views.
CONCAT_TSS = embeddings(("concat",), (True,))
# Everything a four-track feature file supports: 60 views, as in d17eae23a6.
ALL_VIEWS = embeddings(("concat", "add"), (False, True))


def stat_test(trait: str) -> str:
    return "F" if is_quantitative(trait) else "LRT"


def p_column(trait: str) -> str:
    """The per-gene result column a cell's replication is judged on."""
    return "p_F" if is_quantitative(trait) else "p_LRT"


def embed_types(trait: str) -> list[str]:
    return ALL_VIEWS if is_quantitative(trait) else CONCAT_TSS


def cli_args(trait: str) -> list[str]:
    """Settings flags for `insample_hpp_AG.py run`; paths are the caller's."""
    return ["--covariates", COVARIATES,
            "--gene_pca", GENE_PCA,
            "--tests", stat_test(trait),
            "--embed_type", ",".join(embed_types(trait))]


def main(argv: list[str]) -> int:
    if len(argv) == 2 and argv[0] == "args":
        print("\n".join(cli_args(argv[1])))
        return 0
    if len(argv) == 2 and argv[0] == "p_column":
        print(p_column(argv[1]))
        return 0
    print(__doc__, file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

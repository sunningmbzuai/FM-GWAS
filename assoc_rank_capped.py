#!/usr/bin/env python
"""Run FM-GWAS's insample_hpp_AG.py with rank-capped PCA.

    python assoc_rank_capped.py /path/to/insample_hpp_AG.py <its own args...>

Installs rank_capped_pca.RankCappedPCA in place of sklearn's PCA, then runs the
upstream script as __main__ with its own arguments. The upstream file is not
edited, so the fix cannot be lost when FM-GWAS is re-staged, and run_assoc.sh
is the one place that decides whether it applies.
"""
import os
import runpy
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import rank_capped_pca  # noqa: E402


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    script = os.path.abspath(sys.argv[1])
    if not os.path.isfile(script):
        raise SystemExit(f"not a file: {script}")
    rank_capped_pca.install()
    sys.argv = [script] + sys.argv[2:]
    sys.path.insert(0, os.path.dirname(script))
    runpy.run_path(script, run_name="__main__")


if __name__ == "__main__":
    main()

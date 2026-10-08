#!/usr/bin/env bash
# One-time: fetch the GRCh37 GENCODE GTF this pipeline needs for gene coords.
# gencode.v19 is the last release natively on GRCh37.
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/paths.sh"
OUT="$GRCH37_GTF"
mkdir -p "$(dirname "$OUT")"
if [[ -s "$OUT" ]]; then
  echo "[setup_grch37_gtf] already present: $OUT"
  exit 0
fi
URL="https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_19/gencode.v19.annotation.gtf.gz"
echo "[setup_grch37_gtf] downloading $URL"
curl -L "$URL" -o "${OUT}.gz"
gunzip "${OUT}.gz"
echo "[setup_grch37_gtf] done: $OUT"

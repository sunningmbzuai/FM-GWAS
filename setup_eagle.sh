#!/usr/bin/env bash
# One-time: fetch Eagle v2.4.1 + its hg19 genetic map for UKB phasing.
# Mirrors setup_grch37_gtf.sh's pattern. If the run node can't exec off
# UKBV_EAGLE_DIR (e.g. it's FUSE-mounted there), copy the binary to real local
# disk on that node before running.
set -uo pipefail  # no -e: every failure point below is checked explicitly for a clear message

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/paths.sh"
OUT_DIR="$EAGLE_LOCAL_DIR"
mkdir -p "$OUT_DIR"

log() { printf '[setup_eagle] %s\n' "$*" >&2; }

if [[ -x "$OUT_DIR/eagle" && -s "$OUT_DIR/tables/genetic_map_hg19_withX.txt.gz" ]]; then
  log "already present: $OUT_DIR"
  exit 0
fi

TARBALL_URL="https://alkesgroup.broadinstitute.org/Eagle/downloads/Eagle_v2.4.1.tar.gz"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

log "downloading $TARBALL_URL"
HTTP_CODE=$(curl -sS -L --connect-timeout 15 --max-time 300 \
    -w '%{http_code}' -o "$TMP/eagle.tar.gz" "$TARBALL_URL")
CURL_RC=$?

if (( CURL_RC != 0 )); then
    log "FAIL: curl exit code $CURL_RC (rc 6/7/28 = DNS/connect/timeout -- this compute node"
    log "      likely has no outbound internet, which is common on HPC clusters)."
    log "MANUAL FALLBACK: download on a machine with internet access, then scp to the cluster:"
    log "  curl -L -o Eagle_v2.4.1.tar.gz $TARBALL_URL"
    log "  scp Eagle_v2.4.1.tar.gz <cluster>:/tmp/"
    log "  # then on the cluster:"
    log "  mkdir -p $OUT_DIR/tables"
    log "  tar -xzf /tmp/Eagle_v2.4.1.tar.gz -C /tmp"
    log "  cp /tmp/Eagle_v2.4.1/eagle $OUT_DIR/eagle && chmod +x $OUT_DIR/eagle"
    log "  cp -r /tmp/Eagle_v2.4.1/tables/. $OUT_DIR/tables/"
    exit 1
fi
if [[ "$HTTP_CODE" != "200" ]]; then
    log "FAIL: HTTP $HTTP_CODE fetching $TARBALL_URL (download saved to $TMP/eagle.tar.gz for inspection, not cleaned up)"
    trap - EXIT
    exit 1
fi
if [[ ! -s "$TMP/eagle.tar.gz" ]]; then
    log "FAIL: downloaded file is empty despite HTTP 200 -- $TARBALL_URL"
    exit 1
fi

log "extracting"
if ! tar -xzf "$TMP/eagle.tar.gz" -C "$TMP" 2>"$TMP/tar.err"; then
    log "FAIL: tar extraction failed:"
    sed 's/^/  /' "$TMP/tar.err" >&2
    exit 1
fi

SRC_DIR="$TMP/Eagle_v2.4.1"
[[ -f "$SRC_DIR/eagle" ]] || { log "FAIL: $SRC_DIR/eagle not found post-extract -- tarball layout may have changed"; exit 1; }
[[ -d "$SRC_DIR/tables" ]] || { log "FAIL: $SRC_DIR/tables not found post-extract"; exit 1; }

cp "$SRC_DIR/eagle" "$OUT_DIR/eagle"
chmod +x "$OUT_DIR/eagle"
mkdir -p "$OUT_DIR/tables"
cp -r "$SRC_DIR/tables/." "$OUT_DIR/tables/"

if ! "$OUT_DIR/eagle" --help >/dev/null 2>"$TMP/eagle.err"; then
    log "FAIL: $OUT_DIR/eagle copied but won't execute (FUSE-mounted paths often fail to exec):"
    sed 's/^/  /' "$TMP/eagle.err" >&2
    exit 1
fi

log "done: $OUT_DIR"

#!/usr/bin/env bash
# Eagle-phase UKB's QC-passed ukb_snp_all array samples.
#
# Uses Eagle's NATIVE --bfile mode (population-based phasing straight off a
# PLINK bed/bim/fam triple, no reference panel) instead of the HPP pipeline's
# VCF+bcftools-norm chain. That chain exists to clean up low-coverage-WGS VCFs (multi-allelic
# splitting, non-SNP filtering, REF-mismatch checks); UKB array genotypes are
# already clean SNP/biallelic PLINK data, so none of that buys anything here,
# and it was costing a slow whole-cohort plink2->VCF export for no reason.
# --bfile is literally the mode UK Biobank's own in-house phasing used.
#
#   ukb_snp_all bfile
#     -> plink2 --keep <QC> --chr 1-22 --make-bed   (subset, binary->binary, fast)
#     -> eagle --bfile=... --chrom=$c  (per autosome, JOBS-parallel)
#          -> chr$c.haps.gz + chr$c.sample  (Oxford format; --bfile mode has no VCF output)
#     -> plink2 --haps/--sample --export vcf  (all 22 chroms in parallel; per
#          chrom, so downstream tooling
#          that expects a phased VCF -- build_gene_sequences.py -- needs no changes)
#     -> bcftools norm --check-ref ws  (orient REF/ALT to the real reference base)
#     -> bcftools concat -> config.PHASED_VCF
#
# NOT `bcftools convert --hapsample2vcf`: it derives CHROM by parsing the haps
# ID column as CHROM:POS_A0_A1, and Eagle writes a plain rsID there, so it
# fails outright ("Could not determine CHROM in the second column"). plink2
# reads Oxford haps/sample natively, preserves phase, and --output-chr 26 gives
# the bare 1..22 contig naming build_gene_sequences.py needs (it strips "chr"
# off GTF contigs, so a chr-prefixed VCF would silently fetch zero variants).
#
# The bcftools norm pass is not cosmetic: a PLINK bfile carries no true REF, so
# whichever allele lands in the VCF REF column is arbitrary w.r.t. the genome.
# build_gene_sequences.py writes rec.ref at het sites onto a fasta-derived
# ref_seq, so an unoriented REF corrupts every het/hom-alt base at that site.
#
# No multiallelic_blindspots.bed is produced: PLINK bim rows are inherently
# biallelic, so there's nothing analogous to drop.
#
# Run with the "wasp_env" conda env (pysam not needed here, but bcftools/tabix
# are: they are taken from envs/wasp_env/bin/ under UKBV_CONDA_ROOT).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"
PY="${PY:-python3}"

# ---- pull paths out of config.py so there is one source of truth ----------
mapfile -t CFG < <("$PY" - <<'PYEOF'
import config
print(config.UKB_BFILE)
print(config.QC_FAM)
print(config.OUT_ROOT)
print(config.PHASING_DIR)
print(config.PHASED_VCF)
print(config.PHASING_BLINDSPOTS_BED)
print(config.UKB_QC_SUBSET)
print(config.EAGLE_BIN)
print(config.EAGLE_GENETIC_MAP)
print(config.PLINK2)
print(config.CONDA_ROOT)
print(config.GRCH37_FASTA)
PYEOF
)
UKB_BFILE="${CFG[0]}"; QC_FAM="${CFG[1]}"; OUT_ROOT="${CFG[2]}"; PHASING_DIR="${CFG[3]}"
PHASED_VCF="${CFG[4]}"; BLINDSPOTS_BED="${CFG[5]}"; SUBSET="${CFG[6]}"
EAGLE_BIN="${CFG[7]}"; EAGLE_GENETIC_MAP="${CFG[8]}"; PLINK2="${CFG[9]}"; CONDA_ROOT="${CFG[10]}"
GRCH37_FASTA="${CFG[11]}"
# config.py raises (and prints nothing) when a UKBV_* variable is unset.
(( ${#CFG[@]} == 12 )) || { echo "config.py could not resolve the phasing paths -- set the UKBV_* variables (paths.env.example)" >&2; exit 1; }

BCFTOOLS="${CONDA_ROOT}/envs/wasp_env/bin/bcftools"
TABIX="${CONDA_ROOT}/envs/wasp_env/bin/tabix"

JOBS="${JOBS:-4}"                       # concurrent Eagle jobs (STEP 2)
THREADS_PER_JOB="${THREADS_PER_JOB:-8}"  # threads per Eagle job
# STEP 3 is per-chrom independent, so all 22 can run at once. plink2 otherwise
# reserves half the node's RAM and every core per process, so both caps below
# are required, not tuning. 22 x 8 GiB fits the 385 GiB node seen here; lower
# CONVERT_JOBS if a shared network filesystem becomes the bottleneck.
CONVERT_JOBS="${CONVERT_JOBS:-22}"
PLINK_THREADS="${PLINK_THREADS:-2}"
PLINK_MEM_MB="${PLINK_MEM_MB:-8000}"
CONCAT_THREADS="${CONCAT_THREADS:-8}"   # bgzip compression threads for the final concat

log() { printf '[%s] %s\n' "$(date +'%F %T')" "$*" >&2; }
die() { log "ERROR: $*"; exit 1; }

[[ -x "$EAGLE_BIN" ]] || die "Eagle not staged: $EAGLE_BIN -- run ./setup_eagle.sh first"
[[ -r "$QC_FAM" ]]    || die "QC fam not readable: $QC_FAM"
[[ -x "$BCFTOOLS" ]]  || die "bcftools not found: $BCFTOOLS"
[[ -x "$TABIX" ]]     || die "tabix not found: $TABIX"
[[ -x "$PLINK2" ]]    || die "plink2 not found: $PLINK2"
[[ -r "$GRCH37_FASTA" ]] || die "reference fasta not readable: $GRCH37_FASTA"
mkdir -p "$OUT_ROOT" "$PHASING_DIR"

# bgzf_intact FILE -- true if FILE ends with the 28-byte BGZF EOF marker.
# A killed bgzip/bcftools leaves a nonempty but truncated .gz that every -s
# test happily accepts; this is the same check samtools uses, and it costs one
# 28-byte read instead of decompressing the file.
BGZF_EOF='1f8b08040000000000ff0600424302001b0003000000000000000000'
bgzf_intact() {
    [[ -s "$1" ]] || return 1
    local tailhex
    tailhex="$(tail -c 28 "$1" | od -An -tx1 -v | tr -d ' \n')"
    [[ "$tailhex" == "$BGZF_EOF" ]]
}

# retry N SLEEP LABEL CMD...  -- run CMD up to N times, sleeping SLEEP between
# attempts. Network (autofs/NFS) mounts can intermittently fail to
# resolve a binary under concurrent load ("plink2: No such file or directory"
# on one chrom while 21 others exec'd the same path fine), so a single transient
# lookup miss must not cost a whole run.
retry() {
    local n="$1" nap="$2" label="$3"; shift 3
    local i
    for (( i = 1; i <= n; i++ )); do
        "$@" && return 0
        (( i < n )) && { echo "  $label: attempt $i/$n failed, retrying in ${nap}s" >&2; sleep "$nap"; }
    done
    return 1
}

# run_per_chrom MAX_JOBS FN  -- run FN <c> for c in 1..22, MAX_JOBS at a time,
# and leave the chroms whose FN returned nonzero in the global FAILED array.
#
# NOTE: `FN "$c" || FAILED+=("$c") &` does NOT work -- the append runs in the
# background subshell and is invisible to this shell, which is how every Eagle
# failure got silently swallowed. Track pid -> chrom and `wait` each pid for its
# status. FIFO throttle (blocking wait on the oldest job) rather than a kill -0
# poll: bash 4.2 has no `wait -n`, and a reaped-but-unwaited child still answers
# kill -0, which would spin forever.
FAILED=()
run_per_chrom() {
    local max_jobs="$1" fn="$2"
    # Two positionally-matched plain arrays rather than an associative pid->chrom
    # map, so this works on any bash 3.2+ regardless of node.
    local pidq=() chromq=() c
    FAILED=()
    _reap_oldest() {
        local pid="${pidq[0]}" chrom="${chromq[0]}"
        pidq=("${pidq[@]:1}")
        chromq=("${chromq[@]:1}")
        wait "$pid" || FAILED+=("$chrom")
    }
    for c in $(seq 1 22); do
        "$fn" "$c" &
        pidq+=("$!")
        chromq+=("$c")
        while (( ${#pidq[@]} >= max_jobs )); do _reap_oldest; done
    done
    while (( ${#pidq[@]} > 0 )); do _reap_oldest; done
}

# ---------------------------------------------------------------------------
# STEP 1 - subset ukb_snp_all to QC-passed samples, autosomes only
#   (binary bed/bim/fam -> bed/bim/fam, no VCF text conversion)
# ---------------------------------------------------------------------------
if [[ -s "${SUBSET}.bed" ]]; then
    log "QC-sample subset bfile exists, skip: ${SUBSET}.bed"
else
    log "=== subsetting $UKB_BFILE to QC-passed samples, chr1-22 ==="
    KEEP="$OUT_ROOT/qc_keep.txt"
    awk '{print $1, $2}' "$QC_FAM" > "$KEEP"
    log "  QC-passed samples: $(wc -l < "$KEEP")"

    "$PLINK2" --bfile "$UKB_BFILE" \
        --keep "$KEEP" \
        --chr 1-22 \
        --make-bed \
        --out "$SUBSET"
    log "  wrote ${SUBSET}.{bed,bim,fam}"
fi

# ---------------------------------------------------------------------------
# STEP 2 - Eagle, per autosome, straight off the subset bfile
# ---------------------------------------------------------------------------
log "=== Eagle phasing (--bfile mode, ${JOBS} jobs x ${THREADS_PER_JOB} threads) ==="

phase_one() {
    local c="$1"
    local prefix="$PHASING_DIR/chr${c}.phased"
    # Skip only on a COMPLETE prior run: a killed Eagle leaves a nonempty but
    # truncated haps.gz, which a plain -s test would happily accept.
    if [[ -s "${prefix}.haps.gz" && -s "${prefix}.sample" ]]; then
        if gzip -t "${prefix}.haps.gz" 2>/dev/null; then
            echo "  chr$c: exists, skip" >&2; return 0
        fi
        echo "  chr$c: existing haps.gz is truncated -- re-phasing" >&2
        rm -f "${prefix}.haps.gz" "${prefix}.sample"
    fi

    echo "  chr$c: phasing" >&2
    "$EAGLE_BIN" \
        --bfile="$SUBSET" \
        --chrom="$c" \
        --geneticMapFile="$EAGLE_GENETIC_MAP" \
        --outPrefix="$prefix" \
        --numThreads="$THREADS_PER_JOB" \
        > "$PHASING_DIR/chr${c}.eagle.log" 2>&1 \
        || { echo "  chr$c: EAGLE FAILED -> $PHASING_DIR/chr${c}.eagle.log" >&2; return 1; }
}

run_per_chrom "$JOBS" phase_one
(( ${#FAILED[@]} == 0 )) || die "Eagle failed on chrom(s): ${FAILED[*]} -- see $PHASING_DIR/chr*.eagle.log"

# ---------------------------------------------------------------------------
# STEP 3 - haps/sample -> VCF per chrom, then concat
# ---------------------------------------------------------------------------
log "=== converting haps/sample -> VCF (plink2), orienting REF to fasta ==="
log "    ${CONVERT_JOBS} chroms at a time x ${PLINK_THREADS} threads, ${PLINK_MEM_MB} MiB each"

convert_one() {
    local c="$1"
    local prefix="$PHASING_DIR/chr${c}.phased"
    local raw="${prefix}.rawref.vcf.gz"   # plink2 export, REF still arbitrary
    local out="${prefix}.vcf.gz"          # REF oriented to GRCh37
    # Integrity-checked skip, not just -s: a conversion killed mid-write leaves
    # a truncated VCF that would be silently concatenated as partial data.
    if [[ -s "$out" ]]; then
        if bgzf_intact "$out" && [[ -s "${out}.tbi" ]]; then
            echo "  chr$c: VCF exists, skip" >&2; return 0
        fi
        echo "  chr$c: existing VCF is truncated/unindexed -- rebuilding" >&2
        rm -f "$out" "${out}.tbi"
    fi

    echo "  chr$c: haps -> VCF" >&2
    # --memory/--threads are mandatory here, not tuning: plink2 defaults to
    # reserving half the node's RAM and using every core, which 22 concurrent
    # jobs cannot share.
    # ref-last, not ref-first: Eagle's allele0 is PLINK's A1 (the counted,
    # usually minor allele), so allele1 is the REF-ish one. Measured on chr22:
    # ref-last needs 1495/9752 REF swaps below, ref-first 8204/9752.
    # id-paste=iid: Eagle's .sample carries ID_1/ID_2 both set to the eid, and
    # downstream matches participant_id against bare sample names, not plink's
    # FID_IID default.
    _plink_export() {
        "$PLINK2" --haps "${prefix}.haps.gz" ref-last \
                  --sample "${prefix}.sample" \
                  --output-chr 26 \
                  --threads "$PLINK_THREADS" \
                  --memory "$PLINK_MEM_MB" \
                  --export vcf-4.2 bgz id-paste=iid \
                  --out "${prefix}.rawref" \
            > "$PHASING_DIR/chr${c}.plink2.log" 2>&1
    }
    retry 3 10 "chr$c plink2" _plink_export \
        || { echo "  chr$c: PLINK2 EXPORT FAILED -> $PHASING_DIR/chr${c}.plink2.log" >&2; return 1; }
    [[ -s "$raw" ]] || { echo "  chr$c: plink2 wrote no $raw" >&2; return 1; }

    # -c ws = warn + swap REF/ALT (and the genotypes with them) where the fasta
    # base is the recorded ALT. Phase is preserved across the swap.
    echo "  chr$c: orienting REF against $(basename "$GRCH37_FASTA")" >&2
    _norm_ref() {
        "$BCFTOOLS" norm --check-ref ws -f "$GRCH37_FASTA" "$raw" -Oz -o "$out" \
            2> "$PHASING_DIR/chr${c}.norm.log"
    }
    retry 3 10 "chr$c norm" _norm_ref \
        || { echo "  chr$c: BCFTOOLS NORM FAILED -> $PHASING_DIR/chr${c}.norm.log" >&2; rm -f "$out"; return 1; }
    bgzf_intact "$out" \
        || { echo "  chr$c: norm output is truncated (disk full? job killed?)" >&2; rm -f "$out"; return 1; }
    "$TABIX" -f -p vcf "$out" || { echo "  chr$c: tabix failed" >&2; rm -f "$out"; return 1; }
    rm -f "$raw" "${raw}.tbi"
    grep -h 'REF/ALT total/modified/added\|Lines *total' \
        "$PHASING_DIR/chr${c}.norm.log" | sed "s/^/  chr${c}: /" >&2 || true
}

# Preflight every chrom's inputs before spending any conversion time, and hard-
# fail rather than skipping: a silently dropped chromosome yields a partial
# concat that looks like successful whole-genome phasing.
for c in $(seq 1 22); do
    prefix="$PHASING_DIR/chr${c}.phased"
    [[ -s "${prefix}.haps.gz" && -s "${prefix}.sample" ]] \
        || die "chr$c: missing ${prefix}.haps.gz/.sample -- STEP 2 did not finish for this chrom"
    gzip -t "${prefix}.haps.gz" 2>/dev/null \
        || die "chr$c: ${prefix}.haps.gz is truncated -- delete it and re-run to re-phase"
done

run_per_chrom "$CONVERT_JOBS" convert_one
(( ${#FAILED[@]} == 0 )) \
    || die "conversion failed on chrom(s): ${FAILED[*]} -- see $PHASING_DIR/chr*.{plink2,norm}.log"

# Merge list built here, serially, NOT appended from inside the parallel jobs --
# bcftools concat requires chromosome order, and concurrent appends would
# interleave arbitrarily.
MERGE_LIST="$PHASING_DIR/merge_list.txt"
: > "$MERGE_LIST"
for c in $(seq 1 22); do
    out="$PHASING_DIR/chr${c}.phased.vcf.gz"
    [[ -s "$out" ]] || die "chr$c: $out missing after a successful conversion pass"
    bgzf_intact "$out" \
        || die "chr$c: $out is truncated -- delete it and re-run to rebuild that chrom"
    echo "$out" >> "$MERGE_LIST"
done

N_MERGE=$(wc -l < "$MERGE_LIST")
(( N_MERGE == 22 )) || die "merge list has $N_MERGE chromosomes, expected 22"
# Integrity-checked skip, same shape as the per-chrom guards: an intact bgzf
# with an index that is no older than every input chrom is a finished concat.
# The -nt test matters -- if one chrom got deleted and re-phased, its VCF is
# newer than the concat, and silently keeping the stale concat would drop the
# re-phased calls.
concat_current() {
    [[ -s "$PHASED_VCF" && -s "${PHASED_VCF}.tbi" ]] || return 1
    bgzf_intact "$PHASED_VCF" || return 1
    local c
    for c in $(seq 1 22); do
        [[ "${PHASED_VCF}.tbi" -nt "$PHASING_DIR/chr${c}.phased.vcf.gz" ]] || return 1
    done
    return 0
}

if concat_current; then
    log "=== concat exists and is newer than all 22 chroms, skip ==="
else
    log "=== concatenating $N_MERGE chromosomes ==="
    "$BCFTOOLS" concat --file-list "$MERGE_LIST" --threads "$CONCAT_THREADS" -Oz -o "$PHASED_VCF" \
        || die "bcftools concat failed"
    bgzf_intact "$PHASED_VCF" || die "$PHASED_VCF is truncated after concat"
    "$TABIX" -f -p vcf "$PHASED_VCF" || die "tabix failed on $PHASED_VCF"
fi

# PLINK bim rows are inherently biallelic -- nothing to log as a blind spot,
# but build_gene_sequences.py's load_blindspots() expects the path to exist.
: > "$BLINDSPOTS_BED"

# ---------------------------------------------------------------------------
# STEP 4 - verify
# ---------------------------------------------------------------------------
UNPH=$("$BCFTOOLS" query -f '[%GT\n]' "$PHASED_VCF" | grep -E '^[0-9]+/[1-9][0-9]*$' | wc -l || true)
log "=== Done ==="
log "Output        : $PHASED_VCF"
log "Sites         : $("$BCFTOOLS" index -n "$PHASED_VCF")"
log "Samples       : $("$BCFTOOLS" query -l "$PHASED_VCF" | wc -l)"
log "Unphased hets : $UNPH   (expect 0)"

FIRST_CONTIG=$("$BCFTOOLS" index -s "$PHASED_VCF" | head -1 | cut -f1)
[[ "$FIRST_CONTIG" != chr* ]] \
    || die "VCF contigs are chr-prefixed ('$FIRST_CONTIG'); build_gene_sequences.py strips 'chr' off GTF contigs and would fetch zero variants"
log "Contig naming : $FIRST_CONTIG (bare, as build_gene_sequences.py expects)"
# "REF/ALT total/modified/added:  9752/1495/0" -> total, modified
read -r REF_TOTAL REF_SWAPPED < <(
    grep -h 'REF/ALT total/modified/added' "$PHASING_DIR"/chr*.norm.log 2>/dev/null \
    | awk -F'[:/[:space:]]+' '{t+=$(NF-2); m+=$(NF-1)} END{print t+0, m+0}')
# "Lines   total/split/realigned/skipped:  9752/0/1/0" -> skipped (unfixable REF)
SKIPPED=$(grep -h 'Lines .*total/split/realigned/skipped' "$PHASING_DIR"/chr*.norm.log 2>/dev/null \
          | awk -F'[:/[:space:]]+' '{n+=$NF} END{print n+0}')
log "REF/ALT oriented  : ${REF_SWAPPED}/${REF_TOTAL} sites swapped to match fasta"
log "  ~15% is expected (array A2 often isn't the reference base). Near 100%"
log "  would mean the --haps ref-last/ref-first orientation is inverted."
log "Unfixable REF     : $SKIPPED sites skipped (fasta base matched neither allele)"
if (( REF_TOTAL > 0 && SKIPPED * 100 > REF_TOTAL )); then
    die "$SKIPPED/$REF_TOTAL sites (>1%) have a fasta base matching neither allele -- \
build/strand problem, not a phasing one; re-check validate_ukb_build.py before using this VCF"
fi

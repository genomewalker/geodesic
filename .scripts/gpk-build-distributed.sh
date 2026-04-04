#!/usr/bin/env bash
# Build a taxonomy-optimised multi-pack archive across N machines via SSH.
#
# Input is partitioned by GENUS (LPT bin-packing) so all genomes of a given
# genus land on the same node.  Each node builds with --taxon-group, producing
# per-genus shards internally.
#
# Output is a DIRECTORY of .gpk parts — no merge step required.
# Pass the output directory to geodesic --pack and it will be opened as a
# MultiPackReader automatically.
#
# Usage:
#   gpk-build-distributed.sh <input.tsv> <output_dir> [options] <host1> [host2 ...]
#
# Options:
#   -z <level>   zstd compression level (default: 3)
#   -t <n>       I/O threads per node (default: 24)
#   -r <rank>    Taxonomy rank for grouping: g=genus (default), f=family
#   --sketch     Compute OPH sketches inline (writes SKCH section)
#   --2bit       Enable 2-bit nucleotide packing (~1.5-2x extra compression)
#
# All paths must be on shared NFS visible from all hosts.
# Requires passwordless SSH to all hosts.

set -euo pipefail

GENOPACK=/maps/projects/fernandezguerra/apps/repos/genopack/build/genopack
PYTHON=${PYTHON:-python3}

TSV="${1:?Usage: $0 <input.tsv> <output_dir> [opts] <host1> [host2 ...]}"
OUTPUT_DIR="${2:?}"
shift 2

# Parse options
ZSTD_LEVEL=3
IO_THREADS=24
TAXON_RANK=g
EXTRA_FLAGS="--no-hnsw --no-cidx --taxon-group --mem-delta"

while [[ "${1:-}" == -* ]]; do
    case "$1" in
        -z)          ZSTD_LEVEL="$2"; shift 2 ;;
        -t)          IO_THREADS="$2"; shift 2 ;;
        -r)          TAXON_RANK="$2"; shift 2 ;;
        --sketch)    EXTRA_FLAGS="$EXTRA_FLAGS --sketch"; shift ;;
        --2bit)      EXTRA_FLAGS="$EXTRA_FLAGS --2bit"; shift ;;
        --mem-delta) shift ;;  # already default
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

HOSTS=("$@")
N=${#HOSTS[@]}
[[ $N -gt 0 ]] || { echo "ERROR: no hosts specified" >&2; exit 1; }

mkdir -p "$OUTPUT_DIR"
PARTS_DIR="${OUTPUT_DIR}/_partitions"
LOG_DIR="${OUTPUT_DIR}/_logs"
mkdir -p "$PARTS_DIR" "$LOG_DIR"

echo "[$(date)] Input: $TSV"
echo "[$(date)] Output: $OUTPUT_DIR"
echo "[$(date)] Hosts: ${HOSTS[*]}"
echo "[$(date)] Partitioning by ${TAXON_RANK}__ across $N nodes..."

# ---------------------------------------------------------------------------
# Genus-balanced partitioning
# ---------------------------------------------------------------------------
$PYTHON - "$TSV" "$N" "$PARTS_DIR" "$TAXON_RANK" <<'EOF'
import sys, os, collections

tsv_path, n_str, parts_dir, rank = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
N = int(n_str)
rank_prefix = rank + "__"

print(f"[partition] Reading taxonomy from {tsv_path}...")

with open(tsv_path) as f:
    header = f.readline().rstrip("\n")
    rows = [line for line in f]

def rank_key(line):
    parts = line.split("\t", 2)
    if len(parts) < 2:
        return "__unknown__"
    tax = parts[1]
    idx = tax.find(";" + rank_prefix)
    if idx < 0:
        idx = tax.find(rank_prefix)
    if idx < 0:
        return "__unknown__"
    start = idx + 1 if tax[idx] == ";" else idx
    end = tax.find(";", start)
    return tax[start:end] if end >= 0 else tax[start:].strip()

genus_rows = collections.defaultdict(list)
for row in rows:
    genus_rows[rank_key(row)].append(row)

# LPT bin-packing
node_genera = [[] for _ in range(N)]
node_counts = [0] * N

for genus, genus_list in sorted(genus_rows.items(), key=lambda x: -len(x[1])):
    target = min(range(N), key=lambda i: node_counts[i])
    node_genera[target].append((genus, genus_list))
    node_counts[target] += len(genus_list)

for i, genera in enumerate(node_genera):
    path = os.path.join(parts_dir, f"part_{i}.tsv")
    with open(path, "w") as f:
        f.write(header + "\n")
        all_rows = [row for _, rows in sorted(genera) for row in rows]
        f.writelines(all_rows)
    print(f"[partition] part_{i}: {node_counts[i]} genomes, {len(genera)} genera → {path}")

print(f"[partition] Load balance: {node_counts}")
EOF

echo "[$(date)] Partition done. Launching builds on ${N} nodes..."

# ---------------------------------------------------------------------------
# Launch parallel builds — each node writes directly into OUTPUT_DIR
# ---------------------------------------------------------------------------
PART_GPKS=()
PIDS=()

for i in "${!HOSTS[@]}"; do
    host="${HOSTS[$i]}"
    slice="${PARTS_DIR}/part_${i}.tsv"
    part_gpk="${OUTPUT_DIR}/part_${i}.gpk"
    log="${LOG_DIR}/part_${i}.log"
    PART_GPKS+=("$part_gpk")

    n_genomes=$(( $(wc -l < "$slice") - 1 ))
    echo "[$(date)] Launching part $i on $host ($n_genomes genomes) → $part_gpk"

    ssh -n "$host" "nohup bash -c '$GENOPACK build \
        -i $slice \
        -o $part_gpk \
        -t $IO_THREADS \
        -z $ZSTD_LEVEL \
        --taxon-rank $TAXON_RANK \
        $EXTRA_FLAGS \
        -v > $log 2>&1' </dev/null &" &
    PIDS+=($!)
done

echo "[$(date)] All $N builds launched. Polling every 60s..."

# ---------------------------------------------------------------------------
# Wait for completion
# ---------------------------------------------------------------------------
while true; do
    done=0
    for i in "${!HOSTS[@]}"; do
        log="${LOG_DIR}/part_${i}.log"
        if [[ -f "$log" ]] && grep -q "genopack archive written" "$log" 2>/dev/null; then
            ((done++)) || true
        fi
    done
    echo "[$(date)] $done/$N parts complete"
    [[ $done -eq $N ]] && break
    for i in "${!HOSTS[@]}"; do
        log="${LOG_DIR}/part_${i}.log"
        if [[ -f "$log" ]] && grep -qE "terminate|runtime_error|ERROR|Segmentation" "$log" 2>/dev/null; then
            echo "[$(date)] ERROR in part $i (${HOSTS[$i]}), check $log" >&2
        fi
    done
    sleep 60
done

for pid in "${PIDS[@]}"; do wait "$pid" 2>/dev/null || true; done

# ---------------------------------------------------------------------------
# Validate parts
# ---------------------------------------------------------------------------
echo "[$(date)] All parts done. Validating..."
for gpk in "${PART_GPKS[@]}"; do
    echo "[$(date)] stat: $gpk"
    "$GENOPACK" stat "$gpk"
done

# ---------------------------------------------------------------------------
# Cleanup partition TSVs and logs
# ---------------------------------------------------------------------------
echo "[$(date)] Cleaning partition files..."
rm -rf "$PARTS_DIR" "$LOG_DIR"

echo "[$(date)] Done: $OUTPUT_DIR"
echo "[$(date)] Use: geodesic derep --pack $OUTPUT_DIR ..."

#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GEODESIC="${GEODESIC:-${SCRIPT_DIR}/../../build/geodesic}"
SKANI="${SKANI:-/maps/projects/fernandezguerra/apps/opt/conda/envs/bioinfo/bin/skani}"
DUCKDB="${DUCKDB:-/maps/projects/fernandezguerra/apps/opt/conda/envs/bioinfo/bin/duckdb}"
INPUT_TSV="${1:-${SCRIPT_DIR}/test_input.tsv}"
ANI_THRESHOLD="${2:-95.0}"
THREADS="${3:-4}"

for tool in "$GEODESIC" "$SKANI" "$DUCKDB"; do
    [[ -x "$tool" ]] || { echo "ERROR: not found or not executable: $tool"; exit 1; }
done
[[ -f "$INPUT_TSV" ]] || { echo "ERROR: input TSV not found: $INPUT_TSV"; exit 1; }

N_INPUT=$(tail -n +2 "$INPUT_TSV" | wc -l)
echo "Input: $N_INPUT genomes from $INPUT_TSV"
echo "ANI threshold: $ANI_THRESHOLD"

WORKDIR=$(mktemp -d /tmp/geodesic_inttest_XXXXXX)
trap 'rm -rf "$WORKDIR"' EXIT
echo "Working directory: $WORKDIR"

# --- Step 1: Run geodesic ---
echo ""
echo "==> Step 1: Running geodesic derep..."
"$GEODESIC" derep \
    --tax-file "$INPUT_TSV" \
    --db-path "$WORKDIR/test.db" \
    --prefix test \
    --threads "$THREADS" \
    --ani-threshold "$ANI_THRESHOLD" \
    -q \
    2>&1 | tail -10

DB="$WORKDIR/test.db"
[[ -f "$DB" ]] || { echo "ERROR: geodesic did not produce $DB"; exit 1; }

# --- Step 2: Extract rep and non-rep file lists ---
echo ""
echo "==> Step 2: Extracting representatives and non-representatives from DB..."

# Get non-rep accessions and file paths (CSV: accession,file)
"$DUCKDB" "$DB" -csv -noheader \
    "SELECT g.accession, g.file
     FROM genomes g
     JOIN genomes_derep gd ON g.accession = gd.accession
     WHERE gd.representative = false" \
    > "$WORKDIR/nonreps.csv"

# Get rep accessions and file paths (CSV: accession,file)
"$DUCKDB" "$DB" -csv -noheader \
    "SELECT g.accession, g.file
     FROM genomes g
     JOIN genomes_derep gd ON g.accession = gd.accession
     WHERE gd.representative = true" \
    > "$WORKDIR/reps.csv"

N_REPS=$(wc -l < "$WORKDIR/reps.csv")
N_NONREPS=$(wc -l < "$WORKDIR/nonreps.csv")

echo "  Representatives: $N_REPS"
echo "  Non-representatives: $N_NONREPS"

if [[ "$N_NONREPS" -eq 0 ]]; then
    echo "PASS: all genomes are representatives (nothing to verify)"
    exit 0
fi

if [[ "$N_REPS" -eq 0 ]]; then
    echo "ERROR: no representatives found but $N_NONREPS non-reps exist"
    exit 1
fi

# Build file lists for skani
cut -d',' -f2 "$WORKDIR/nonreps.csv" | sort -u > "$WORKDIR/query_files.txt"
cut -d',' -f2 "$WORKDIR/reps.csv"    | sort -u > "$WORKDIR/ref_files.txt"

# --- Step 3: Run skani dist ---
echo ""
echo "==> Step 3: Running skani dist (non-reps vs reps)..."
"$SKANI" dist \
    --ql "$WORKDIR/query_files.txt" \
    --rl "$WORKDIR/ref_files.txt" \
    -t "$THREADS" \
    --min-af 0 \
    -o "$WORKDIR/ani_results.tsv" \
    2>/dev/null

[[ -f "$WORKDIR/ani_results.tsv" ]] || { echo "ERROR: skani produced no output"; exit 1; }
N_PAIRS=$(tail -n +2 "$WORKDIR/ani_results.tsv" | wc -l)
echo "  skani produced $N_PAIRS pairwise results"

# --- Step 4: Verify coverage ---
echo ""
echo "==> Step 4: Verifying ANI coverage..."
python3 - "$WORKDIR/nonreps.csv" "$WORKDIR/reps.csv" "$WORKDIR/ani_results.tsv" "$ANI_THRESHOLD" <<'PYEOF'
import sys, csv, os

nonreps_file = sys.argv[1]
reps_file    = sys.argv[2]
ani_file     = sys.argv[3]
threshold    = float(sys.argv[4])

# Load CSV (accession,file) → build file path → accession map
def load_acc_file_map(path):
    m = {}
    with open(path) as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) >= 2:
                acc, fpath = row[0], row[1]
                m[fpath] = acc
    return m

nonrep_map = load_acc_file_map(nonreps_file)
rep_map    = load_acc_file_map(reps_file)

# Collect all non-rep accessions
nonrep_accs = set()
with open(nonreps_file) as f:
    reader = csv.reader(f)
    for row in reader:
        if row:
            nonrep_accs.add(row[0])

# For each non-rep, find best ANI to any rep
best_ani = {}  # non-rep accession → best ANI
best_rep = {}  # non-rep accession → best rep accession

with open(ani_file) as f:
    reader = csv.DictReader(f, delimiter='\t')
    for row in reader:
        ref_file   = row['Ref_file']
        query_file = row['Query_file']
        try:
            ani = float(row['ANI'])
        except (ValueError, KeyError):
            continue

        # Determine which is non-rep and which is rep
        # skani dist: query=non-rep, ref=rep
        q_basename = os.path.basename(query_file)
        r_basename = os.path.basename(ref_file)

        q_acc = nonrep_map.get(query_file) or nonrep_map.get(q_basename)
        r_acc = rep_map.get(ref_file) or rep_map.get(r_basename)

        if q_acc and r_acc and q_acc in nonrep_accs:
            if ani > best_ani.get(q_acc, 0.0):
                best_ani[q_acc] = ani
                best_rep[q_acc] = r_acc

failures = []
for acc in sorted(nonrep_accs):
    ani = best_ani.get(acc, 0.0)
    rep = best_rep.get(acc, 'NONE')
    if ani < threshold:
        failures.append((acc, rep, ani))

if failures:
    print(f"\nFAIL: {len(failures)}/{len(nonrep_accs)} non-reps below {threshold}% ANI\n")
    for acc, rep, ani in failures[:20]:
        print(f"  {acc} -> best rep {rep} ANI={ani:.2f}%")
    if len(failures) > 20:
        print(f"  ... and {len(failures) - 20} more")
    sys.exit(1)
else:
    anis = list(best_ani.values())
    mean_ani = sum(anis) / len(anis) if anis else 0
    min_ani  = min(anis) if anis else 0
    print(f"PASS: all {len(nonrep_accs)} non-reps covered at >= {threshold}% ANI")
    print(f"  Mean best-ANI: {mean_ani:.2f}%  Min: {min_ani:.2f}%")
PYEOF

#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GEODESIC="${GEODESIC:-${SCRIPT_DIR}/../../build/geodesic}"
SKANI="${SKANI:-/maps/projects/fernandezguerra/apps/opt/conda/envs/bioinfo/bin/skani}"
INPUT_TSV="${1:-${SCRIPT_DIR}/test_input.tsv}"
ANI_THRESHOLD="${2:-95.0}"
THREADS="${3:-4}"

for tool in "$GEODESIC" "$SKANI"; do
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
    --prefix test \
    --out-dir "$WORKDIR" \
    --threads "$THREADS" \
    --ani-threshold "$ANI_THRESHOLD" \
    -q \
    2>&1 | tail -10

DEREP_TSV="$WORKDIR/test_derep_genomes.tsv"
[[ -f "$DEREP_TSV" ]] || { echo "ERROR: geodesic did not produce $DEREP_TSV"; exit 1; }

# --- Step 2: Extract rep and non-rep file lists from TSV ---
echo ""
echo "==> Step 2: Extracting representatives and non-representatives from TSV..."

python3 - "$DEREP_TSV" "$INPUT_TSV" "$WORKDIR" <<'PYEOF'
import sys, csv, os

derep_file = sys.argv[1]
input_file = sys.argv[2]
workdir    = sys.argv[3]

# Build accession → file path map from input TSV
acc_to_file = {}
with open(input_file) as f:
    reader = csv.DictReader(f, delimiter='\t')
    for row in reader:
        acc_to_file[row['accession']] = row['file']

# Parse derep TSV to classify reps vs non-reps
reps = []
nonreps = []
with open(derep_file) as f:
    reader = csv.DictReader(f, delimiter='\t')
    for row in reader:
        acc = row['accession']
        fpath = acc_to_file.get(acc, '')
        if not fpath:
            continue
        is_rep = row.get('is_representative', row.get('representative', '')).lower() in ('true', '1', 'yes')
        if is_rep:
            reps.append((acc, fpath))
        else:
            nonreps.append((acc, fpath))

with open(os.path.join(workdir, 'reps.csv'), 'w') as f:
    for acc, fpath in reps:
        f.write(f"{acc},{fpath}\n")

with open(os.path.join(workdir, 'nonreps.csv'), 'w') as f:
    for acc, fpath in nonreps:
        f.write(f"{acc},{fpath}\n")

print(f"  Representatives: {len(reps)}")
print(f"  Non-representatives: {len(nonreps)}")

if len(nonreps) == 0:
    print("PASS: all genomes are representatives (nothing to verify)")
    sys.exit(0)
if len(reps) == 0:
    print(f"ERROR: no representatives found but {len(nonreps)} non-reps exist")
    sys.exit(1)
PYEOF

N_REPS=$(wc -l < "$WORKDIR/reps.csv")
N_NONREPS=$(wc -l < "$WORKDIR/nonreps.csv")

if [[ "$N_NONREPS" -eq 0 ]]; then
    exit 0
fi
if [[ "$N_REPS" -eq 0 ]]; then
    echo "ERROR: no representatives found"
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

nonrep_accs = set()
with open(nonreps_file) as f:
    reader = csv.reader(f)
    for row in reader:
        if row:
            nonrep_accs.add(row[0])

best_ani = {}
best_rep = {}

with open(ani_file) as f:
    reader = csv.DictReader(f, delimiter='\t')
    for row in reader:
        ref_file   = row['Ref_file']
        query_file = row['Query_file']
        try:
            ani = float(row['ANI'])
        except (ValueError, KeyError):
            continue

        q_acc = nonrep_map.get(query_file)
        r_acc = rep_map.get(ref_file)

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

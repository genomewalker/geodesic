# Integration Tests

## test_coverage_gate.sh

Verifies that every non-representative genome is within `ani_threshold` ANI of at least one representative, using exact skani pairwise comparison as ground truth.

### Usage

```bash
# Default: 50 S. enterica genomes, 95% ANI threshold
./test_coverage_gate.sh

# Custom input and threshold
./test_coverage_gate.sh /path/to/input.tsv 97.0

# Third arg: threads (default 4)
./test_coverage_gate.sh test_input.tsv 95.0 8
```

### Environment variables

| Variable   | Default                                          |
|------------|--------------------------------------------------|
| `GEODESIC` | `../../build/geodesic`                           |
| `SKANI`    | `/maps/projects/.../conda/envs/bioinfo/bin/skani`|
| `DUCKDB`   | `/maps/projects/.../conda/envs/bioinfo/bin/duckdb`|

### What it does

1. Runs `geodesic derep` on the input TSV
2. Queries the output DuckDB for representative and non-representative genome files
3. Runs `skani dist` (non-reps as queries, reps as references)
4. Asserts every non-rep has at least one rep with ANI >= threshold

Exit 0 on pass, 1 on failure.

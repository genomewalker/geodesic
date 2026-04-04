#!/bin/bash
#SBATCH --job-name=gpk-5m
#SBATCH --partition=compregular
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=7-00:00:00
#SBATCH --output=/maps/projects/caeg/scratch/kbd606/gpk_5m_build_%j.log
#SBATCH --error=/maps/projects/caeg/scratch/kbd606/gpk_5m_build_%j.log

set -euo pipefail

INPUT=/maps/projects/caeg/scratch/kbd606/all_genomes_r226.tsv
OUTPUT=/maps/projects/caeg/scratch/kbd606/all_genomes_r226.gpk
GENOPACK=/maps/projects/fernandezguerra/apps/repos/genopack/build/genopack
THREADS=32
PARALLEL=4

echo "$(date) — Starting genopack build: $(wc -l < "$INPUT") lines"
echo "Input:  $INPUT"
echo "Output: $OUTPUT"

"$GENOPACK" build \
    -i "$INPUT" \
    -o "$OUTPUT" \
    -t "$THREADS" \
    -p "$PARALLEL" \
    --sketch \
    --sketch-kmer 16 \
    --sketch-size 10000 \
    --taxon-group \
    --no-hnsw \
    --no-cidx \
    -v

echo "$(date) — Build complete"

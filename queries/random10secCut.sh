#!/bin/bash

# This script cuts 10 seconds from the beginning of each wav file in the random20/wav directory
# and stores the cut files in the cut_output directory.
# The start time of the cut is randomly chosen between 0 and 20 seconds

INPUT_DIR="random20/wav"
OUTPUT_DIR="cut_output"

mkdir -p "$OUTPUT_DIR"

# Seed
SEED=1234
RANDOM=$SEED

get_random_start() {
    echo $((RANDOM % 21))
}

for file in "$INPUT_DIR"/*.wav; do
    filename=$(basename "$file" .wav)
    output_path="$OUTPUT_DIR/${filename}_cut.wav"

    start_time=$(get_random_start)

    ffmpeg -i "$file" -ss "$start_time" -t 10 "$output_path"
    echo "Cut: $file -> $output_path (start: ${start_time}s)"
done

#!/bin/bash
# Convert mp3 files to wav files
# We picked 20 random mp3 files from the dataset and stored them in the random20/mp3 directory
# This script converts these mp3 files to wav files and stores them in the random20/wav directory

INPUT_DIR="random20/mp3"
OUTPUT_DIR="random20/wav"

echo "Converting mp3 files to wav files..."
mkdir -p "$OUTPUT_DIR"

for file in "$INPUT_DIR"/*.mp3; do
    filename=$(basename "$file" .mp3)
    output_path="$OUTPUT_DIR/$filename.wav"

    ffmpeg -i "$file" "$output_path"
    echo "Konverted: $file -> $output_path"
done
echo "Conversion finished."
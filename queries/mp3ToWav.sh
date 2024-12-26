#!/bin/bash

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
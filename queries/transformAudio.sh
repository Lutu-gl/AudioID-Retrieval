#!/bin/bash

# This script is used for transforming audio files to add noise and apply compression
# The input files are the cut_output files obtained from the random10SecCut.sh script
# The output files are stored in the noise_output and coding_output directories

INPUT_DIR="cut_output"
NOISE_OUTPUT_DIR="noise_output"
CODING_OUTPUT_DIR="coding_output"

mkdir -p "$NOISE_OUTPUT_DIR"
mkdir -p "$CODING_OUTPUT_DIR"

add_gaussian_noise() {
    input_file=$1
    output_file=$2
    ffmpeg -i "$input_file" -filter_complex "anoisesrc=d=10:c=white:r=44100:amplitude=0.1 [noise]; [0:a][noise] amix=inputs=2:duration=first:dropout_transition=2" "$output_file"
}

apply_compression_and_convert_to_wav() {
    input_file=$1
    temp_mp3="temp.mp3"
    output_file=$2

    ffmpeg -i "$input_file" -b:a 12k -ar 8000 "$temp_mp3"
    ffmpeg -i "$temp_mp3" "$output_file"
    rm -f "$temp_mp3"
}

for file in "$INPUT_DIR"/*.wav; do
    filename=$(basename "$file" .wav)

    noise_output_path="$NOISE_OUTPUT_DIR/${filename}_noise.wav"
    add_gaussian_noise "$file" "$noise_output_path"

    coding_output_path="$CODING_OUTPUT_DIR/${filename}_coding.wav"
    apply_compression_and_convert_to_wav "$file" "$coding_output_path"
done

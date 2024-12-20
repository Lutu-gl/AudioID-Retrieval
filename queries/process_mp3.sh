#!/bin/bash
mkdir -p noise_output coding_output  # Folder for results

for file in cut_output/*.mp3; do
  filename=$(basename "$file" .mp3)

  # Add Gaussian Noise
  ffmpeg -i "$file" -filter_complex "anoisesrc=color=white:amplitude=0.02[a];[a]amix=inputs=2:duration=shortest" -b:a 192k "noise_output/${filename}_noise.mp3"

  # Coding version / highly compressed
  ffmpeg -i "$file" -b:a 32k "coding_output/${filename}_compressed.mp3"
done

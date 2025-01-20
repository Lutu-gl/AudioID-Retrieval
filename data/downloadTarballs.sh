#!/bin/bash

# Use this script to download tarballs from the Freesound MTG-Jamendo dataset.
# This script will download all the tarballs from 00 to 60 and extract them into the data directory.
# Used for Milestone 2 of the project.
# Base URL for the files
BASE_URL="https://cdn.freesound.org/mtg-jamendo/raw_30s/audio-low/raw_30s_audio-low-"

# Loop from 00 to 60
for i in $(seq -w 0 60); do
  # Check if the directory already exists
  DEST_DIR="${i}"
  if [ -d "$DEST_DIR" ]; then
    echo "Skipping $DEST_DIR as it already exists."
    continue
  fi

  # Construct the full URL
  FILE_URL="${BASE_URL}${i}.tar"

  # Create the destination directory
  mkdir -p "$DEST_DIR"

  # Download the file
  echo "Downloading $FILE_URL..."
  curl -O "$FILE_URL"

  # Extract the file into the destination directory, stripping the top-level directory
  TAR_FILE="raw_30s_audio-low-${i}.tar"
  echo "Extracting $TAR_FILE to $DEST_DIR..."
  tar --strip-components=1 -xvf "$TAR_FILE" -C "$DEST_DIR"

  # Optionally, remove the tar file to save space
  rm "$TAR_FILE"
done

echo "All files downloaded and extracted."

#!/usr/bin/env bash

if [ "$#" -ne 3 ]; then
    echo "Usage: $0 VIDEO_FILE OUTPUT_DIR SIZE"
    exit 1
fi

input_file="$1"
output_dir="$2"
size="$3"

mkdir -p "$output_dir"
ffmpeg -i "$input_file" -vf "scale=${size}:${size}:force_original_aspect_ratio=decrease,pad=${size}:${size}:(ow-iw)/2:(oh-ih)/2" "${output_dir}/%08d.png"

#!/bin/bash

# Usage: ./split_dataset.sh <path_to_dataset> [--splits N]

if [ -z "$1" ]; then
    echo "Usage: $0 <path_to_dataset> [--splits N]"
    exit 1
fi

input_dir="$1"
splits=4

if [[ "$2" == "--splits" && "$3" =~ ^[0-9]+$ ]]; then
    splits=$3
fi

if [ ! -d "$input_dir/train" ] || [ ! -d "$input_dir/dev" ]; then
    echo "Error: input directory must contain 'train/' and 'dev/' subdirectories."
    exit 2
fi

echo "Splitting .spacy files in $input_dir into $splits chunks..."

for split in train dev; do
    mapfile -t files < <(find "$input_dir/$split" -type f -name "*.spacy" | shuf)
    total=${#files[@]}
    echo "Found $total $split files"

    for i in $(seq 1 $splits); do
        mkdir -p "$input_dir/$i/$split"
    done

    for i in "${!files[@]}"; do
        target=$(( (i % splits) + 1 ))
        cp "${files[$i]}" "$input_dir/$target/$split/"
    done
done

echo "Done. Created $splits subfolders in $input_dir."

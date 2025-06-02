#!/usr/bin/env python3
"""
spacy_folder2bio.py
===================
Convert **all** `.spacy` files inside the usual `train/`, `dev/`, and `validation/` sub‑folders of an input directory into three merged BIO files:

    train.txt   – all docs from train/*.spacy
    dev.txt     – all docs from dev/*.spacy
    test.txt    – all docs from validation/*.spacy

Usage (from the shell) ────────────────────────────────────────────────────────

    python spacy_folder2bio.py /path/to/corpus /path/to/output --lang en

Arguments:
    INPUT_DIR   Folder that contains `train/`, `dev/`, `validation/`.
    OUTPUT_DIR  Destination folder. It will be created if missing and will
                hold `train.txt`, `dev.txt`, `test.txt`.

Options:
    --lang/-l   Language code for loading a blank spaCy tokenizer. When in
                doubt, leave at default "xx" (language‑agnostic).
    --silent    Suppress per‑file progress messages.

Requirements:
    pip install spacy tqdm

The script is self‑contained; no third‑party converters are required. It
streams the corpus, so memory usage stays low (≈ a few MB) even for
multi‑GB corpora.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import spacy
from spacy.tokens import DocBin
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def to_bio(token) -> str:
    """Return token in BIO format (O, B-XXX, I-XXX)."""
    if token.ent_iob_ == "O":
        return "O"
    return f"{token.ent_iob_}-{token.ent_type_}"


def convert_spacy_file(spacy_path: Path, outfile_handle, nlp):
    """Write all docs from a .spacy file to the given open file handle."""
    docbin = DocBin().from_disk(spacy_path)
    for doc in docbin.get_docs(nlp.vocab):
        for tok in doc:
            if tok.is_space:
                continue
            outfile_handle.write(f"{tok.text} {to_bio(tok)}\n")
        outfile_handle.write("\n")  # sentence/doc separator


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(
        description="Merge .spacy datasets into single BIO files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("input_dir", type=Path, help="Directory with train/dev/validation sub‑folders")
    parser.add_argument("output_dir", type=Path, help="Destination directory for .txt files")
    parser.add_argument("--lang", "-l", default="xx", help="spaCy language code for blank tokenizer")
    parser.add_argument("--silent", action="store_true", help="Disable progress output")
    args = parser.parse_args(argv)

    subset_map = {
        "train": "train.txt",
        "dev": "dev.txt",
        "validation": "test.txt",
    }

    # Early sanity checks ---------------------------------------------------
    if not args.input_dir.exists():
        sys.exit(f"Input directory {args.input_dir} does not exist.")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        print(f"⚠️  Output directory {args.output_dir} already contains files; they may be overwritten.", file=sys.stderr)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load a blank nlp object for deserialisation ---------------------------
    nlp = spacy.blank(args.lang)

    for subset, outfile_name in subset_map.items():
        subset_dir = args.input_dir / subset
        outfile_path = args.output_dir / outfile_name

        if not subset_dir.exists():
            print(f"Subset directory {subset_dir} not found; skipping.", file=sys.stderr)
            continue

        spacy_files = sorted(subset_dir.glob("*.spacy"))
        if not spacy_files:
            print(f"No .spacy files in {subset_dir}; skipping.", file=sys.stderr)
            continue

        if not args.silent:
            print(f"→ Converting {len(spacy_files)} files from {subset_dir} → {outfile_path}")

        with outfile_path.open("w", encoding="utf-8") as out_handle:
            iterable = spacy_files if args.silent else tqdm(spacy_files, desc=subset, unit="file")
            for spacy_file in iterable:
                convert_spacy_file(spacy_file, out_handle, nlp)

    if not args.silent:
        print("Done.")


if __name__ == "__main__":
    main()

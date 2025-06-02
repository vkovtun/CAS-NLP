#!/usr/bin/env python3
"""
Convert a folder of spaCy DocBin (.spacy) files into BIO‑tagged text files.

Input folder structure is expected to look like:
    corpus/
        train/
            file1.spacy
            file2.spacy
        dev/
            *.spacy
        validation/
            *.spacy

The output folder will mirror the same relative structure but contain .txt files:
    converted/
        train/
            file1.txt
            file2.txt
        dev/
            *.txt
        validation/
            *.txt

Usage
-----
python spacy_folder2bio.py /path/to/corpus /path/to/converted --lang en

Arguments
~~~~~~~~~
* in_root  - Root directory that contains the subfolders with .spacy files.
* out_root - Destination root directory where .txt files will be written.
* --lang   - spaCy language code (defaults to "xx" for blank multi‑lang).
"""
import argparse
from pathlib import Path
import spacy
from spacy.tokens import DocBin

def tok_to_tag(tok):
    """Return BIO tag string for a token."""
    return "O" if tok.ent_iob_ == "O" else f"{tok.ent_iob_}-{tok.ent_type_}"

def convert_file(in_path: Path, out_path: Path, nlp):
    """Convert a single .spacy file to BIO .txt."""
    doc_bin = DocBin().from_disk(in_path)
    docs = doc_bin.get_docs(nlp.vocab)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as out_f:
        for doc in docs:
            for tok in doc:
                if tok.is_space:
                    continue
                out_f.write(f"{tok.text} {tok_to_tag(tok)}\n")
            out_f.write("\n")  # sentence/doc separator


def main():
    parser = argparse.ArgumentParser(
        description="Recursively convert .spacy DocBin files to BIO‑tagged text while preserving directory structure.")
    parser.add_argument("in_root", type=Path, help="Input root directory containing .spacy files")
    parser.add_argument("out_root", type=Path, help="Output root directory for .txt files")
    parser.add_argument("--lang", default="xx", help="spaCy language code (default: xx)")
    args = parser.parse_args()

    nlp = spacy.blank(args.lang)

    # Iterate all .spacy files below in_root
    for in_file in args.in_root.rglob("*.spacy"):
        rel_path = in_file.relative_to(args.in_root)
        out_file = args.out_root / rel_path.with_suffix(".txt")
        print(f"Converting {in_file} -> {out_file}")
        convert_file(in_file, out_file, nlp)


if __name__ == "__main__":
    main()

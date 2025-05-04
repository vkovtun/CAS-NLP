#!/usr/bin/env python
"""
NER Model Evaluation Script

Evaluate a Hugging Face **TNER** checkpoint on an NER dataset and print precision, recall, and F1.

Example:
    python ner_evaluate.py --model models/cs --dataset wnut_17 --split test
"""

import argparse
import glob
from pathlib import Path
from typing import List, Tuple, Iterator

import spacy
from spacy import Language
from spacy.tokens import DocBin, Doc, Token
from spacy.scorer import Scorer
from spacy.training import Example
from tqdm.std import tqdm

from datasets import load_dataset, Dataset, DatasetDict
from seqeval.metrics import classification_report, precision_score, recall_score, f1_score

_PREFIXES: Tuple[str, ...] = ("Ġ", "▁", "##")
LABEL_LIST = ['B-LOC', 'B-ORG', 'B-PER', 'I-LOC', 'I-ORG', 'I-PER', 'O' ]
SPACY_BLANK_LANGUAGES = {'be': 'xx', 'bg': 'bg', 'bs': 'bs', 'cs': 'cs', 'hr': 'hr', 'mk': 'mk', 'pl': 'pl', 'ru': 'ru', 'sh': 'sh', 'sk': 'sk',
                         'sl': 'sl', 'sr': 'sr', 'uk': 'uk'}


def _clean(tok: str) -> str:
    """Strip common sub‑word prefixes."""
    while tok and tok.startswith(_PREFIXES):
        tok = tok[1:]
    return tok


def align_predictions_to_tokens(tokens: List[str], pred_input: List[str], ent_spans):
    """Map span‑level entity predictions to token‑level IOB2 labels."""
    mapping, j = [], 0
    for tok in tokens:
        while j < len(pred_input) and _clean(pred_input[j]).lower() != tok.lower():
            j += 1
        mapping.append(j)
        j += 1
    labels = ["O"] * len(tokens)
    for pos_list, label in ent_spans:
        if not pos_list:
            continue
        start_tok_idx = next((idx for idx, mp in enumerate(mapping) if mp == pos_list[0]), None)
        end_tok_idx = next((idx for idx, mp in enumerate(mapping) if mp == pos_list[-1]), start_tok_idx)
        if start_tok_idx is None:
            continue
        labels[start_tok_idx] = f"B-{label}"
        for i in range(start_tok_idx + 1, (end_tok_idx or start_tok_idx) + 1):
            labels[i] = f"I-{label}"
    return labels


def get_dataset(local_dataset: dict) -> Tuple[DatasetDict, dict]:
    """Load dataset from local files.
    
    Args:
        local_dataset: Dictionary containing paths to train, validation and test files
        
    Returns:
        Tuple containing dataset and metadata
    """
    dataset_dict = {}
    for key, filepath in local_dataset.items():
        tokens, tags = [], []
        with open(filepath, 'r', encoding='utf-8') as f:
            current_tokens, current_tags = [], []
            for line in f:
                if line.strip():
                    token, tag = line.strip().split('\t')
                    current_tokens.append(token)
                    current_tags.append(LABEL_LIST.index(tag))
                elif current_tokens:
                    tokens.append(current_tokens)
                    tags.append(current_tags)
                    current_tokens, current_tags = [], []
            if current_tokens:
                tokens.append(current_tokens)
                tags.append(current_tags)
        dataset_dict[key] = Dataset.from_dict({"tokens": tokens, "tags": tags})
    return DatasetDict(dataset_dict), {}


def evaluate_model(language: str) -> None:
    """Evaluate the model at *entity‑span* level and print a spaCy report."""
    print(f"Loading model for language: {language} …")
    model = spacy.load(Path(f"models/wikiann/{language}/model-best"))

    # --- load the gold‑standard validation set ----------------------------
    nlp: Language = spacy.blank(SPACY_BLANK_LANGUAGES[language])

    validation_files = glob.glob(f"datasets/wikiann/{language}/validation/*.spacy")
    print(f"Loading {len(validation_files)} validation files...")
    gold_docs: List[Doc] = []
    for file in validation_files:
        print(f"  → {file}")
        db = DocBin().from_disk(file)
        gold_docs.extend(list(db.get_docs(nlp.vocab)))

    # --- score span matches ------------------------------------------------
    examples: List[Example] = []
    for gold_doc in tqdm(gold_docs, desc="Creating examples"):
        pred_doc = model(gold_doc.text)
        examples.append(Example(pred_doc, gold_doc))

    scorer = Scorer()
    results = scorer.score(examples)                  # returns the metrics dict

    print("\n=== Span‑level named‑entity evaluation ===")
    print(f"Precision : {results['ents_p']:.4f}")
    print(f"Recall    : {results['ents_r']:.4f}")
    print(f"F1        : {results['ents_f']:.4f}")

    if results.get("ents_per_type"):
        print("\nPer‑label breakdown:")
        for label, m in sorted(results["ents_per_type"].items()):
            print(f"  {label:12}  P={m['p']:.4f}  R={m['r']:.4f}  F1={m['f']:.4f}")


def main() -> None:
    """CLI wrapper."""
    parser = argparse.ArgumentParser(description="Evaluate a TNER model on an HF dataset")
    parser.add_argument("--language", help="Language")
    args = parser.parse_args()
    evaluate_model(args.language)


if __name__ == "__main__":
    main()

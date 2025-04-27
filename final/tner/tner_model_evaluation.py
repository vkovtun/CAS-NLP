#!/usr/bin/env python
"""
NER Model Evaluation Script

Evaluate a Hugging Face **TNER** checkpoint on an NER dataset and print precision, recall, and F1.

Example:
    python ner_evaluate.py --model models/cs --dataset wnut_17 --split test
"""

import argparse
from typing import List, Tuple

from tner import TransformersNER, get_dataset
from datasets import load_dataset
from seqeval.metrics import classification_report, precision_score, recall_score, f1_score

_PREFIXES: Tuple[str, ...] = ("Ġ", "▁", "##")
LABEL_LIST = ['B-LOC', 'B-ORG', 'B-PER', 'I-LOC', 'I-ORG', 'I-PER', 'O' ]


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


def evaluate_model(model_name: str, language: str) -> None:
    """Run evaluation and print a seqeval report."""
    print(f"Loading model: {model_name} …")
    model = TransformersNER(model_name)

    data_files = {
        "train": f"datasets/wikiann/{language}/train.txt",
        "validation": f"datasets/wikiann/{language}/dev.txt",
        "test": f"datasets/wikiann/{language}/test.txt"
    }
    # data_files = [f"datasets/wikiann/{language}/test.jsonl"]
    ds, metadata = get_dataset(local_dataset=data_files)
    ds_test = ds["test"]
    ds_test_tokens = ds_test["tokens"]
    ds_test_tags = ds_test["tags"]

    if len(ds_test_tokens) != len(ds_test_tags):
        raise ValueError(f"Number of tokens ({len(ds_test_tokens)}) does not match number of tags ({len(ds_test_tags)})")

    true_labels, pred_labels = [], []
    for i in range(0, len(ds_test_tokens)):
        tokens = ds_test_tokens[i]
        tags = ds_test_tags[i]

        gold = [LABEL_LIST[i] for i in tags]
        sent = " ".join(tokens)

        outputs = model.predict([sent])
        pred_input = outputs["input"][0]
        ent_pred = [(ent["position"], ent["type"]) for ent in outputs["entity_prediction"][0]]

        pred = align_predictions_to_tokens(tokens, pred_input, ent_pred)
        true_labels.append(gold)
        pred_labels.append(pred)

    print(classification_report(true_labels, pred_labels, digits=4))
    print(f"Precision : {precision_score(true_labels, pred_labels):.4f}")
    print(f"Recall    : {recall_score(true_labels, pred_labels):.4f}")
    print(f"F1        : {f1_score(true_labels, pred_labels):.4f}")


def main() -> None:
    """CLI wrapper."""
    parser = argparse.ArgumentParser(description="Evaluate a TNER model on an HF dataset")
    parser.add_argument("--model", default="tner/roberta-large-wnut2017", help="Model name or local path")
    parser.add_argument("--language", help="Language")
    args = parser.parse_args()
    evaluate_model(args.model, args.language)


if __name__ == "__main__":
    main()

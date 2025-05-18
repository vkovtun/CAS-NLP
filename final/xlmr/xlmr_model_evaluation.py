#!/usr/bin/env python
import argparse
from typing import List

from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from seqeval.metrics import classification_report, precision_score, recall_score, f1_score

LABEL_LIST = ['B-LOC', 'B-ORG', 'B-PER', 'I-LOC', 'I-ORG', 'I-PER', 'O']


def load_conll_txt(filepath: str):
    """Load CoNLL-style txt file with token-label pairs."""
    sentences, labels = [], []
    with open(filepath, encoding="utf-8") as f:
        tokens, tags = [], []
        for line in f:
            line = line.strip()
            if not line:
                if tokens:
                    sentences.append(tokens)
                    labels.append(tags)
                    tokens, tags = [], []
            else:
                parts = line.split()
                if len(parts) >= 2:
                    tokens.append(parts[0])
                    tags.append(parts[-1])
        if tokens:  # handle last sentence if no trailing newline
            sentences.append(tokens)
            labels.append(tags)
    return sentences, labels


def align_labels(tokens: List[str], entities: List[dict], sentence: str) -> List[str]:
    labels = ["O"] * len(tokens)
    char_to_token_map = []
    i = 0
    for tok in tokens:
        while i < len(sentence) and sentence[i].isspace():
            i += 1
        start = i
        for c in tok:
            if i < len(sentence) and sentence[i] == c:
                i += 1
        end = i
        char_to_token_map.append((start, end))

    for ent in entities:
        ent_start = ent["start"]
        ent_end = ent["end"]
        label = ent.get("entity_group", "O")

        start_idx, end_idx = None, None
        for idx, (s, e) in enumerate(char_to_token_map):
            if s <= ent_start < e and start_idx is None:
                start_idx = idx
            if s < ent_end <= e:
                end_idx = idx
        if start_idx is not None:
            labels[start_idx] = f"B-{label}"
            if end_idx is None:
                end_idx = start_idx
            for i in range(start_idx + 1, end_idx + 1):
                labels[i] = f"I-{label}"
    return labels



def evaluate_model(model_name: str, language: str) -> None:
    print(f"Loading model: {model_name} …")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(model_name)
    ner = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="first")

    test_path = f"datasets/wikiann/{language}/test.txt"
    all_tokens, all_gold_labels = load_conll_txt(test_path)

    true_labels, pred_labels = [], []
    batch_size = 32
    for i in range(0, len(all_tokens), batch_size):
        token_batch = all_tokens[i:i + batch_size]
        gold_batch = all_gold_labels[i:i + batch_size]
        sentence_batch = [" ".join(tokens) for tokens in token_batch]
        predictions_batch = ner(sentence_batch)

        for tokens, gold, prediction, sentence in zip(token_batch, gold_batch, predictions_batch, sentence_batch):
            pred = align_labels(tokens, prediction, sentence)
            true_labels.append(gold)
            pred_labels.append(pred)

    print(classification_report(true_labels, pred_labels, digits=4))
    print(f"Precision : {precision_score(true_labels, pred_labels):.4f}")
    print(f"Recall    : {recall_score(true_labels, pred_labels):.4f}")
    print(f"F1        : {f1_score(true_labels, pred_labels):.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a 🤗 token classification model")
    parser.add_argument("--model", default="ivlcic/xlmr-ner-slavic", help="Model name or local path")
    parser.add_argument("--language", help="Language")
    args = parser.parse_args()
    evaluate_model(args.model, args.language)


if __name__ == "__main__":
    main()

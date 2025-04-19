import json
import os

label_mapping = ['B-LOC', 'B-ORG', 'B-PER', 'I-LOC', 'I-ORG', 'I-PER', 'O' ]

def jsonl_to_conll(input_path, output_path):
    with open(input_path, 'r', encoding='utf-8') as infile, open(output_path, 'w', encoding='utf-8') as outfile:
        for line in infile:
            data = json.loads(line)
            tokens = data['tokens']
            tags = [label_mapping[int(tag)] if isinstance(tag, int) else tag for tag in data['tags']]
            for token, tag in zip(tokens, tags):
                outfile.write(f"{token} {tag}\n")
            outfile.write("\n")  # Sentence separator

# Convert each split

WIKIANN_DIR = 'datasets/tner/wikiann/'

for f in os.scandir(WIKIANN_DIR):
    if f.is_dir():
        path = f.path
        jsonl_to_conll(f'{path}/train.jsonl', f'{path}/train.txt')
        jsonl_to_conll(f'{path}/dev.jsonl', f'{path}/dev.txt')
        jsonl_to_conll(f'{path}/test.jsonl', f'{path}/test.txt')


# jsonl_to_conll('datasets/tner/wikiann/bg/train.jsonl', 'datasets/tner/wikiann/bg/train.txt')
# jsonl_to_conll('datasets/tner/wikiann/bg/dev.jsonl', 'datasets/tner/wikiann/bg/valid.txt')
# jsonl_to_conll('datasets/tner/wikiann/bg/test.jsonl', 'datasets/tner/wikiann/bg/test.txt')

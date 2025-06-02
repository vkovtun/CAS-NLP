import os

from tqdm import tqdm

import datasets
from datasets import DatasetDict

LANGUAGES =             ['be', 'bg', 'bs', 'cs', 'hr', 'mk', 'pl', 'ru', 'sh', 'sk', 'sl', 'sr', 'uk']
SPACY_BLANK_LANGUAGES = ['xx', 'bg', 'bs', 'cs', 'hr', 'mk', 'pl', 'ru', 'sh', 'sk', 'sl', 'sr', 'uk']

def load_and_split_ds(path, name, test_size=0.2):
    ds = datasets.load_dataset(path, name)
    ds_split_1 = ds['train'].train_test_split(test_size=test_size)

    if 'validation' in ds:
        return DatasetDict({
            'train': ds_split_1['train'],
            'test': ds_split_1['test'],
            'validation': ds['validation']})
    else:
        ds_split_2 = ds_split_1['test'].train_test_split(test_size=0.5)
        return DatasetDict({
            'train': ds_split_1['train'],
            'test': ds_split_2['test'],
            'validation': ds_split_2['train']})

def get_lines_set(file_path):
    lines_set = set()
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if line:
                lines_set.add(line)
    return lines_set

PER_WIKIDATA_ENTITIES = get_lines_set('PER-ND.txt') | get_lines_set('PER-FI.txt')
LOC_WIKIDATA_ENTITIES = get_lines_set('LOC-ND.txt')
ORG_WIKIDATA_ENTITIES = get_lines_set('ORG-ND.txt')

def get_entity_by_qid(qid):
    if not qid:
        return 'MISC'
    elif qid in LOC_WIKIDATA_ENTITIES:
        return 'LOC'
    elif qid in PER_WIKIDATA_ENTITIES:
        return 'PER'
    elif qid in ORG_WIKIDATA_ENTITIES:
        return 'ORG'
    else:
        return 'MISC'

def convert_row_wikiann(row):
    entities = []
    for anchor in row['paragraph_anchors']:
        start_raw = anchor.get('start')
        end_raw = anchor.get('end')
        qid = anchor.get('qid')
        label = get_entity_by_qid(str(qid))

        if start_raw is None or end_raw is None or label is None:
            continue

        try:
            start = int(start_raw)
            end = int(end_raw)
        except ValueError:
            continue

        entities.append({"start": start, "end": end, "label": label})

    return {"text": row["paragraph_text"], "entities": entities}

def tokenize_and_tag(text, entities):
    tokens = []
    current = 0
    sorted_entities = sorted(entities, key=lambda x: x['start'])

    while current < len(text):
        if text[current].isspace():
            current += 1
            continue

        end = current + 1
        while end < len(text) and not text[end].isspace():
            end += 1
        token = text[current:end]

        label = 'O'
        for entity in sorted_entities:
            if current >= entity['start'] and end <= entity['end']:
                prefix = 'B-' if current == entity['start'] else 'I-'
                label = prefix + entity['label']
                break

        tokens.append((token, label))
        current = end

    return tokens

def write_tagged_text_file(dataset, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in tqdm(dataset, desc=f"Writing {output_path}"):
            text = item['text']
            entities = item['entities']
            for token, label in tokenize_and_tag(text, entities):
                f.write(f"{token} {label}\n")
            f.write("\n")

def create_text_files(data_source, language):
    train_ner = data_source['train'].shuffle().select(range(min(3200000, len(data_source['train'])))).map(convert_row_wikiann)
    write_tagged_text_file(train_ner, f'./datasets/wikiann/{language}/train.txt')

    dev_ner = data_source['test'].shuffle().select(range(min(960000, len(data_source['test'])))).map(convert_row_wikiann)
    write_tagged_text_file(dev_ner, f'./datasets/wikiann/{language}/dev.txt')

    valid_ner = data_source['validation'].shuffle().select(range(min(25000, len(data_source['validation'])))).map(convert_row_wikiann)
    write_tagged_text_file(valid_ner, f'./datasets/wikiann/{language}/validation.txt')

def main():
    for i, language in enumerate(LANGUAGES):
        print(f"Processing language: {language}")
        ds = load_and_split_ds('unimelb-nlp/wikiann', language)
        create_text_files(ds, language)


if __name__ == "__main__":
    main()
import os

import datasets
import spacy
from datasets import DatasetDict
from spacy.tokens import DocBin, Doc
from tqdm import tqdm

PUNCTUATION_MARKS = {
    ',', '.', '!', '?', ';', ':', '"', "'", "''", '[', ']', '(', ')', '{', '}',
    '“', '”', '‘', '’', '–', '-', '—', '/', '\\', '|', '<', '>', '<<', '>>', '#', '*', '&', '%', '$', '@', '`', '~',
    '‹', '›', '«', '»'
}

NO_SPACE_BEFORE_MARKS = {
    ',', '.', '!', '?', ';', ':', ']', ')', '}', '”', '’', '>', '>>', '›', '»'
}

NO_SPACE_AFTER_MARKS = {
    '[', '(', '{', '“', '‘', '<', '<<', '‹', '«',
}

NO_SPACE_BEFORE_SEQUENCE = ["''", "'"]
NO_SPACE_AFTER_SEQUENCE = ["'", "''"]

DS_PATH = 'unimelb-nlp/wikiann'

LANGUAGES =             ['be', 'bg', 'bs', 'cs', 'hr', 'mk', 'pl', 'ru', 'sh', 'sk', 'sl', 'sr', 'uk']
SPACY_BLANK_LANGUAGES = ['xx', 'bg', 'bs', 'cs', 'hr', 'mk', 'pl', 'ru', 'sh', 'sk', 'sl', 'sr', 'uk']


def tokens_to_spans(tokens, tags, language, tag_list):
    """
    Converts a list of tokens into a formatted sentence and extracts entity positions.
    Handles spacing before punctuation marks correctly.

    Args:
        tokens (List[str]): The list of tokens to be converted.
        tags (List[str]): The corresponding list of tags.
        language (str): The language.
        tag_list (List[str]): A callable that maps tags from one type to another.

    Returns:
        Span: The formatted sentence and entity list.
    """

    sentence = ""
    entities = []
    current_pos = 0
    current_entity = None

    for token, tag_idx in zip(tokens, tags):
        if token in PUNCTUATION_MARKS:
            sentence += token  # Attach punctuation directly
            current_pos += len(token)
        else:
            if sentence and not sentence.endswith(' '):
                sentence += ' '
                current_pos += 1
            start_pos = current_pos
            sentence += token
            end_pos = current_pos + len(token)
            current_pos = end_pos
            tag = tag_list[tag_idx]

            if tag.startswith('B-'):
                if current_entity:
                    entities.append(current_entity)
                current_entity = {'start': start_pos, 'end': end_pos, 'label': tag[2:]}
            elif tag.startswith('I-') and current_entity:
                current_entity['end'] = end_pos
            elif tag != 'O':
                if current_entity:
                    entities.append(current_entity)
                    current_entity = None
                entities.append({'start': start_pos, 'end': end_pos, 'label': tag})
            else:
                if current_entity:
                    entities.append(current_entity)
                    current_entity = None

    if current_entity:
        entities.append(current_entity)

    return sentence, entities


def load_ds(path, name, language):
    ds = datasets.load_dataset(path, name)

    # Apply filter to each split
    ds = ds.filter(lambda example: example['language'] == language)

    return ds


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


def create_doc(tokens, ner_tags, nlp):
    tokens_len = len(tokens)
    spaces = [True] * tokens_len
    spaces[tokens_len - 1] = False

    for i, token in enumerate(tokens):
        if i > 0 and (token in NO_SPACE_BEFORE_MARKS or
                i < tokens_len - 1 and tokens[i] == "''" and tokens[i + 1] == "'"):
            spaces[i - 1] = False
        if token in NO_SPACE_AFTER_MARKS or \
                i > 0 and tokens[i - 1] == "'" and tokens[i] == "''":
            spaces[i] = False

    return Doc(nlp.vocab, tokens, spaces, ents=ner_tags)


def create_spacy_doc_bin_files(dataset, output_dir, file_name, nlp, tag_list, chunk_size=100):
    os.makedirs(output_dir, exist_ok=True)  # Ensure output directory exists

    docs_limit = len(dataset)
    file_index = 0

    for i in tqdm(range(0, docs_limit, chunk_size), "Serialization:"):
        db = DocBin()
        for j in range(i, min(i + chunk_size, docs_limit)):
            datum = dataset[j]
            tokens = datum['tokens']
            ner_tags = [tag_list[i] for i in datum['ner_tags']]

            doc = create_doc(tokens, ner_tags, nlp)

            db.add(doc)

        # Save the chunk to a new file
        output_file = os.path.join(output_dir, f'{file_name}{file_index + 1}.spacy')
        db.to_disk(output_file)
        file_index += 1


def create_spacy_files(data_source, language, nlp, tag_list):
    train_ner = data_source['train'].shuffle().select(range(min(3200000, len(data_source['train']))))
    create_spacy_doc_bin_files(dataset=train_ner, file_name='train', output_dir=f'./{language}/train', nlp=nlp, tag_list=tag_list)

    dev_ner = data_source['test'].shuffle().select(range(min(960000, len(data_source['test']))))
    create_spacy_doc_bin_files(dataset=dev_ner, file_name='dev', output_dir=f'./{language}/dev', nlp=nlp, tag_list=tag_list)

    valid_ner = data_source['validation'].shuffle().select(range(min(480000, len(data_source['validation']))))
    create_spacy_doc_bin_files(dataset=valid_ner, file_name='validation', output_dir=f'./{language}/validation', nlp=nlp, tag_list=tag_list)

    print(f"len(train_ner)={len(train_ner)}")
    print(f"len(dev_ner)={len(dev_ner)}")
    print(f"len(valid_ner)={len(valid_ner)}")

tags = ['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC']

# Slavic languages supported by 'unimelb-nlp/wikiann':
# be – Belarusian (East Slavic)
# bg – Bulgarian (South Slavic)
# bs – Bosnian (South Slavic)
# cs – Czech (West Slavic)
# hr – Croatian (South Slavic)
# mk – Macedonian (South Slavic)
# pl – Polish (West Slavic)
# ru – Russian (East Slavic)
# sh – Serbo-Croatian (South Slavic; sometimes used as a common code for Bosnian, Croatian, Serbian, etc.)
# sk – Slovak (West Slavic)
# sl – Slovenian (South Slavic)
# sr – Serbian (South Slavic)
# uk – Ukrainian (East Slavic)
#
# These are the number of data fields each language is trained at.
# ┌────────────┬─────────┬────────────┬───────┐
# │ Language   │ Train   │ Validation │ Test  │
# ├────────────┼─────────┼────────────┼───────┤
# │ be         │ 15000   │ 1000       │ 1000  │
# │ bg         │ 20000   │ 10000      │ 10000 │
# │ bs         │ 15000   │ 1000       │ 1000  │
# │ cs         │ 20000   │ 10000      │ 10000 │
# │ hr         │ 20000   │ 10000      │ 10000 │
# │ mk         │ 10000   │ 1000       │ 1000  │
# │ pl         │ 20000   │ 10000      │ 10000 │
# │ ru         │ 20000   │ 10000      │ 10000 │
# │ sh         │ 20000   │ 10000      │ 10000 │
# │ sk         │ 20000   │ 10000      │ 10000 │
# │ sl         │ 15000   │ 10000      │ 10000 │
# │ sr         │ 20000   │ 10000      │ 10000 │
# │ uk         │ 20000   │ 10000      │ 10000 │
# └────────────┴─────────┴────────────┴───────┘

for i, language in enumerate(LANGUAGES):
    print(f"Processing language: {language}")
    ds = datasets.load_dataset(DS_PATH, language)
    create_spacy_files(ds, language, spacy.blank(SPACY_BLANK_LANGUAGES[i]), tags)
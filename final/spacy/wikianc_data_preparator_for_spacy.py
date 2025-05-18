import os

import spacy
from spacy.tokens import DocBin
from tqdm import tqdm

import datasets
from datasets import DatasetDict

LANGUAGES =             ['be', 'bg', 'bs', 'cs', 'hr', 'mk', 'pl', 'ru', 'sh', 'sk', 'sl', 'sr', 'uk']
SPACY_BLANK_LANGUAGES = ['xx', 'bg', 'bs', 'cs', 'hr', 'mk', 'pl', 'ru', 'sh', 'sk', 'sl', 'sr', 'uk']

# python -m spacy train config_wikiann_bs.cfg --output models/wikiann/bs --gpu-id 0

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
    """
    Gets a set of lines from a path.

    :param file_path: The file to read from.
    :return: A set of lines from the file.
    """
    lines_set = set()
    with open(file_path, 'r') as file:
        for line in file:
            # Remove leading/trailing spaces and check if the line is not empty
            line = line.strip()
            if line:
                lines_set.add(line)
    return lines_set


PER_WIKIDATA_ENTITIES = get_lines_set('PER-ND.txt') | get_lines_set('PER-FI.txt')
LOC_WIKIDATA_ENTITIES = get_lines_set('LOC-ND.txt')
ORG_WIKIDATA_ENTITIES = get_lines_set('ORG-ND.txt')


def get_entity_by_qid(qid):
    """
    Returns the entity by WikiData QID.

    :return: Entity corresponding to the QID.
    """
    if not qid:
        return 'MISC'
    elif qid in LOC_WIKIDATA_ENTITIES:
        return 'LOC'
    elif qid in PER_WIKIDATA_ENTITIES:
        return 'PER'
    elif qid in ORG_WIKIDATA_ENTITIES:
        return 'ORG'
    else:
        return None


def convert_row_wikianc(row):
    entities = []
    for anchor in row['paragraph_anchors']:
        start_raw = anchor.get('start')
        end_raw = anchor.get('end')
        qid = anchor.get('qid')
        label = get_entity_by_qid(str(qid))

        # skip if any are None
        if start_raw is None or end_raw is None or label is None:
            continue

        # ensure these are actually integers
        try:
            start = int(start_raw)
            end = int(end_raw)
        except ValueError:
            # if you can't convert them to int, skip
            print(f"start_raw={start_raw}, type(start_raw)={type(start_raw)}")
            print(f"start_raw={end_raw}, type(start_raw)={type(end_raw)}")
            continue

        entities.append({
            "start": start,
            "end": end,
            "label": label,
        })

    data_point = {
        "text": row["paragraph_text"],
        "entities": entities,
    }
    return data_point


def create_spacy_doc_bin_files(dataset, output_dir, file_name, nlp, chunk_size=5000):
    os.makedirs(output_dir, exist_ok=True)  # Ensure output directory exists

    docs_limit = len(dataset)
    file_index = 0

    for i in tqdm(range(0, docs_limit, chunk_size), "Serialization:"):
        db = DocBin()
        for j in range(i, min(i + chunk_size, docs_limit)):
            datum = dataset[j]
            text = datum['text']
            doc = nlp(text)
            ents = []
            for entities in datum.get('entities', []):
                start = entities.get('start')
                end = entities.get('end')
                label = entities.get('label')

                span = doc.char_span(start, end, label=label)

                try:
                    if text[start].isspace():
                        print(f"Entity span '{text[start:end]}' has leading whitespace. Skipping.")
                        print(f"Text: '{text}'")
                        span = None

                    if text[end - 1].isspace():
                        print(f"Entity span '{text[start:end]}' has trailing whitespace. Skipping.")
                        print(f"Text: '{text}'")
                        span = None
                except IndexError:
                    print(f"Index is out of range. start: {start}, end: {end}, test: '{text}'")
                    span = None

                if span is not None:
                    ents.append(span)


            # Discard overlapping entities and keep the longest one
            ents = sorted(ents, key=lambda x: (x.start, -x.end + x.start))
            filtered_ents = []
            for ent in ents:
                if not filtered_ents or ent.start >= filtered_ents[-1].end:
                    filtered_ents.append(ent)

            try:
                doc.ents = filtered_ents
            except ValueError as ex:
                print(f"ValueError raised.")
                print(f"filtered_ents={filtered_ents}, text={text}")
                raise ex
            db.add(doc)

        # Save the chunk to a new file
        output_file = os.path.join(output_dir, f'{file_name}{file_index + 1}.spacy')
        db.to_disk(output_file)
        file_index += 1


def create_spacy_files(data_source, language, nlp):
    train_ner = data_source['train'].shuffle().select(range(min(3200000, len(data_source['train'])))).map(convert_row_wikianc)
    create_spacy_doc_bin_files(dataset=train_ner, file_name='train', output_dir=f'./datasets/wikianc/{language}/train', nlp=nlp)

    dev_ner = data_source['test'].shuffle().select(range(min(960000, len(data_source['test'])))).map(convert_row_wikianc)
    create_spacy_doc_bin_files(dataset=dev_ner, file_name='dev', output_dir=f'./datasets/wikianc/{language}/dev', nlp=nlp)

    valid_ner = data_source['validation'].shuffle().select(range(min(25000, len(data_source['validation'])))).map(convert_row_wikianc)
    create_spacy_doc_bin_files(dataset=valid_ner, file_name='validation', output_dir=f'./datasets/wikianc/{language}/validation', nlp=nlp)


# Slavic languages supported by 'unimelb-nlp/wikianc':
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


def main():
    for i, language in enumerate(LANGUAGES):
        print(f"Processing language: {language}")

        ds = load_and_split_ds('cyanic-selkie/wikianc', language)
        create_spacy_files(ds, language, spacy.blank(SPACY_BLANK_LANGUAGES[i]))


if __name__ == "__main__":
    main()
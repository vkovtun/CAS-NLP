import os

import datasets
import spacy
from datasets import DatasetDict
from spacy.tokens import DocBin
from tqdm import tqdm


def tokens_to_text_with_tags(tokens, tags, tag_map):
    """
    Converts a list of tokens into a formatted sentence and extracts entity positions.
    Handles spacing before punctuation marks correctly.

    Args:
        tokens (List[str]): The list of tokens to be converted.
        tags (List[str]): The corresponding list of tags.
        tag_map (Callable[[str], str]): A callable that maps tags from one type to another.

    Returns:
        Tuple[str, List[Dict[str, int | str]]]: The formatted sentence and entity list.
    """
    sentence = ""
    entities = []
    current_pos = 0
    current_entity = None

    for token, tag in zip(tokens, tags):
        if token in {',', '.', '!', '?', ';', ':', '"', "'s"}:
            sentence += token  # Attach punctuation directly
        else:
            if sentence and not sentence.endswith(' '):
                sentence += ' '
                current_pos += 1
            start_pos = current_pos
            sentence += token
            end_pos = current_pos + len(token)
            current_pos = end_pos

            if tag.startswith('B-'):
                if current_entity:
                    entities.append(current_entity)
                current_entity = {'start': start_pos, 'end': end_pos, 'label': tag_map[tag[2:]]}
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


def create_spacy_doc_bin_files(dataset, output_dir, file_name, language, tag_map, chunk_size=100):
    os.makedirs(output_dir, exist_ok=True)  # Ensure output directory exists

    nlp = spacy.blank(language)
    docs_limit = len(dataset)
    file_index = 0

    for i in tqdm(range(0, docs_limit, chunk_size), "Serialization:"):
        db = DocBin()
        for j in range(i, min(i + chunk_size, docs_limit)):
            datum = dataset[j]
            text, tags = tokens_to_text_with_tags(datum['tokens'], datum['coarse_grained'], tag_map)
            doc = nlp(text)
            ents = []
            for tag in tags:
                start = tag.get('start')
                end = tag.get('end')
                label = tag.get('label')

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


def create_spacy_files(data_source, language, tag_map):
    train_ner = data_source['train'].shuffle().select(range(min(3200000, len(data_source['train']))))
    create_spacy_doc_bin_files(dataset=train_ner, file_name='train', output_dir=f'./{language}/train', language='xx', tag_map=tag_map)

    dev_ner = data_source['test'].shuffle().select(range(min(960000, len(data_source['test']))))
    create_spacy_doc_bin_files(dataset=dev_ner, file_name='dev', output_dir=f'./{language}/dev', language='xx', tag_map=tag_map)

    valid_ner = data_source['validation'].shuffle().select(range(min(480000, len(data_source['validation']))))
    create_spacy_doc_bin_files(dataset=valid_ner, file_name='validation', output_dir=f'./{language}/validation', language='xx', tag_map=tag_map)

    print(f"len(train_ner)={len(train_ner)}")
    print(f"len(dev_ner)={len(dev_ner)}")
    print(f"len(valid_ner)={len(valid_ner)}")


mapa_to_spacy_tag_map = {
    'PERSON': 'PER',
    'ORGANISATION': 'ORG',
    'ADDRESS': 'LOC',
    'DATE': 'MISC',
    'TIME': 'MISC',
    'AMOUNT': 'MISC',
    'O': 'O'
}
    

# Load Datasets

# bg_ds = load_ds('joelniklaus/mapa', 'default', 'bg')
cs_ds = load_ds('joelniklaus/mapa', 'default', 'cs')
# sk_ds = load_ds('joelniklaus/mapa', 'default', 'sk')
# sv_ds = load_ds('joelniklaus/mapa', 'default', 'sv')

# Training Model

## Document Files Initialization

# create_spacy_files(bg_ds,'bg')
create_spacy_files(cs_ds,'cs', mapa_to_spacy_tag_map)
# create_spacy_files(sk_ds,'sk')
# create_spacy_files(sv_ds,'sv')
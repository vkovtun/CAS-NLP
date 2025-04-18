import os
import zipfile
from pathlib import Path
from huggingface_hub import snapshot_download

import gdown
import spacy
import streamlit as st

DEST_ZIP = "model_wikianc_uk_2.zip"
DEST_DIR = "model_wikianc_uk_2/"
TRANSFORMER_DIR = f"{DEST_DIR}/model-best/transformer/"
MODEL_FILE = f"{TRANSFORMER_DIR}/model"


def download(file_id, output):
    try:
        print("Attempting download...", flush=True)
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, output, quiet=False)
        print("Download finished. Checking contents...", flush=True)
        print(os.listdir(), flush=True)
    except Exception as e:
        print("Download failed with exception:", flush=True)
        import traceback
        traceback.print_exc()


def unzip(file, output):
    with zipfile.ZipFile(file, 'r') as zip_ref:
        zip_ref.extractall(output)


def download_and_extract():
    zip_file_id = os.environ['ZIP_FILE_ID']
    model_file_id = os.environ['MODEL_FILE_ID']

    print("Downloading ZIP file...")
    download(zip_file_id, DEST_ZIP)

    print("Unzipping...")
    unzip(DEST_ZIP, DEST_DIR)

    print("Contents of DEST_DIR:")
    print(os.listdir(DEST_DIR))

    print("Contents of model-best:")
    print(os.listdir(f"{DEST_DIR}/model-best"))

    print("Contents of transformer:")
    print(os.listdir(f"{DEST_DIR}/model-best/transformer"))

    print("Downloading model file...")
    download(model_file_id, MODEL_FILE)

    print("Contents of model directory:")
    print(os.listdir(DEST_DIR))

    print("Cleaning up...")
    os.remove(DEST_ZIP)

    print(f"Done! Files are in '{DEST_DIR}'.")


# @st.cache_resource
# def load_model():
#     try:
#         print("Attempting to load SpaCy model...", flush=True)
#         model = spacy.load(Path('model_wikianc_uk_2/model-best'))
#         print("SpaCy model loaded!", flush=True)
#         return model
#     except Exception as e:
#         print("Failed to load SpaCy model:", e, flush=True)
#         raise

@st.cache_resource
def load_model():
    model_path = snapshot_download(repo_id="spacy/xx_ent_wiki_sm", revision="main")
    # Load the local model folder with spaCy
    nlp = spacy.load(model_path)
    return nlp


# if not os.path.isdir(DEST_DIR):
#     print(f"Path '{DEST_DIR}' does not exist. Triggering download...")
#     download_and_extract()
# else:
#     print(f"'{DEST_DIR}' already exists. Skipping download.")

if 'model' not in st.session_state:
    model = load_model()
    st.session_state['model'] = model

st.markdown("# NER tagger app")
st.markdown("This is a names entity recognition app for Multiple languages from SpaCy.")
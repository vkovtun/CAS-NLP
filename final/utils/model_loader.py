import os
import re
import zipfile
from pathlib import Path

import gdown
import spacy
import streamlit as st

MODELS_PATH = "spacy1/models"
WIKIANN_DIR = "wikiann"
MODEL_BEST_DIR = "model-best"

def download(file_id, output):
    try:
        print("Attempting download...", flush=True)
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, output, quiet=False)
        print("Download finished. Checking contents...", flush=True)
    except Exception as e:
        raise RuntimeError(f"Download failed with exception: {e}")


def get_file_id(folder_id, language):
    target_pattern = re.compile(f"^{WIKIANN_DIR}/{language}.zip$")

    # 1 List the folder contents without downloading files
    files = gdown.download_folder(
        id=folder_id,
        skip_download=True,          # only metadata
        quiet=True,
        remaining_ok=True            # ignore the 50-file hard cap warning
    )

    # 2 Pick the file you care about
    matches = [f.id for f in files if target_pattern.match(f.path)]

    if not matches:
        raise ValueError("Pattern not found in folder!")
    if len(matches) > 1:
        raise ValueError("More than one file match the pattern.")

    return matches[0]


def unzip(file, output):
    with zipfile.ZipFile(file, "r") as zip_ref:
        zip_ref.extractall(output)


def download_and_extract(language):
    models_folder_id = os.environ["MODELS_FOLDER_ID"]

    print(f"Downloading ZIP file for language {language}...", flush=True)
    wikiann_path = Path(MODELS_PATH, WIKIANN_DIR)
    model_file_id = get_file_id(models_folder_id, language)

    wikiann_path.mkdir(parents=True, exist_ok=True)
    zip_file_path = Path(wikiann_path, f"{language}.zip")
    download(model_file_id, str(zip_file_path))

    dest_dir = Path(MODELS_PATH, WIKIANN_DIR, language)
    print(f"Unzipping {zip_file_path}...", flush=True)
    unzip(zip_file_path, dest_dir)

    print("Contents of DEST_DIR:", flush=True)
    print(os.listdir(dest_dir))

    print("Cleaning up...", flush=True)
    os.remove(zip_file_path)

    print(f"Done! Files are in '{dest_dir}'.", flush=True)


@st.cache_resource
def load_model(language):
    try:
        print("Attempting to load SpaCy model...", flush=True)
        path = Path(MODELS_PATH, WIKIANN_DIR, language, MODEL_BEST_DIR)

        if not path.exists():
            print(f"Path {str(path)} does not exist. Fetching the model from Google Drive.", flush=True)
            download_and_extract(language)

        model = spacy.load(path)
        print("SpaCy model loaded!", flush=True)
        return model
    except Exception as e:
        raise RuntimeError(f"Could not load model: {e}")
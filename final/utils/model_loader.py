import spacy
from huggingface_hub import snapshot_download
from pathlib import Path
import streamlit as st

# TODO: add language selector
# DEST_ZIP = "model_wikianc_uk_2.zip"
DEST_DIR = "spacy/models/wikiann/uk"
MODEL_BEST_DIR = f"{DEST_DIR}/model-best"
MODEL_FILE = f"{MODEL_BEST_DIR}/transformer/model"

# TODO: Uncomment these later and make sure it works with a file from Google Drive.

# def download(file_id, output):
#     try:
#         print("Attempting download...", flush=True)
#         url = f"https://drive.google.com/uc?id={file_id}"
#         gdown.download(url, output, quiet=False)
#         print("Download finished. Checking contents...", flush=True)
#         print(os.listdir(), flush=True)
#     except Exception as e:
#         print("Download failed with exception:", flush=True)
#         import traceback
#         traceback.print_exc()
#
#
# def unzip(file, output):
#     with zipfile.ZipFile(file, 'r') as zip_ref:
#         zip_ref.extractall(output)
#
# def download_and_extract():
#     zip_file_id = os.environ['ZIP_FILE_ID']
#     model_file_id = os.environ['MODEL_FILE_ID']
#
#     print("Downloading ZIP file...")
#     download(zip_file_id, DEST_ZIP)
#
#     print("Unzipping...")
#     unzip(DEST_ZIP, DEST_DIR)
#
#     print("Contents of DEST_DIR:")
#     print(os.listdir(DEST_DIR))
#
#     print("Contents of model-best:")
#     print(os.listdir(f"{DEST_DIR}/model-best"))
#
#     print("Contents of transformer:")
#     print(os.listdir(f"{DEST_DIR}/model-best/transformer"))
#
#     print("Downloading model file...")
#     download(model_file_id, MODEL_FILE)
#
#     print("Contents of model directory:")
#     print(os.listdir(DEST_DIR))
#
#     print("Cleaning up...")
#     os.remove(DEST_ZIP)
#
#     print(f"Done! Files are in '{DEST_DIR}'.")

@st.cache_resource
def load_model():
    try:
        print("Attempting to load SpaCy model...", flush=True)
        model = spacy.load(Path(MODEL_BEST_DIR))
        print("SpaCy model loaded!", flush=True)
        return model
    except Exception as e:
        raise RuntimeError(f"Could not load model: {e}")
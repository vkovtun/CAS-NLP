import streamlit as st
import spacy
import os
import zipfile
import gdown
from pathlib import Path

FILE_ID="1mleDunOQr-pvRfPq7Q5pzjlz3Kc0eX2y"
DEST_ZIP = "model_wikianc_uk_2.zip"
DEST_DIR = "model_wikianc_uk_2"

def download_and_extract():
    if os.path.isdir(DEST_DIR):
        print(f"'{DEST_DIR}' already exists. Skipping download.")
        return

    # Construct the URL
    url = f"https://drive.google.com/uc?id={FILE_ID}"
    print("Downloading ZIP file...")
    gdown.download(url, DEST_ZIP, quiet=False)

    print("Unzipping...")
    with zipfile.ZipFile(DEST_ZIP, 'r') as zip_ref:
        zip_ref.extractall(DEST_DIR)

    print("Cleaning up...")
    os.remove(DEST_ZIP)

    print(f"Done! Files are in '{DEST_DIR}'.")

download_and_extract()

if 'model' not in st.session_state:
    model = spacy.load(Path('model_wikianc_uk_2/model-best'))
    st.session_state['model'] = model

st.markdown("# NER tagger app")
st.markdown("This is a names entity recognition app for English, Czech, Hungarian and Ukrainian.")
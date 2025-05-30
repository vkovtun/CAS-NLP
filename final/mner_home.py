import os
import zipfile
from pathlib import Path
from huggingface_hub import snapshot_download

import gdown
import spacy
import streamlit as st
from utils.model_loader import load_model

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

# if not os.path.isdir(DEST_DIR):
#     print(f"Path '{DEST_DIR}' does not exist. Triggering download...")
#     download_and_extract()
# else:
#     print(f"'{DEST_DIR}' already exists. Skipping download.")

st.set_page_config(page_title="NER App", layout="wide")

# Sidebar
st.sidebar.title("Language Selection")
AVAILABLE_LANGUAGES = ["be", "bg", "cs", "hr", "mk", "pl", "ru", "sk", "sl", "sr", "uk"]

if "selected_language" not in st.session_state:
    st.session_state["selected_language"] = AVAILABLE_LANGUAGES[0]

st.session_state["selected_language"] = st.sidebar.selectbox(
    "Select a model:",
    options=AVAILABLE_LANGUAGES,
    index=AVAILABLE_LANGUAGES.index(st.session_state["selected_language"])
)

# Load the model only once
language = st.session_state["selected_language"]

if "model" not in st.session_state:
    st.session_state["model"] = {}

if language not in st.session_state["model"]:
    with st.spinner("Loading NER model... Please wait."):
        try:
            model = load_model(language)
            st.session_state["model"][language] = model
        except Exception as e:
            st.error(f"Failed to load the model: {e}")
            st.stop()  # Stop here if load fails

    # Stop rendering further until model is set
    st.rerun()  # optional: force refresh to clean up the UI
else:
    model = st.session_state["model"][language]

st.markdown("# NER tagger app")
st.markdown("This is a named entity recognition app for Slavic languages from SpaCy.")
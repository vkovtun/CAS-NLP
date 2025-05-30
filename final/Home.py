import streamlit as st

from utils.ui_components import setup_sidebar_and_model

st.set_page_config(page_title="NER App", layout="wide")

language, model = setup_sidebar_and_model()

st.markdown("# NER tagger app")
st.markdown("This is a named entity recognition app for Slavic languages from SpaCy.")
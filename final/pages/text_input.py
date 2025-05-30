import streamlit as st
import spacy
from spacy import displacy
from pathlib import Path
from spacy_streamlit import visualize_ner
from utils.ui_components import setup_sidebar_and_model

st.set_page_config(page_title="Text input NER")

language, model = setup_sidebar_and_model()

text = st.text_area("Insert a text to get the NER tags for it")

if text:
    html_results = displacy.render(model(text), style="dep", minify=True, page=True)
    visualize_ner(model(text), labels=model.get_pipe("ner").labels)
import streamlit as st
import spacy
from spacy import displacy
from pathlib import Path
from spacy_streamlit import visualize_ner

st.set_page_config(
    page_title="Text input NER",
)

text = st.text_area("Insert a text to get the NER tags for it")

if text:
    model = st.session_state['model']
    html_results = displacy.render(model(text), style="dep", minify=True, page=True)
    visualize_ner(model(text), labels=model.get_pipe("ner").labels)
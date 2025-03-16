import streamlit as st
import spacy
from spacy import displacy
from pathlib import Path
from spacy_streamlit import visualize_ner

st.set_page_config(
    page_title="Text input NER",
)

if 'model' not in st.session_state:
    model = spacy.load(Path('model_wikianc_uk_2/model-best'))
    st.session_state['model'] = model

text = st.text_area("Insert a text to get the NER tags for it")

if text:
    model = st.session_state['model']
    html_results = displacy.render(model(text), style="dep", minify=True, page=True)
    visualize_ner(model(text), labels=model.get_pipe("ner").labels)
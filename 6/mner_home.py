import streamlit as st
import spacy
from pathlib import Path

# model = spacy.load(Path('model_wikianc_uk_2/model-best'))
# print("Pipeline components:", model.pipe_names)
# print(model)

if 'model' not in st.session_state:
    model = spacy.load(Path('model_wikianc_uk_2/model-best'))
    st.session_state['model'] = model

st.markdown("# NER tagger app")
st.markdown("This is a names entity recognition app for English, Czech, Hungarian and Ukrainian.")
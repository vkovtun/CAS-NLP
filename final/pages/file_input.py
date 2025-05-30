import streamlit as st
import spacy
import random
from spacy import displacy
from pathlib import Path
from io import StringIO

st.set_page_config(
        page_title="File input NER",
)


if 'model' not in st.session_state:
    model = spacy.load(Path('model_wikianc_uk_2/model-best'))
    st.session_state['model'] = model


st.write("You can also upload a file of raw text and the output will be an HTML file.")

uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
    stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
    text = stringio.read()

    model = st.session_state['model']
    html = displacy.render(model(text), style="ent")

    file_name = f"tagged_text_{random.randint(10000,100000)}.html"
    st.download_button("download the output file", html, file_name)
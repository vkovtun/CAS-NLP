import dill
import streamlit as st
from nltk.tokenize import word_tokenize
from io import StringIO
import random
import nltk

nltk.download('punkt')

st.set_page_config(
        page_title="POS Tagger - Tag a file",
)


# Load the trained model from the file
with open('hmm_tagger.pkl', 'rb') as f:
    loaded_tagger = dill.load(f)


st.write("You can also upload a file of raw text and the output will be a file with the following format: 'token pos_tag' ")

uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
    stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
    string_data = stringio.read()
    tokens = word_tokenize(string_data)
    tagged_tokens = loaded_tagger.tag(tokens)
    output = ""
    for token, tag in tagged_tokens:
        output += token + " " + tag + "\n"
    file_name = f"tagged_text_{random.randint(10000,100000)}.txt"
    st.download_button("download the output file", output, file_name)
import streamlit as st
import dill
import nltk
from nltk.tokenize import word_tokenize

st.set_page_config(
    page_title="Short text POS",
)

nltk.download('punkt')

with open("hmm_tagger.pkl", 'rb') as f:
    loaded_tagger = dill.load(f)


text = st.text_input("Insert a text to get the POS tags for it")

if text:
    tokens = word_tokenize(text)
    tagged_sentence = loaded_tagger.tag(tokens)
    html_results = ""
    for word, tag in tagged_sentence:
        html_results += f"<span style='color:red;'>{word} </span><span style='color:blue;'>{tag}</span> "
    st.markdown(html_results, unsafe_allow_html=True)
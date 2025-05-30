import streamlit as st
from utils.model_loader import load_model

AVAILABLE_LANGUAGES = ["be", "bg", "cs", "hr", "mk", "pl", "ru", "sk", "sl", "sr", "uk"]

def setup_sidebar_and_model():
    st.sidebar.title("Language Selection")

    if "selected_language" not in st.session_state:
        st.session_state["selected_language"] = AVAILABLE_LANGUAGES[0]

    selected = st.sidebar.selectbox(
        "Select a model:",
        options=AVAILABLE_LANGUAGES,
        index=AVAILABLE_LANGUAGES.index(st.session_state["selected_language"])
    )

    st.session_state["selected_language"] = selected
    language = selected

    if "model" not in st.session_state:
        st.session_state["model"] = {}

    if language not in st.session_state["model"]:
        with st.spinner("Loading NER model... Please wait."):
            try:
                model = load_model(language)
                st.session_state["model"][language] = model
            except Exception as e:
                st.error(f"Failed to load the model: {e}")
                st.stop()
        st.rerun()  # ensures clean re-render after model is loaded

    return language, st.session_state["model"][language]

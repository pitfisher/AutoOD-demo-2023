from pathlib import Path
import PIL
import streamlit as st
import settings
import weapons_helper

def proctoring_demo():
    st.set_page_config(
        page_title="Proctoring",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.title("Proctoring")
    st.sidebar.header("ML Model Config")

    # confidence = float(st.sidebar.slider(
    #     "Select Model Confidence", 25, 100, 40)) / 100
    st.text("Work in progress")

proctoring_demo()

# show_code(plotting_demo)
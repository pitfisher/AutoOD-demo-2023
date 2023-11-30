from pathlib import Path
import PIL
import streamlit as st
import settings
import weapons_helper

def palms_demo():
    st.set_page_config(
        page_title="Palms detection",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.title("Palms detection")
    st.sidebar.header("ML Model Config")

    # confidence = float(st.sidebar.slider(
    #     "Select Model Confidence", 25, 100, 40)) / 100
    st.text("Work in progress")

palms_demo()

# show_code(plotting_demo)
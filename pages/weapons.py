from pathlib import Path
import PIL
import streamlit as st
import settings
import weapons_helper

def weapon_demo():
    st.set_page_config(
        page_title="Weapons detection",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.title("Weapons detection")
    st.sidebar.header("ML Model Config")

    # confidence = float(st.sidebar.slider(
    #     "Select Model Confidence", 25, 100, 40)) / 100

    model_path = Path(settings.WEAPONS_DETECTION_MODEL)
    config_path = Path(settings.WEAPONS_DETECTION_MODEL_CONFIG)

    try:
        model = weapons_helper.load_model(config_file=config_path, checkpoint_file=model_path)
    except Exception as ex:
        st.error(f"Unable to load model. Check the specified path: {model_path}")
        st.error(ex)

    # weapons_helper.play_stored_video(confidence, model)
    weapons_helper.play_stored_video(model)

weapon_demo()

# show_code(plotting_demo)
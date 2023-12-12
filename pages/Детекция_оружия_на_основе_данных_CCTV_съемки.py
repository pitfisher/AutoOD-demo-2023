from pathlib import Path
import PIL
import streamlit as st
import settings
import weapons_helper

def weapon_demo():
    st.set_page_config(
        page_title="Детекция оружия",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.title("Детекция оружия")
    st.sidebar.header("Настройки")

    # confidence = float(st.sidebar.slider(
    #     "Select Model Confidence", 25, 100, 40)) / 100

    model_path = Path(settings.WEAPONS_DETECTION_MODEL)
    config_path = Path(settings.WEAPONS_DETECTION_MODEL_CONFIG)

    try:
        model = weapons_helper.load_model(config_file=config_path, checkpoint_file=model_path)
    except Exception as ex:
        st.error(f"Unable to load model. Check the specified path: {model_path}")
        st.error(ex)

    st.sidebar.header("Настройки")
    source_radio = st.sidebar.radio("Выберите источник: ", settings.SOURCES_LIST)
    # weapons_helper.play_stored_video(confidence, model)
    if source_radio == settings.WEBCAM:
        weapons_helper.play_webcam(model)
    elif source_radio == settings.VIDEO:
        weapons_helper.play_stored_video(model)
    else:
        st.error("Please select a valid source type!")

weapon_demo()

# show_code(plotting_demo)

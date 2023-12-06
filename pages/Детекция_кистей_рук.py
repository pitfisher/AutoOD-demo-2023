from pathlib import Path
from PIL import Image
import numpy as np
import streamlit as st
import settings
import yolo_helper

def palms_demo():
    st.set_page_config(
        page_title="Palms detection",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.title("Palms detection")
    st.sidebar.header("ML Model Config")

    path_to_json_config = 'yolo_config.json'
    config_loader = yolo_helper.ConfigLoader(path_to_json_config)
    config = config_loader.get_config()

    image_file = st.file_uploader("Upload Your Image", type=['jpg', 'png', 'jpeg'])

    if not image_file:
        return None
    
    original_image = Image.open(image_file)
    original_image_np = np.asarray(original_image)
    translator = config["to_russian_palm"]
    confidence = config["palm_model_conf_v5"]
    weights = config["palm_detection_model_path_v5"]
    try:
        yolo_v5_class_obj = yolo_helper.YOLOv5Model()
        model = yolo_v5_class_obj.load_model(weights)
    except Exception as ex:
        st.error(f"Unable to load model. Check the specified path: {settings.BICYCLE_PARTS_MODEL}")
        st.error(ex)

    yolo_v5_class_obj.detect_objects(frame=original_image_np,
                                            model=model,
                                            current_model_conf=confidence,
                                            image_displayer=yolo_helper.ImageDisplayer(),
                                            labels_translator=translator)
    st.text("Detection results")
    col1, col2 = st.columns(2)
    col1.image(original_image, caption = "Original image")
    col2.image(original_image_np, caption = "Detection results")
    # st.image(cutout_images, clamp=True)

palms_demo()

# show_code(plotting_demo)

from pathlib import Path
from PIL import Image
import streamlit as st
import settings
import drone_helper
import numpy as np

big_model, small_model = drone_helper.init_models(settings.DRONE_BIG_MODEL, settings.DRONE_SMALL_MODEL)

def drone_demo():
    st.set_page_config(layout="wide")

    image_file = st.file_uploader("Upload Your Image", type=['jpg', 'png', 'jpeg'])
    if not image_file:
        return None

    original_image = Image.open(image_file)

    
     
    prepared_image = drone_helper.prepare_image(original_image)
    results, cutout_images, big_bboxes = drone_helper.inference(big_model, small_model, prepared_image)
    cutout_images = drone_helper.preprocess_cutout_images(cutout_images)

    processed_image = drone_helper.draw_bboxes(np.array(original_image), results, big_bboxes)

    col1, col2 = st.columns(2)
    st.text("Detected people")
    col1.image(original_image, caption = "Original image")
    col2.image(processed_image, caption = "Detection results")
    st.image(cutout_images, clamp=True)
    
drone_demo()
from pathlib import Path
from PIL import Image
import streamlit as st
import settings
import drone_helper
import numpy as np
from utils import iterator_from_images_directory
big_model, small_model = drone_helper.init_models(settings.DRONE_BIG_MODEL, settings.DRONE_SMALL_MODEL)

images_directory_path = Path("media/images/drone")

def drone_demo():
    st.set_page_config(layout="wide")

    source_type = st.sidebar.radio(
    "Тип источника", ['Файл', 'Папка'])

    if source_type == 'Файл':
        image_file = st.file_uploader("Upload Your Image", type=['jpg', 'png', 'jpeg'])
        if not image_file:
            return None

        original_image = Image.open(image_file)

        prepared_image = drone_helper.prepare_image(original_image)
        results, cutout_images, big_bboxes = drone_helper.inference(big_model, small_model, prepared_image)
        cutout_images = drone_helper.preprocess_cutout_images(cutout_images)

        processed_image = drone_helper.draw_bboxes(np.array(original_image), results, big_bboxes)

        col1, col2 = st.columns(2)
        st.text("Обнаруженные люди")
        col1.image(original_image, caption = "Исходное изображение")
        col2.image(processed_image, caption = "Результаты распознавания")
        st.image(cutout_images, clamp=True)
    elif source_type == 'Папка':
        st_frame = st.empty()
        for original_image in iterator_from_images_directory(images_directory_path):
                if not original_image:
                    return None

                prepared_image = drone_helper.prepare_image(original_image)
                results, cutout_images, big_bboxes = drone_helper.inference(big_model, small_model, prepared_image)
                cutout_images = drone_helper.preprocess_cutout_images(cutout_images)

                processed_image = drone_helper.draw_bboxes(np.array(original_image), results, big_bboxes)
                with st_frame:
                    st.text("Обнаруженные люди")
                    col1, col2 = st.columns(2)
                    col1.image(original_image, caption = "Исходное изображение")
                    col2.image(processed_image, caption = "Результаты распознавания")
                    # st.image(cutout_images, clamp=True)
    else:
        st.error("Please select a valid source type!")

drone_demo()

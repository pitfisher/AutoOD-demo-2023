from pathlib import Path
from PIL import Image
import streamlit as st
import settings
import drone_helper
import numpy as np
from helper import iterator_from_images_directory
from time import perf_counter, sleep
from statistics import mean

big_model, small_model = drone_helper.init_models(settings.DRONE_BIG_MODEL, settings.DRONE_SMALL_MODEL)

images_directory_path = Path("media/images/drone")
images_directory_path = Path("media/images/drone_closed_val")

inference_time_list = []

def drone_demo():
    st.set_page_config(
        page_title="Детекция людей с БПЛА",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.title("Детекция людей с БПЛА")
    st.sidebar.header("Настройки")
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
        st_frame1 = st.empty()
        st_frame2 = st.empty()
        for original_image in iterator_from_images_directory(images_directory_path):
                if not original_image:
                    return None
                
                prepared_image = drone_helper.prepare_image(original_image)
                timestamp_before_inference = perf_counter()
                results, cutout_images, big_bboxes = drone_helper.inference(big_model, small_model, prepared_image)
                inference_time = perf_counter() - timestamp_before_inference
                print(f"Inference time: {inference_time * 1000:.0f}ms")
                inference_time_list.append(inference_time)
                cutout_images = drone_helper.preprocess_cutout_images(cutout_images)
                processed_image = drone_helper.draw_bboxes(np.array(original_image), results, big_bboxes)
                with st_frame1:
                    st.text("Обнаруженные люди")
                    col1, col2 = st.columns(2)
                    col1.image(original_image, caption = "Исходное изображение")
                    col2.image(processed_image, caption = "Результаты распознавания")
                with st_frame2:
                    st.image(cutout_images, clamp=True)

        print(f"Mean inference time: {mean(inference_time_list) * 1000:.0f}ms")
    else:
        st.error("Please select a valid source type!")

drone_demo()

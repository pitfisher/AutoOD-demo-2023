from pathlib import Path
from PIL import Image
import numpy as np
import streamlit as st
import settings
import yolo_helper
import cv2

def gestures_demo():
    st.set_page_config(
        page_title="Gestures recognition",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.title("Gestures recognition")
    st.sidebar.header("ML Model Config")

    source_type = st.sidebar.radio(
    "Тип источника", ['Файл', 'Видео'])

    path_to_json_config = 'yolo_config.json'
    config_loader = yolo_helper.ConfigLoader(path_to_json_config)
    config = config_loader.get_config()

    translator = config["to_russian_bicycle_parts"]
    confidence = config["gestures_model_conf_v8"]
    weights = config["gestures_detection_model_path_v8"]
    try:
        yolo_v8_class_obj = yolo_helper.YOLOv8Model()
        model = yolo_v8_class_obj.load_model(weights)
    except Exception as ex:
        st.error(f"Unable to load model. Check the specified path: {config['gestures_detection_model_path_v8']}")
        st.error(ex)

    if source_type == 'Файл':
        image_file = st.file_uploader("Upload Your Image", type=['jpg', 'png', 'jpeg'])

        if not image_file:
            return None
        
        original_image = Image.open(image_file)
        original_image_np = np.asarray(original_image)

        yolo_v8_class_obj.detect_objects(frame=original_image_np,
                                                model=model,
                                                current_model_conf=confidence,
                                                image_size=720,
                                                image_displayer=yolo_helper.ImageDisplayer(),
                                                labels_translator=translator)
        st.text("Результаты распознавания")
        col1, col2 = st.columns(2)
        col1.image(original_image, caption = "Original image")
        col2.image(original_image_np, caption = "Detection results")
        # st.image(cutout_images, clamp=True)

    elif source_type == 'Видео':
        if st.sidebar.button('Начать распознавание'):
            try:
                vid_cap = cv2.VideoCapture(str(settings.VIDEOS_DICT.get("Жесты")))
                st_frame = st.empty()
                while vid_cap.isOpened():
                    success, image = vid_cap.read()
                    original_image = image.copy()
                    if success:
                        yolo_v8_class_obj.detect_objects(frame=image,
                                                                model=model,
                                                                current_model_conf=confidence,
                                                                image_size=720,
                                                                image_displayer=yolo_helper.ImageDisplayer(),
                                                                labels_translator=translator)
                        with st_frame:
                            st.text("Результаты распознавания")
                            col1, col2 = st.columns(2)
                            col1.image(original_image, caption = "Исходное изображение", channels="BGR", use_column_width=True)
                            col2.image(image, caption = "Результаты распознавания", channels="BGR", use_column_width=True)
                    else:
                        vid_cap.release()
                        break
            except Exception as e:
                st.sidebar.error("Error loading video: " + str(e))
    else:
        st.error("Please select a valid source type!")

gestures_demo()

# show_code(plotting_demo)

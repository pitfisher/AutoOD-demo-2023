from pathlib import Path
from PIL import Image
import numpy as np
import streamlit as st
import settings
import yolo_helper
import cv2

def palms_demo():
    st.set_page_config(
        page_title="Детекция кистей рук",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.title("Детекция кистей рук")
    st.sidebar.header("Настройки")

    path_to_json_config = 'yolo_config.json'
    config_loader = yolo_helper.ConfigLoader(path_to_json_config)
    config = config_loader.get_config()

    st.sidebar.header("Настройки")
    source_radio = st.sidebar.radio("Выберите источник: ", settings.SOURCES_LIST_PALMS)
    # weapons_helper.play_stored_video(confidence, model)

    translator = config["to_russian_palm"]
    confidence = config["palm_model_conf_v5"]
    weights = config["palm_detection_model_path_v5"]
    try:
        yolo_v5_class_obj = yolo_helper.YOLOv5Model()
        model = yolo_v5_class_obj.load_model(weights)
    except Exception as ex:
        st.error(f"Unable to load model. Check the specified path: {settings.BICYCLE_PARTS_MODEL}")
        st.error(ex)

    if source_radio == settings.IMAGE:
        image_file = st.file_uploader("Загрузите изображение", type=['jpg', 'png', 'jpeg'])

        if not image_file:
            return None
        
        original_image = Image.open(image_file)
        original_image_np = np.asarray(original_image)

        yolo_v5_class_obj.detect_objects(frame=original_image_np,
                                                model=model,
                                                current_model_conf=confidence,
                                                image_displayer=yolo_helper.ImageDisplayer(),
                                                labels_translator=translator)
        st.text("Обнаруженные люди")
        col1, col2 = st.columns(2)
        col1.image(original_image, caption = "Исходное изображение")
        col2.image(original_image_np, caption = "Результаты распознавания")
        # st.image(cutout_images, clamp=True)
    elif source_radio == settings.WEBCAM:
        source_webcam = settings.WEBCAM_ID

        try:
            vid_cap = cv2.VideoCapture(source_webcam)
            vid_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
            vid_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
            vid_cap.set(cv2.CAP_PROP_FPS, 30)
            st_frame = st.empty()
            while (vid_cap.isOpened()):
                success, image = vid_cap.read()
                if success:
                    original_image = np.copy(image)
                    yolo_v5_class_obj.detect_objects(frame= image,
                                                            model=model,
                                                            current_model_conf=confidence,
                                                            image_displayer=yolo_helper.ImageDisplayer(),
                                                            labels_translator=translator)
                    with st_frame:
                        st.text("Результаты распознавания")
                        col1, col2 = st.columns(2)
                        col1.image(original_image, caption = "Исходное изображение", channels="BGR", use_column_width=True)
                        col2.image(image, caption = "Результаты детекции", channels="BGR", use_column_width=True)
                else:
                    vid_cap.release()
                    break
        except Exception as e:
            st.sidebar.error("Error loading video: " + str(e))

        yolo_v5_class_obj.detect_objects(frame=original_image_np,
                                                model=model,
                                                current_model_conf=confidence,
                                                image_displayer=yolo_helper.ImageDisplayer(),
                                                labels_translator=translator)
        st.text("Обнаруженные люди")
        col1, col2 = st.columns(2)
        col1.image(original_image, caption = "Исходное изображение")
        col2.image(original_image_np, caption = "Результаты распознавания")
        # st.image(cutout_images, clamp=True)
    else:
        st.error("Please select a valid source type!")


palms_demo()

# show_code(plotting_demo)

import numpy as np

import streamlit as st
import cv2
import PIL

from mmdet.apis import DetInferencer
import settings
import torch


DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'


def load_model(config_file, checkpoint_file):
    print(config_file, checkpoint_file)
    return DetInferencer(str(config_file), str(checkpoint_file), device=DEVICE)


def _display_detected_frames(model, st_frame, image):
    image = cv2.resize(image, (720, int(720 * (9 / 16))))

    result = model(image, return_vis=True)
    # print("result", result)
    try:
        visualized_image_array = result['visualization'][0]
    except Exception as e:
        visualized_image_array = np.array(image)

    visualized_image =  PIL.Image.fromarray(cv2.cvtColor(visualized_image_array, cv2.COLOR_BGR2RGB))
    
    with st_frame:
        st.text("Результаты распознавания")
        col1, col2 = st.columns(2)
        col1.image(image, caption = "Исходное изображение", channels="BGR", use_column_width=True)
        col2.image(visualized_image, caption = "Результаты детекции", channels="BGR", use_column_width=True)
    # st.image(cutout_images, clamp=True)
    # st_frame.image(visualized_image, caption='Detected Video', channels="BGR", use_column_width=True)

def play_stored_video(model):
    source_vid = st.sidebar.selectbox("Выберите видео...", settings.VIDEOS_DICT.keys())

    with open(settings.VIDEOS_DICT.get(source_vid), 'rb') as video_file:
        video_bytes = video_file.read()
    if video_bytes:
        st.video(video_bytes)

    if st.sidebar.button('Начать распознавание'):
        try:
            vid_cap = cv2.VideoCapture(
                str(settings.VIDEOS_DICT.get(source_vid)))
            st_frame = st.empty()
            while vid_cap.isOpened():
                success, image = vid_cap.read()
                if success:
                    _display_detected_frames(model, st_frame, image)
                else:
                    vid_cap.release()
                    break
        except Exception as e:
            st.sidebar.error("Ошибка загрузки видео: " + str(e))

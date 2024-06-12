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


def _display_detected_frames(model, st_frame, image, out):
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
    return visualized_image_array
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
            # grab some parameters of video to use them for writing a new, processed video
            width = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            frame_fps = vid_cap.get(cv2.CAP_PROP_FPS)  ##<< No need for an int

            # specify a writer to write a processed video to a disk frame by frame
            fourcc_mp4 = cv2.VideoWriter_fourcc(*'mp4v')
            out_mp4 = cv2.VideoWriter('./weapons_output.mp4', fourcc_mp4, frame_fps, (width, height),isColor = False)
            print("creating a video")
            st_frame = st.empty()
            while vid_cap.isOpened():
                success, image = vid_cap.read()
                if success:
                    visualized_image_array = _display_detected_frames(model,
                                             st_frame,
                                             image,
                                             out_mp4,
                                             )
                    print(visualized_image_array.size)
                    out_mp4.write(image)
                else:
                    vid_cap.release()
                    break
        except Exception as e:
            st.sidebar.error("Ошибка загрузки видео: " + str(e))
        out_mp4.release()
        print("video was saved")
        vid_cap.release()

def play_webcam(model):
    source_webcam = settings.WEBCAM_ID
    if st.sidebar.button('Начать распознавание'):
        try:
            vid_cap = cv2.VideoCapture(source_webcam)
            st_frame = st.empty()
            while (vid_cap.isOpened()):
                success, image = vid_cap.read()
                if success:
                    _display_detected_frames(model,
                                             st_frame,
                                             image, 
                                             )
                else:
                    vid_cap.release()
                    break
        except Exception as e:
            st.sidebar.error("Error loading video: " + str(e))

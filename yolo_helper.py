from abc import ABC, abstractmethod
import cv2
import torch.cuda
import json
import numpy as np
from ultralytics import YOLO

class ConfigLoader:
    def __init__(self, file_path):
        self.file_path = file_path
        self.config = {}
        self.load_config()

    def load_config(self):
        """
        Загружает конфигурацию моделей из JSON файла
        """
        with open(self.file_path, 'r', encoding="UTF-8") as file:
            self.config = json.load(file)

    def get_config(self):
        """
        Возвращает загруженную конфигурацию
        """
        return self.config

class ObjectDetectionModel(ABC):
    @abstractmethod
    def load_model(self, path_to_model_weights, yolo_path):
        pass

    @abstractmethod
    def detect_objects(self, frame, model, current_model_conf, image_dislayer, labels_translator=None):
        pass

    @abstractmethod
    def run_model(self, translator, weights, cur_model_conf, captor, displayer, yolo_path=None):
        pass

    def one_object_display(array: np.ndarray, cur_obj: np.ndarray, des: str) -> None:
        """
        Отрисовка прямоугольника задетектированного объекта
        :param array: полотно для вывода
        :param cur_obj: координаты задетектированного объекта
        :param des: название задетектированного объекта
        :return: None
        """
        cv2.rectangle(array, (int(cur_obj[0]), int(cur_obj[1])),
                      (int(cur_obj[2]), int(cur_obj[3])), (255, 255, 0), 2)
        cv2.putText(array, des, (int(cur_obj[0]), int(cur_obj[1])),
                    cv2.FONT_HERSHEY_COMPLEX,
                    0.7, (0, 0, 0), 1, lineType=cv2.LINE_AA)
        return None

class YOLOv5Model(ObjectDetectionModel):
    def load_model(self, path_to_model_weights):
        current_model = torch.hub.load("yolov5_v7.0", 'custom', path=path_to_model_weights, source='local',
                                       verbose=False)
        return current_model

    def detect_objects(self, frame, model, current_model_conf, image_displayer, labels_translator=None):
        palm_nn_result = model(frame, size=1920)  # обработка кадра моделью
        for item in palm_nn_result.pandas().xyxy:
            for obj in item.values:  # делаем из относительных координат абсолютные с учетом первоначального сдвига
                if obj[4] > current_model_conf:
                    translate = labels_translator
                    if str(int(obj[5])) in translate:
                        descr = translate[str(int(obj[5]))]
                        image_displayer.draw_single_bbox(frame, obj, str(descr + ' ' + str(round(obj[4], 2))))

    def run_model(self, weights, cur_model_conf, captor, displayer, yolo_path = None, translator = None):
        yolo_v5_class_obj = YOLOv5Model()
        model = yolo_v5_class_obj.load_model(weights, yolo_path)
        while True:
            ret, frame = captor.get_frame()
            yolo_v5_class_obj.detect_objects(frame=frame,
                                             model=model,
                                             current_model_conf=cur_model_conf,
                                             image_dislayer=displayer,
                                             labels_translator=translator)
            displayer.display_frame_with_labels(frame)

            
            pressed_key = cv2.waitKey(1)
            if pressed_key & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break

class YOLOv8Model(ObjectDetectionModel):
    def __init__(self):
        pass

    def load_model(self, path_to_model_weights, yolo_path=None):
        current_model = YOLO(path_to_model_weights)
        return current_model

    def detect_objects(self, frame, model, current_model_conf, image_size, image_displayer, labels_translator=None):
        details_nn_results = model.predict(frame, imgsz=image_size)
        for item in details_nn_results:
            boxes = item.boxes.cpu().numpy()  # get boxes on cpu in numpy
            for obj in boxes.data:
                if obj[4] > current_model_conf:
                    translate = labels_translator
                    if str(int(obj[5])) in translate:
                        descr = translate[str(int(obj[5]))]
                        image_displayer.draw_single_bbox(frame, obj, str(descr + ' ' + str(round(obj[4], 2))))

    def run_model(self, weights, cur_model_conf, captor, displayer, yolo_path=None, translator = None):
        yolo_v8_class_obj = YOLOv8Model()
        model = yolo_v8_class_obj.load_model(weights)
        while True:
            ret, frame = captor.get_frame()
            yolo_v8_class_obj.detect_objects(frame=frame,
                                             model=model,
                                             current_model_conf=cur_model_conf,
                                             image_displayer=displayer,
                                             labels_translator=translator)
            displayer.display_frame_with_labels(frame)
            pressed_key = cv2.waitKey(1)
            if pressed_key & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break

class ImageDisplayer:
    def __init__(self):
        pass

    def draw_single_bbox(self, array: np.ndarray, cur_obj: np.ndarray, des: str) -> None:
        """
        Отрисовка прямоугольника задетектированного объекта
        :param array: полотно для вывода
        :param cur_obj: координаты задетектированного объекта
        :param des: название задетектированного объекта
        :return: None
        """
        cv2.rectangle(array, (int(cur_obj[0]), int(cur_obj[1])),
                      (int(cur_obj[2]), int(cur_obj[3])), (255, 255, 0), 2)
        cv2.putText(array, des, (int(cur_obj[0]), int(cur_obj[1])),
                    cv2.FONT_HERSHEY_COMPLEX,
                    0.7, (0, 0, 0), 1, lineType=cv2.LINE_AA)
        return None
    
    def draw_bboxes():
        pass

    def display_frame_with_labels(self, frame):
        cv2.namedWindow('monitor', cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty('monitor', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.moveWindow('monitor', 0, 0)
        cv2.imshow("monitor", frame)  # отрисовка полученного кадра

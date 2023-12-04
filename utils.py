# utils.py
import os
from PIL import Image
from pathlib import Path
def iterator_from_images_directory(images_directory_path):
    image_paths_list = os.listdir(images_directory_path)
    return (Image.open(Path(images_directory_path, image_path)) for image_path in image_paths_list)
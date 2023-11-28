from pathlib import Path
import sys

file_path = Path(__file__).resolve()

root_path = file_path.parent

if root_path not in sys.path:
    sys.path.append(str(root_path))

ROOT = root_path.relative_to(Path.cwd())

VIDEO_DIR = ROOT / "media/videos/"
IMAGES_DIR = ROOT / 'media/images/'

# Images config
DEFAULT_IMAGE = IMAGES_DIR / '13.jpg'
DEFAULT_DETECT_IMAGE = IMAGES_DIR / 'output/vis/13.jpg'

# Videos config
VIDEO_1_PATH = VIDEO_DIR / 'weapon1_1.mp4'
VIDEO_2_PATH = VIDEO_DIR / 'weapon2_2.mp4'
VIDEO_3_PATH = VIDEO_DIR / 'balabanov.webm'
VIDEOS_DICT = {
    'weapon1_1': VIDEO_1_PATH,
    'weapon2_2': VIDEO_2_PATH,
    'balabanov': VIDEO_3_PATH,
}

# WEAPONS_DETECTION_MODEL = ROOT / "weights/weapons_best2.pt"
WEAPONS_DETECTION_MODEL = ROOT / "weights/best_coco_bbox_mAP_epoch_250.pth"
WEAPONS_DETECTION_MODEL_CONFIG = ROOT / 'weights/rtmdet_l_8xb32-300e_coco_custom.py'

# path to FasterRCNN model
DRONE_BIG_MODEL = ROOT / "weights/drone_big_model.ckpt"
# path to SSD MobileNetv3 model
DRONE_SMALL_MODEL = ROOT / "weights/drone_small_model.ckpt"

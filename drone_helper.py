import os
import cv2
import torch
import numpy as np
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

detected_cutouts = []
# stub black images
black_image = np.zeros((240,240))
detected_cutouts.append(black_image)
detected_cutouts.append(black_image)
detected_cutouts.append(black_image)

# prepare uploaded image for inference
def prepare_image(image_file):
    transform = torchvision.transforms.ToTensor()
    image = transform(image_file)
    image = image.unsqueeze(0)
    return image
    # image = []
    # image.append(original_image)

# load object detection models from disk
def init_models(big_model_path, small_model_path):
    big_model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False, pretrained_backbone=False)
    in_features = big_model.roi_heads.box_predictor.cls_score.in_features
    big_model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)

    s_d = torch.load(big_model_path)
    big_model.load_state_dict(s_d)
    big_model.eval()
    big_model = big_model.to("cuda")

    small_model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(pretrained=False, pretrained_backbone=False)
    in_features = small_model.roi_heads.box_predictor.cls_score.in_features
    small_model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)

    s_d = torch.load(small_model_path)
    small_model.load_state_dict(s_d)
    del s_d
    small_model.eval()
    small_model = small_model.to("cuda")
    return big_model, small_model

# perform object detection on user-loaded image
def inference(big_model, small_model, image):
    test_preds = []
    cutout_images = []
    big_bboxes = []
    with torch.no_grad():
            # image = torchvision.transforms.functional.convert_image_dtype(original_image.permute(0, 3, 1, 2), torch.float)
            outputs = big_model(image.to("cuda"))
            outputs = [{k: v.to('cpu') for k, v in t.items()} for t in outputs]

            boxes = outputs[0]['boxes'].data.numpy()
            scores = outputs[0]['scores'].data.numpy()
            boxes = boxes[scores >= 0.75].astype(np.int32)

            t_preds = {
                "boxes": [],
                "labels": [],
                "scores": [],
                "large_box_probs": [],
            }

            for box_ind in range(len(boxes)):
                box = boxes[box_ind]
                box_score = scores[box_ind]
                if box_score >= 0.5:
                    big_bboxes.append(box)
                xmin, ymin, xmax, ymax = box
                dx = max(0, 640 - (xmax - xmin))
                dy = max(0, 640 - (ymax - ymin))
                n_box = np.array([
                    max(xmin - dx/2, 0), max(ymin - dy/2, 0),
                    min(xmax + dx/2, image.shape[3]), min(ymax + dy/2, image.shape[2]),
                ]).astype(np.int32)
                # n_box = np.array([
                #     xmin, ymin, xmax, ymax
                # ]).astype(np.int32)
                cutout_images.append(image[:, :, box[1]:box[3], box[0]:box[2]])
                t_img = image[:, :, n_box[1]:n_box[3], n_box[0]:n_box[2]]
                small_outputs = small_model(t_img.to("cuda"))
                small_outputs = [{k: v.to('cpu') for k, v in t.items()} for t in small_outputs]
                for out in small_outputs:
                    for pred_box in out['boxes']:
                        # pred_box[0] = (pred_box[0] + n_box[0])
                        # pred_box[1] = (pred_box[1] + n_box[1])
                        # pred_box[2] = (pred_box[2] + n_box[0])
                        # pred_box[3] = (pred_box[3] + n_box[1])
                        pred_box[0] = (pred_box[0] + n_box[0]) / image.shape[3]
                        pred_box[1] = (pred_box[1] + n_box[1]) / image.shape[2]
                        pred_box[2] = (pred_box[2] + n_box[0]) / image.shape[3]
                        pred_box[3] = (pred_box[3] + n_box[1]) / image.shape[2]
                t_preds['boxes'].extend(small_outputs[0]['boxes'].numpy())
                t_preds['labels'].extend(small_outputs[0]['labels'].numpy())
                t_preds['scores'].extend(small_outputs[0]['scores'].numpy())
                t_preds['large_box_probs'].extend([box_score] * len(small_outputs[0]['boxes']))

            t_preds['boxes'] = np.array(t_preds['boxes'])
            t_preds['labels'] = np.array(t_preds['labels'])
            t_preds['scores'] = np.array(t_preds['scores'])
            t_preds['large_box_probs'] = np.array(t_preds['large_box_probs'])
            test_preds.append(t_preds)

    results = []
    for i, pred in enumerate(test_preds):
        fname = "1_000160"
        image_id = os.path.splitext(os.path.basename(fname))[0]
        for n, box in enumerate(pred["boxes"]):
            if pred["scores"][n] > 0.95:
                xmin, ymin, xmax, ymax = box
                result = {
                    'image_id': image_id,
                    'xc': round((xmin + xmax) / 2, 4),
                    'yc': round((ymin + ymax) / 2, 4),
                    'w': round((xmax - xmin), 4),
                    'h': round((ymax - ymin), 4),
                    'label': 0,
                    'score': round(pred["scores"][n], 4)
                }
                results.append(result)
    return results, cutout_images, big_bboxes


# preprocess big bbox regions to look better when shown
def preprocess_cutout_images(images):
    processed_cutout_images = []
    for image in images:
        image = np.array(image.squeeze(0).swapaxes(0,2).swapaxes(0,1))
        image = cv2.copyMakeBorder(image, 3, 3, 3, 3, cv2.BORDER_CONSTANT, value=[255,255,255])
        processed_cutout_images.append(image)
    return processed_cutout_images

# draw big and small bboxes on original image
def draw_bboxes(image, results, big_bboxes):
    for result, big_bbox in zip(results, big_bboxes):
        xmin = result["xc"] - result["w"] / 2
        ymin = result["yc"] - result["h"] / 2
        xmax = result["xc"] + result["w"] / 2
        ymax = result["yc"] + result["h"] / 2
        box = [xmin * image.shape[1], ymin * image.shape[0], xmax * image.shape[1], ymax * image.shape[0]]
        color = [245, 85, 121]
        color_big = [0, 255, 0]
        cv2.rectangle(image,
                    (int(box[0]), int(box[1])),
                    (int(box[2]), int(box[3])),
                    color, 20)
        cv2.rectangle(image,
                    (int(big_bbox[0]), int(big_bbox[1])),
                    (int(big_bbox[2]), int(big_bbox[3])),
                    color_big, 20)
    return image
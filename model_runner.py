import threading
from pathlib import Path

import torch

from models import *

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def start_test(model_name, media_path, media_type, device_type, model_type):
    t1 = threading.Thread(
        target=run_detection,
        args=(model_name, media_path, media_type, device_type, model_type),
    )
    t1.start()


def run_detection(model_name, media_path, media_type, device_type, model_type):
    model_name = model_name.lower()
    media_path = Path(media_path)
    device = "cuda" if device_type == "GPU" else "cpu"

    match (model_name, media_type, model_type):
        case ("yolov11", "videos", "coco"):
            run_yolo_coco_videos(model_name, media_path, device)
        case ("yolov11", "images", "coco"):
            run_yolo_coco_images(model_name, media_path, device)
        case ("yolov12", "videos", "coco"):
            run_yolo_coco_videos(model_name, media_path, device)
        case ("yolov12", "images", "coco"):
            run_yolo_coco_images(model_name, media_path, device)

        case ("yolov11", "videos", "custom"):
            run_yolo_custom_videos(model_name, media_path, device)
        case ("yolov11", "images", "custom"):
            run_yolo_custom_images(model_name, media_path, device)
        case ("yolov12", "videos", "custom"):
            run_yolo_custom_videos(model_name, media_path, device)
        case ("yolov12", "images", "custom"):
            run_yolo_custom_images(model_name, media_path, device)

        case ("ssd", "videos", "coco"):
            run_ssd_coco_videos(media_path, device)
        case ("ssd", "images", "coco"):
            run_ssd_coco_images(media_path, device)

        case ("ssd", "videos", "custom"):
            run_ssd_custom_videos(media_path, device)
        case ("ssd", "images", "custom"):
            run_ssd_custom_images(media_path, device)

        case ("efficientdet", "videos", "coco"):
            run_efficientdet_coco_videos(media_path, device)
        case ("efficientdet", "images", "coco"):
            run_efficientdet_coco_images(media_path, device)

        case ("efficientdet", "videos", "custom"):
            run_efficientdet_custom_videos(media_path, device)
        case ("efficientdet", "images", "custom"):
            run_efficientdet_custom_images(media_path, device)

        case ("rt-detrv2", "videos", "coco"):
            run_rtdetrv2_coco_videos(media_path, device)
        case ("rt-detrv2", "images", "coco"):
            run_rtdetrv2_coco_images(media_path, device)

        case ("rt-detrv2", "videos", "custom"):
            run_rtdetrv2_custom_videos(media_path, device)
        case ("rt-detrv2", "images", "custom"):
            run_rtdetrv2_custom_images(media_path, device)

        case _:
            return

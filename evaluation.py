import threading
from pathlib import Path

import torch

from coco_models import *

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def start_test(model_name, media_path, media_type, device_type):
    if media_type == "videos":
        t1 = threading.Thread(
            target=run_videos, args=(model_name, media_path, device_type)
        )
        t1.start()
    elif media_type == "images":
        t1 = threading.Thread(
            target=run_images, args=(model_name, media_path, device_type)
        )
        t1.start()


def run_videos(model_name, media_path, device_type):
    model_name = model_name.lower()
    media_path = Path(media_path)
    device = "cuda" if device_type == "GPU" else "cpu"
    match model_name:
        case "yolov11":
            run_yolo_for_videos(model_name, media_path, device)
        case "yolov12":
            run_yolo_for_videos(model_name, media_path, device)
        case "ssd":
            run_ssd_for_videos(media_path, device)
        case "efficientdet":
            run_efficientdet_for_videos(media_path, device)
        case "rt-detrv2":
            run_rtdetrv2_for_videos(media_path, device)
        case _:
            return


def run_images(model_name, media_path, device_type):
    model_name = model_name.lower()
    media_path = Path(media_path)
    device = "cuda" if device_type == "GPU" else "cpu"
    match model_name:
        case "yolov11":
            run_yolo_for_images(model_name, media_path, device)
        case "yolov12":
            run_yolo_for_images(model_name, media_path, device)
        case "ssd":
            run_ssd_for_images(media_path, device)
        case "efficientdet":
            run_efficientdet_for_images(media_path, device)
        case "rt-detrv2":
            run_rtdetrv2_for_images(media_path, device)
        case _:
            return

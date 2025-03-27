from ultralytics import YOLO
import matplotlib.pyplot as plt
from pathlib import Path
import threading


def start_test(model_name, media_path):
    t1 = threading.Thread(target=run, args=(model_name, media_path))
    t1.start()


def run(model_name, media_path):
    # print(model_name)
    # print(Path(media_path) / "00279_jpg.rf.a00e60b3f6960a08073ad2ec452da28e.jpg")
    model_name = model_name.lower()
    media_path = Path(media_path) / "00279_jpg.rf.a00e60b3f6960a08073ad2ec452da28e.jpg"
    if model_name == "yolov11":
        model = YOLO("./models/yolo11n.pt")
        results = model(media_path)
        results[0].show()

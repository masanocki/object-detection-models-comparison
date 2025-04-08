import time
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO

from metrics import *


def run_yolo_for_videos(model_name, media_path, device):
    if model_name == "yolov11":
        path = Path("./saved_models/yolo11n.pt")
    elif model_name == "yolov12":
        path = Path("./saved_models/yolo12n.pt")

    results_data = []
    model = YOLO(path).to(device)
    for video in media_path.glob("*.avi"):
        if device == "cuda":
            torch.cuda.empty_cache()

        cap = cv2.VideoCapture(video)
        frame_count = 0
        frame_times = []
        start_time = time.time()

        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                frame_start_time = time.time()
                results = model.track(frame, persist=True)

                if device == "cuda":
                    torch.cuda.synchronize()

                frame_time = time.time() - frame_start_time
                frame_times.append(frame_time)
                frame_count += 1

                annotated_frame = results[0].plot()

                cv2.imshow(f"{model_name} Tracking", annotated_frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            else:
                break
        cap.release()
        cv2.destroyAllWindows()

        if device == "cuda":
            del results
            torch.cuda.empty_cache()

        total_time = time.time() - start_time
        metrics = calculate_fps_and_time(frame_times, total_time, frame_count)

        results_data.append(
            {
                "video_name": video.name,
                "min_fps": metrics["min_fps"],
                "avg_fps": metrics["avg_fps"],
                "max_fps": metrics["max_fps"],
                "total_detection_time": total_time,
                "avg_frame_time": metrics["avg_frame_time"],
                "device": device,
                "frames_processed": frame_count,
            }
        )
    print(results_data)


def run_yolo_for_images(model_name, media_path, device):
    pass

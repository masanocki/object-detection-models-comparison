import time
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO

from models.utils.metrics import *
from models.utils.helpers import *


def run_yolo_custom_videos(model_name, media_path, device, sport_type, gui):
    path = get_correct_custom_model(sport_type, model_name)

    model = YOLO(path).to(device)

    results_data = []
    global_start_time = time.time()

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

                current_video_time = time.time() - start_time
                total_processing_time = time.time() - global_start_time

                ### METRICS VISUALIZER UPDATE ###
                if gui.metric_fps_checkbox.get():
                    current_fps = 1 / frame_time if frame_time > 0 else 0
                    gui.update_metric("FPS", f"{current_fps:.1f}")

                if gui.metric_detection_time_checkbox.get():
                    gui.update_metric(
                        "Video Detection Time", f"{current_video_time:.1f} s"
                    )
                    gui.update_metric("Total Time", f"{total_processing_time:.1f} s")
                ### METRICS VISUALIZER UPDATE ###

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


def run_yolo_custom_images(model_name, media_path, device, sport_type, gui):
    path = get_correct_custom_model(sport_type, model_name)

    model = YOLO(path).to(device)

    if device == "cuda":
        torch.cuda.empty_cache()

    results_data = []
    frame_times = []
    start_time = time.time()
    processed_count = 0

    for image_path in media_path.glob("*.jpg"):
        image = cv2.imread(image_path)

        frame_start_time = time.time()
        results = model.predict(image)

        if device == "cuda":
            torch.cuda.synchronize()

        frame_time = time.time() - frame_start_time
        frame_times.append(frame_time)
        processed_count += 1

        annotated_frame = results[0].plot()

        cv2.imshow(f"{model_name} Detection", annotated_frame)

        cv2.waitKey(1)

    cv2.destroyAllWindows()
    if device == "cuda":
        del results
        torch.cuda.empty_cache()

    total_time = time.time() - start_time
    metrics = calculate_fps_and_time(frame_times, total_time, processed_count)

    results_data.append(
        {
            "folder_path": media_path,
            "total_detection_time": total_time,
            "avg_image_time": metrics["avg_frame_time"],
            "device": device,
            "images_processed": processed_count,
        }
    )

    print(results_data)

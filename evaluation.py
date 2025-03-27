from ultralytics import YOLO
import matplotlib.pyplot as plt
from pathlib import Path
import threading
import cv2
import time
import torch


def start_test(model_name, media_path, media_type):
    if media_type == "videos":
        t1 = threading.Thread(target=run_videos, args=(model_name, media_path))
        t1.start()


def run_videos(model_name, media_path):
    model_name = model_name.lower()
    media_path = Path(media_path)
    match model_name:
        case "yolov11":
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = YOLO("./models/yolo11n.pt").to(device)

            results_data = []

            for video in media_path.glob("*.avi"):
                cap = cv2.VideoCapture(video)

                frame_count = 0
                total_detection_time = 0
                fps_list = []

                start_time = time.time()

                while cap.isOpened():
                    ret, frame = cap.read()
                    if ret:

                        frame_start_time = time.time()
                        results = model.track(frame, persist=True)
                        frame_detection_time = time.time() - frame_start_time

                        fps = (
                            1 / frame_detection_time if frame_detection_time > 0 else 0
                        )
                        fps_list.append(fps)

                        total_detection_time += frame_detection_time
                        frame_count += 1

                        annotated_frame = results[0].plot()

                        cv2.imshow("YOLOv11 Tracking", annotated_frame)
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            break
                    else:
                        break
                cap.release()
                cv2.destroyAllWindows()

                min_fps = min(fps_list) if fps_list else 0
                max_fps = max(fps_list) if fps_list else 0
                avg_fps = sum(fps_list) / len(fps_list) if fps_list else 0
                avg_frame_time = (
                    total_detection_time / frame_count if frame_count > 0 else 0
                )
                total_time = time.time() - start_time

                results_data.append(
                    {
                        "video_name": video.name,
                        "min_fps": min_fps,
                        "avg_fps": avg_fps,
                        "max_fps": max_fps,
                        "total_detection_time": total_time,
                        "avg_frame_time": avg_frame_time,
                        "model": model.device.type,
                    }
                )
            print(results_data)
        case "yolov12":
            print(model_name)
        case "ssd":
            print(model_name)
        case "fasterrcnn":
            print(model_name)
        case "efficientdet":
            print(model_name)
        case "detr":
            print(model_name)
        case "retinanet":
            print(model_name)
        case _:
            return

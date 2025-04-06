from ultralytics import YOLO
import matplotlib.pyplot as plt
from pathlib import Path
import threading
import cv2
import time
import torch
import mediapipe as mp
import tensorflow as tf
import numpy as np
from transformers import RTDetrV2ForObjectDetection, AutoImageProcessor
from PIL import Image

torch.set_float32_matmul_precision("high")


def start_test(model_name, media_path, media_type, device_type):
    if media_type == "videos":
        t1 = threading.Thread(
            target=run_videos, args=(model_name, media_path, device_type)
        )
        t1.start()


def run_videos(model_name, media_path, device_type):
    model_name = model_name.lower()
    media_path = Path(media_path)
    device = "cuda" if device_type == "GPU" else "cpu"
    match model_name:
        case "yolov11":
            run_yolo(model_name, media_path, device)
        case "yolov12":
            run_yolo(model_name, media_path, device)
        case "ssd":
            run_se(model_name, media_path, device)
        case "fasterrcnn":
            run_frcnn(model_name, media_path, device)
        case "efficientdet":
            run_se(model_name, media_path, device)
        case "detr":
            run_rtdetrv2(media_path, device)
        case "retinanet":
            print(model_name)
        case _:
            return


def run_yolo(model, media_path, device):
    if model == "yolov11":
        path = Path("./models/yolo11n.pt")
    elif model == "yolov12":
        path = Path("./models/yolo12n.pt")

    model = YOLO(path).to(device)
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

                fps = 1 / frame_detection_time
                fps_list.append(fps)
                total_detection_time += frame_detection_time
                frame_count += 1

                annotated_frame = results[0].plot()
                cv2.imshow(f"{model} Tracking", annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            else:
                break
        cap.release()
        cv2.destroyAllWindows()

        min_fps = min(fps_list)
        max_fps = max(fps_list)
        avg_fps = sum(fps_list) / len(fps_list)
        avg_frame_time = total_detection_time / frame_count
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


def run_se(model, media_path, device):
    if model == "ssd":
        path = Path("./models/ssd_mobilenet_v2.tflite")
    elif model == "efficientdet":
        path = Path("./models/efficientdet_lite0.tflite")
    # TODO: FASTER RCNN OR SOME OTHER ALGORITHM

    BaseOptions = mp.tasks.BaseOptions
    ObjectDetector = mp.tasks.vision.ObjectDetector
    ObjectDetectorOptions = mp.tasks.vision.ObjectDetectorOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = ObjectDetectorOptions(
        base_options=BaseOptions(model_asset_path=path),
        max_results=5,
        running_mode=VisionRunningMode.VIDEO,
    )

    results_data = []
    for video in media_path.glob("*.avi"):
        cap = cv2.VideoCapture(video)

        frame_count = 0
        with ObjectDetector.create_from_options(options) as detector:
            frame_count = 0
            total_detection_time = 0
            fps_list = []
            start_time = time.time()
            video_fps = cap.get(cv2.CAP_PROP_FPS)
            timestamp_ms = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    mp_image = mp.Image(
                        image_format=mp.ImageFormat.SRGB, data=frame_rgb
                    )

                    frame_start_time = time.time()
                    detection_result = detector.detect_for_video(
                        mp_image, timestamp_ms=timestamp_ms
                    )
                    frame_detection_time = time.time() - frame_start_time

                    fps = 1 / frame_detection_time
                    fps_list.append(fps)
                    total_detection_time += frame_detection_time
                    frame_count += 1
                    timestamp_ms = int((frame_count / video_fps) * 1000)

                    for detection in detection_result.detections:
                        bbox = detection.bounding_box
                        x1 = int(bbox.origin_x)
                        y1 = int(bbox.origin_y)
                        x2 = int(bbox.origin_x + bbox.width)
                        y2 = int(bbox.origin_y + bbox.height)

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    if detection.categories:
                        category = detection.categories[0]
                        class_name = category.category_name
                        score = category.score
                        cv2.putText(
                            frame,
                            f"{class_name}: {score:.2f}",
                            (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 255, 0),
                            2,
                        )

                    cv2.imshow("Object Detection", frame)

                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                else:
                    break
        min_fps = min(fps_list)
        max_fps = max(fps_list)
        avg_fps = sum(fps_list) / len(fps_list)
        avg_frame_time = total_detection_time / frame_count
        total_time = time.time() - start_time
        results_data.append(
            {
                "video_name": video.name,
                "min_fps": min_fps,
                "avg_fps": avg_fps,
                "max_fps": max_fps,
                "total_detection_time": total_time,
                "avg_frame_time": avg_frame_time,
            }
        )
        cap.release()
        cv2.destroyAllWindows()
    print(results_data)


def run_frcnn(model, media_path, device):
    pass


def run_rtdetrv2(media_path, device):
    processor = AutoImageProcessor.from_pretrained("PekingU/rtdetr_v2_r18vd")
    model = RTDetrV2ForObjectDetection.from_pretrained("PekingU/rtdetr_v2_r18vd")
    model.eval()

    for video in media_path.glob("*.avi"):
        cap = cv2.VideoCapture(video)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            inputs = processor(images=image, return_tensors="pt")

            with torch.no_grad():
                outputs = model(**inputs)

            target_sizes = torch.tensor([image.size[::-1]])
            results = processor.post_process_object_detection(
                outputs, target_sizes=target_sizes
            )[0]

            for score, label, box in zip(
                results["scores"], results["labels"], results["boxes"]
            ):
                if score > 0.5:
                    box = [int(i) for i in box.tolist()]
                    label_text = f"{model.config.id2label[label.item()]}: {score:.2f}"
                    cv2.rectangle(
                        frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2
                    )
                    cv2.putText(
                        frame,
                        label_text,
                        (box[0], box[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        2,
                    )

            cv2.imshow(f"RT-DETRv2", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        cap.release()
        cv2.destroyAllWindows()

import time

import cv2
import torch
from PIL import Image
from transformers import AutoModelForObjectDetection, AutoProcessor

from models.utils.metrics import *

from models.utils.helpers import get_correct_custom_model


def run_rtdetrv2_custom_videos(media_path, device, sport_type, gui):

    path = get_correct_custom_model(sport_type, "rtdetrv2")

    processor = AutoProcessor.from_pretrained("PekingU/rtdetr_v2_r18vd", use_fast=True)
    model = AutoModelForObjectDetection.from_pretrained(path).to(device)
    model.eval()

    results_data = []
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

                image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

                inputs = processor(images=image, return_tensors="pt").to(device)

                with torch.no_grad():
                    outputs = model(**inputs)

                target_sizes = torch.tensor([image.size[::-1]])
                results = processor.post_process_object_detection(
                    outputs, target_sizes=target_sizes
                )[0]

                if device == "cuda":
                    torch.cuda.synchronize()

                frame_time = time.time() - frame_start_time
                frame_times.append(frame_time)
                frame_count += 1

                for score, label, box in zip(
                    results["scores"], results["labels"], results["boxes"]
                ):
                    if score > 0.5:
                        box = [int(i) for i in box.tolist()]
                        label_text = (
                            f"{model.config.id2label[label.item()]}: {score:.2f}"
                        )
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
            else:
                break
        cap.release()
        cv2.destroyAllWindows()

        if device == "cuda":
            del outputs
            del inputs
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


def run_rtdetrv2_custom_images(media_path, device, sport_type, gui):
    path = get_correct_custom_model(sport_type, "rtdetrv2")

    processor = AutoProcessor.from_pretrained("PekingU/rtdetr_v2_r18vd", use_fast=True)
    model = AutoModelForObjectDetection.from_pretrained(path).to(device)
    model.eval()

    if device == "cuda":
        torch.cuda.empty_cache()

    results_data = []
    frame_times = []
    start_time = time.time()
    processed_count = 0

    for image_path in media_path.glob("*.jpg"):
        image = cv2.imread(image_path)

        frame_start_time = time.time()

        image_processed = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        inputs = processor(images=image_processed, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model(**inputs)

        target_sizes = torch.tensor([image_processed.size[::-1]])
        results = processor.post_process_object_detection(
            outputs, target_sizes=target_sizes
        )[0]

        if device == "cuda":
            torch.cuda.synchronize()

        frame_time = time.time() - frame_start_time
        frame_times.append(frame_time)
        processed_count += 1

        for score, label, box in zip(
            results["scores"], results["labels"], results["boxes"]
        ):
            if score > 0.5:
                box = [int(i) for i in box.tolist()]
                label_text = f"{model.config.id2label[label.item()]}: {score:.2f}"
                cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
                cv2.putText(
                    image,
                    label_text,
                    (box[0], box[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2,
                )

        cv2.imshow(f"RT-DETRv2", image)
        cv2.waitKey(1)

    cv2.destroyAllWindows()

    if device == "cuda":
        del outputs
        del inputs
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

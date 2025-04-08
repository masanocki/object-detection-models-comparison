import json
import threading
import time
from pathlib import Path

import cv2
import torch
from effdet import DetBenchPredict, create_model
from PIL import Image
from torchvision.models.detection import SSD300_VGG16_Weights, ssd300_vgg16
from torchvision.transforms import functional as F
from torchvision.transforms.functional import to_tensor
from transformers import AutoImageProcessor, RTDetrV2ForObjectDetection
from ultralytics import YOLO

# from visualizer import *

# torch.set_float32_matmul_precision("high")
# if torch.cuda.is_available():
#     # Enable CUDA memory allocation caching
#     torch.cuda.empty_cache()
#     # Set memory allocation strategy
#     # torch.cuda.set_per_process_memory_fraction(0.8)  # Use 80% of available GPU memory
#     # Enable TF32 precision
#     torch.backends.cuda.matmul.allow_tf32 = True
#     torch.backends.cudnn.allow_tf32 = True


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
            run_ssd(media_path, device)
        case "efficientdet":
            run_effdet(media_path, device)
        case "rt-detrv2":
            run_rtdetrv2(media_path, device)
        case _:
            return


def run_yolo(model_name, media_path, device):
    if model_name == "yolov11":
        path = Path("./pretrained_models/yolo11n.pt")
    elif model_name == "yolov12":
        path = Path("./pretrained_models/yolo12n.pt")

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

                cv2.imshow(f"{model_name} Tracking", annotated_frame)

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
                "device": device,
            }
        )
    print(results_data)


def run_ssd(media_path, device):
    with open("./coco_classnames.json", "r") as f:
        class_names = {int(v): k for k, v in json.load(f).items()}

    model = ssd300_vgg16(weights=SSD300_VGG16_Weights.DEFAULT)
    model = model.to(device)
    model.eval()

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
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = F.to_tensor(rgb_frame).to(device)

                frame_start_time = time.time()
                with torch.no_grad():
                    predictions = model([image])
                frame_detection_time = time.time() - frame_start_time

                fps = 1 / frame_detection_time
                fps_list.append(fps)
                total_detection_time += frame_detection_time
                frame_count += 1

                boxes = predictions[0]["boxes"].cpu()
                scores = predictions[0]["scores"].cpu()
                labels = predictions[0]["labels"].cpu()

                confidence_threshold = 0.5
                mask = scores > confidence_threshold
                boxes = boxes[mask].numpy()
                scores = scores[mask].numpy()
                labels = labels[mask].numpy()

                for box, score, label in zip(boxes, scores, labels):
                    x1, y1, x2, y2 = map(int, box)
                    class_name = class_names.get(int(label), "Unknown")
                    label_text = f"{class_name}: {score:.2f}"

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(
                        frame,
                        label_text,
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        2,
                    )

                cv2.imshow("SSD300 Detection", frame)
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
                "device": device,
            }
        )
    print(results_data)


def run_effdet(media_path, device):
    with open("./coco_classnames.json", "r") as f:
        class_names = {int(v): k for k, v in json.load(f).items()}

    model = create_model("tf_efficientdet_d0", pretrained=True, num_classes=90)
    model = DetBenchPredict(model).to(device)
    model.eval()

    torch.backends.cudnn.benchmark = True
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
                original_height, original_width = frame.shape[:2]
                resized_frame = cv2.resize(frame, (512, 512))
                rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
                input_tensor = (
                    to_tensor(rgb_frame).unsqueeze(0).to(device, non_blocking=True)
                )

                # to testuje
                # if frame_count % 5 == 0:
                #     torch.cuda.empty_cache()

                frame_start_time = time.time()
                with torch.no_grad():
                    detections = model(input_tensor)
                    detections = detections[0].cpu()
                frame_detection_time = time.time() - frame_start_time

                fps = 1 / frame_detection_time
                fps_list.append(fps)
                total_detection_time += frame_detection_time
                frame_count += 1

                boxes = detections[:, :4].cpu()
                scores = detections[:, 4].cpu()
                labels = detections[:, 5].cpu()

                # to testuje
                # del detections
                # del input_tensor
                # torch.cuda.empty_cache()

                confidence_threshold = 0.25
                keep = scores > confidence_threshold
                boxes = boxes[keep]
                scores = scores[keep]
                labels = labels[keep]

                # keep = nms(boxes, scores, iou_threshold=0.5)
                # boxes = boxes[keep]
                # scores = scores[keep]
                # labels = labels[keep]

                scale_x = original_width / 512
                scale_y = original_height / 512
                boxes[:, [0, 2]] *= scale_x
                boxes[:, [1, 3]] *= scale_y

                for box, score, label in zip(boxes, scores, labels):
                    x1, y1, x2, y2 = map(int, box.tolist())
                    class_name = class_names.get(int(label.item()), "Unknown")
                    label_text = f"{class_name}: {score.item():.2f}"

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(
                        frame,
                        label_text,
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        2,
                    )
                cv2.imshow("EfficientDet-D0", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            else:
                break
        torch.cuda.empty_cache()
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
                "device": device,
            }
        )
    print(results_data)


def run_rtdetrv2(media_path, device):
    processor = AutoImageProcessor.from_pretrained("PekingU/rtdetr_v2_r18vd")
    model = RTDetrV2ForObjectDetection.from_pretrained("PekingU/rtdetr_v2_r18vd").to(
        device
    )
    model.eval()

    for video in media_path.glob("*.avi"):
        cap = cv2.VideoCapture(video)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            inputs = processor(images=image, return_tensors="pt").to(device)

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

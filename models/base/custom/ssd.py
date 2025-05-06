import json
import time
from pathlib import Path

import cv2
import torch
from torchvision.models.detection import SSD300_VGG16_Weights, ssd300_vgg16
from torchvision.transforms.functional import to_tensor
from pycocotools.coco import COCO
from torchmetrics.detection.mean_ap import MeanAveragePrecision

from models.utils.metrics import *

from models.utils.helpers import get_correct_custom_model, visualize_effdet_ssd


def run_ssd_custom_videos(media_path, device, sport_type, gui):

    ### VISUALIZATION CHECKBOXES ###
    enable_visualization = gui.enable_visualization_var.get()
    show_boxes = gui.show_bounding_boxes_checkbox.get()
    show_scores = gui.show_confidence_scores_checkbox.get()
    show_labels = gui.show_labels_checkbox.get()
    total_test_files = gui.total_files.get()
    files_counter = 0
    ###

    path = get_correct_custom_model(sport_type, "ssd")

    with open(path.parents[1] / "class_names.json", "r") as f:
        sport_class_map = json.load(f)
    class_names = {int(k): v for k, v in sport_class_map[sport_type].items()}

    model = ssd300_vgg16(weights=None, num_classes=len(class_names))
    checkpoint = torch.load(path, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    results_data = []
    global_start_time = time.time()

    for video in media_path.glob("*.avi"):
        if device == "cuda":
            torch.cuda.empty_cache()

        cap = cv2.VideoCapture(video)
        frame_count = 0
        files_counter += 1
        pure_frame_times = []
        full_frame_times = []
        start_time = time.time()

        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                full_start_time = time.time()

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = to_tensor(rgb_frame).to(device)

                pure_start_time = time.time()
                with torch.no_grad():
                    predictions = model([image])

                if device == "cuda":
                    torch.cuda.synchronize()

                pure_time = time.time() - pure_start_time
                pure_frame_times.append(pure_time)

                frame_count += 1

                current_video_time = time.time() - start_time
                total_processing_time = time.time() - global_start_time

                ### PROGRESS VISUALIZER UPDATE ###
                current_fps = 1 / pure_time if pure_time > 0 else 0
                gui.update_progress("FPS", f"{current_fps:.1f}")
                gui.update_progress(
                    "Media Detection Time", f"{current_video_time:.1f} s"
                )
                gui.update_progress("Total Time", f"{total_processing_time:.1f} s")
                gui.update_progress(
                    "Processing Media", f"{files_counter}/{total_test_files}"
                )
                ###

                boxes = predictions[0]["boxes"].cpu()
                scores = predictions[0]["scores"].cpu()
                labels = predictions[0]["labels"].cpu()

                confidence_threshold = 0.5
                mask = scores > confidence_threshold
                boxes = boxes[mask].numpy()
                scores = scores[mask].numpy()
                labels = labels[mask].numpy()

                ### VISUALIZATION SECTION ###
                if enable_visualization:
                    visualize_effdet_ssd(
                        show_boxes,
                        show_scores,
                        show_labels,
                        class_names,
                        boxes,
                        scores,
                        labels,
                        frame,
                    )
                    cv2.imshow("ssd", frame)
                    cv2.waitKey(1)
                ###

                full_time = time.time() - full_start_time
                full_frame_times.append(full_time)

            else:
                break

        cap.release()
        cv2.destroyAllWindows()

        if device == "cuda":
            del predictions
            torch.cuda.empty_cache()

        total_time = time.time() - start_time

        pure_metrics = calculate_fps_and_time(
            pure_frame_times, sum(pure_frame_times), frame_count
        )
        full_metrics = calculate_fps_and_time(
            full_frame_times, sum(full_frame_times), frame_count
        )

        results_data.append(
            {
                "video_name": video.name,
                "pure_min_fps": pure_metrics["min_fps"],
                "pure_avg_fps": pure_metrics["avg_fps"],
                "pure_max_fps": pure_metrics["max_fps"],
                "pure_avg_frame_time": pure_metrics["avg_frame_time"],
                "gui_min_fps": full_metrics["min_fps"],
                "gui_avg_fps": full_metrics["avg_fps"],
                "gui_max_fps": full_metrics["max_fps"],
                "gui_avg_frame_time": full_metrics["avg_frame_time"],
                "total_detection_time": total_time,
                "device": device,
                "frames_processed": frame_count,
            }
        )

    if gui.auto_save_checkbox.get():
        save_performence_metrics(results_data, "ssd", sport_type, "videos", "custom")
    gui._close_detection_screen()
    print(results_data)


def run_ssd_custom_images(media_path, device, sport_type, gui):

    ### VISUALIZATION CHECKBOXES ###
    enable_visualization = gui.enable_visualization_var.get()
    show_boxes = gui.show_bounding_boxes_checkbox.get()
    show_scores = gui.show_confidence_scores_checkbox.get()
    show_labels = gui.show_labels_checkbox.get()
    total_test_files = int(gui.total_files.get()) - 1
    ###

    path = get_correct_custom_model(sport_type, "ssd")

    with open(path.parents[1] / "class_names.json", "r") as f:
        sport_class_map = json.load(f)
    class_names = {int(k): v for k, v in sport_class_map[sport_type].items()}

    model = ssd300_vgg16(weights=None, num_classes=len(class_names))
    checkpoint = torch.load(path, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    if device == "cuda":
        torch.cuda.empty_cache()

    coco = COCO(str(media_path / "_annotations.coco.json"))
    id_to_filename = {img["id"]: img["file_name"] for img in coco.dataset["images"]}
    filename_to_id = {v: k for k, v in id_to_filename.items()}
    metric = MeanAveragePrecision(
        iou_type="bbox", class_metrics=True, extended_summary=True
    )

    results_data = []
    pure_frame_times = []
    full_frame_times = []
    processed_count = 0
    start_time = time.time()

    for image_path in media_path.glob("*.jpg"):
        img_name = image_path.name
        image = cv2.imread(image_path)

        full_start_time = time.time()

        rgb_frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_tensor = to_tensor(rgb_frame).to(device)

        pure_start_time = time.time()
        with torch.no_grad():
            predictions = model([image_tensor])

        if device == "cuda":
            torch.cuda.synchronize()

        pure_time = time.time() - pure_start_time
        pure_frame_times.append(pure_time)

        processed_count += 1
        total_processing_time = time.time() - start_time

        ### PROGRESS VISUALIZER UPDATE ###
        gui.update_progress("Total Time", f"{total_processing_time:.1f} s")
        gui.update_progress("Processing Media", f"{processed_count}/{total_test_files}")
        ###

        boxes = predictions[0]["boxes"].cpu()
        scores = predictions[0]["scores"].cpu()
        labels = predictions[0]["labels"].cpu()

        confidence_threshold = 0.5
        mask = scores > confidence_threshold
        boxes = boxes[mask]
        scores = scores[mask]
        labels = labels[mask]

        preds = [{"boxes": boxes, "scores": scores, "labels": labels}]

        img_id = filename_to_id.get(img_name, None)
        ann_ids = coco.getAnnIds(imgIds=[img_id])
        anns = coco.loadAnns(ann_ids)

        target_boxes = []
        target_labels = []
        for ann in anns:
            bbox = ann["bbox"]
            x1, y1, w, h = bbox
            target_boxes.append([x1, y1, x1 + w, y1 + h])
            target_labels.append(ann["category_id"])

        if target_boxes:
            targets = [
                {
                    "boxes": torch.tensor(target_boxes, dtype=torch.float32),
                    "labels": torch.tensor(target_labels, dtype=torch.int64),
                }
            ]
            metric.update(preds, targets)

        ### VISUALIZATION SECTION ###
        if enable_visualization:
            visualize_effdet_ssd(
                show_boxes,
                show_scores,
                show_labels,
                class_names,
                boxes,
                scores,
                labels,
                image,
            )
            cv2.imshow("ssd", image)
            cv2.waitKey(1)
        ###

        full_time = time.time() - full_start_time
        full_frame_times.append(full_time)

    cv2.destroyAllWindows()

    if device == "cuda":
        del predictions
        torch.cuda.empty_cache()

    total_time = time.time() - start_time

    pure_metrics = calculate_fps_and_time(
        pure_frame_times, sum(pure_frame_times), processed_count
    )
    full_metrics = calculate_fps_and_time(
        full_frame_times, sum(full_frame_times), processed_count
    )

    results_data.append(
        {
            "folder_path": str(media_path),
            "gui_avg_image_time": full_metrics["avg_frame_time"],
            "pure_avg_image_time": pure_metrics["avg_frame_time"],
            "total_detection_time": total_time,
            "images_processed": processed_count,
            "device": device,
        }
    )
    if gui.auto_save_checkbox.get():
        calculate_torch_results(metric, class_names, "ssd", sport_type, True)
        save_performence_metrics(results_data, "ssd", sport_type, "images", "custom")
    else:
        calculate_torch_results(metric, class_names, "ssd", sport_type, False)
    gui._close_detection_screen()
    print(results_data)

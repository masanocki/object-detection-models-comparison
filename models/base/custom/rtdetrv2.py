import time

import cv2
import torch
from PIL import Image
from transformers import AutoModelForObjectDetection, AutoProcessor
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from pycocotools.coco import COCO

from models.utils.metrics import *

from models.utils.helpers import get_correct_custom_model, visualize_rtdetr


def run_rtdetrv2_custom_videos(media_path, device, sport_type, gui):

    ### VISUALIZATION CHECKBOXES ###
    enable_visualization = gui.enable_visualization_var.get()
    show_boxes = gui.show_bounding_boxes_checkbox.get()
    show_scores = gui.show_confidence_scores_checkbox.get()
    show_labels = gui.show_labels_checkbox.get()
    total_test_files = gui.total_files.get()
    files_counter = 0
    ###

    path = get_correct_custom_model(sport_type, "rtdetrv2")

    processor = AutoProcessor.from_pretrained("PekingU/rtdetr_v2_r18vd", use_fast=True)
    model = AutoModelForObjectDetection.from_pretrained(path).to(device)
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

                image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                inputs = processor(images=image, return_tensors="pt").to(device)

                pure_start_time = time.time()
                with torch.no_grad():
                    outputs = model(**inputs)

                if device == "cuda":
                    torch.cuda.synchronize()

                pure_time = time.time() - pure_start_time
                pure_frame_times.append(pure_time)

                target_sizes = torch.tensor([image.size[::-1]])
                results = processor.post_process_object_detection(
                    outputs, target_sizes=target_sizes
                )[0]

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

                ### VISUALIZATION SECTION ###
                if enable_visualization:
                    visualize_rtdetr(
                        show_boxes,
                        show_scores,
                        show_labels,
                        model,
                        results,
                        frame,
                        0.5,
                    )
                    cv2.imshow("rtdetrv2", frame)
                    cv2.waitKey(1)
                ###

                full_time = time.time() - full_start_time
                full_frame_times.append(full_time)

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
        save_performence_metrics(
            results_data, "rtdetrv2", sport_type, "videos", "custom"
        )
    gui._close_detection_screen()
    print(results_data)


def run_rtdetrv2_custom_images(media_path, device, sport_type, gui):

    ### VISUALIZATION CHECKBOXES ###
    enable_visualization = gui.enable_visualization_var.get()
    show_boxes = gui.show_bounding_boxes_checkbox.get()
    show_scores = gui.show_confidence_scores_checkbox.get()
    show_labels = gui.show_labels_checkbox.get()
    total_test_files = int(gui.total_files.get()) - 1
    ###

    path = get_correct_custom_model(sport_type, "rtdetrv2")

    processor = AutoProcessor.from_pretrained("PekingU/rtdetr_v2_r18vd", use_fast=True)
    model = AutoModelForObjectDetection.from_pretrained(path).to(device)
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
        image = cv2.imread(image_path)

        full_start_time = time.time()

        image_processed = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        inputs = processor(images=image_processed, return_tensors="pt").to(device)

        pure_start_time = time.time()
        with torch.no_grad():
            outputs = model(**inputs)

        if device == "cuda":
            torch.cuda.synchronize()

        pure_time = time.time() - pure_start_time
        pure_frame_times.append(pure_time)

        target_sizes = torch.tensor([image_processed.size[::-1]])
        results = processor.post_process_object_detection(
            outputs, target_sizes=target_sizes
        )[0]

        ### FOR METRICS CALCULATION ###
        preds = [
            {
                "boxes": results["boxes"].cpu(),
                "scores": results["scores"].cpu(),
                "labels": results["labels"].cpu(),
            }
        ]

        img_id = filename_to_id.get(image_path.name, None)
        ann_ids = coco.getAnnIds(imgIds=[img_id])
        anns = coco.loadAnns(ann_ids)

        target_boxes = []
        target_labels = []
        for ann in anns:
            x1, y1, w, h = ann["bbox"]
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
        ###

        processed_count += 1
        total_processing_time = time.time() - start_time

        ### PROGRESS VISUALIZER UPDATE ###
        gui.update_progress("Total Time", f"{total_processing_time:.1f} s")
        gui.update_progress("Processing Media", f"{processed_count}/{total_test_files}")
        ###

        ### VISUALIZATION SECTION ###
        if enable_visualization:
            visualize_rtdetr(
                show_boxes,
                show_scores,
                show_labels,
                model,
                results,
                image,
                0.5,
            )
            cv2.imshow("rtdetrv2", image)
            cv2.waitKey(1)
        ###

        full_time = time.time() - full_start_time
        full_frame_times.append(full_time)

    cv2.destroyAllWindows()

    if device == "cuda":
        del outputs
        del inputs
        del results
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
        calculate_rtdetr_results(metric, model, "rtdetrv2", sport_type, True)
        save_performence_metrics(
            results_data, "rtdetrv2", sport_type, "images", "custom"
        )
    else:
        calculate_rtdetr_results(metric, model, "rtdetrv2", sport_type, False)
    gui._close_detection_screen()
    print(results_data)

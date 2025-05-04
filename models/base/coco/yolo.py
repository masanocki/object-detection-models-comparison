import time
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO

from models.utils.metrics import *


def run_yolo_coco_videos(model_name, media_path, device, gui):

    ### VISUALIZATION CHECKBOXES ###
    enable_visualization = gui.enable_visualization_var.get()
    show_boxes = gui.show_bounding_boxes_checkbox.get()
    show_scores = gui.show_confidence_scores_checkbox.get()
    show_labels = gui.show_labels_checkbox.get()
    total_test_files = gui.total_files.get()
    files_counter = 0
    ###

    if model_name == "yolov11":
        path = Path("./saved_models/yolo11n.pt")
    elif model_name == "yolov12":
        path = Path("./saved_models/yolo12n.pt")

    model = YOLO(path).to(device)

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
                pure_start_time = time.time()
                results = model.track(frame, persist=True, verbose=False)

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

                ### VISUALIZATION SECTION ###
                if enable_visualization:
                    annotated_frame = results[0].plot(
                        boxes=show_boxes, labels=show_labels, conf=show_scores
                    )
                    cv2.imshow(model_name, annotated_frame)
                    cv2.waitKey(1)
                ###

                full_time = time.time() - pure_start_time
                full_frame_times.append(full_time)

            else:
                break

        cap.release()
        cv2.destroyAllWindows()

        if device == "cuda":
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

    gui._close_detection_screen()
    print(results_data)


def run_yolo_coco_images(model_name, media_path, device, gui):

    ### VISUALIZATION CHECKBOXES ###
    enable_visualization = gui.enable_visualization_var.get()
    show_boxes = gui.show_bounding_boxes_checkbox.get()
    show_scores = gui.show_confidence_scores_checkbox.get()
    show_labels = gui.show_labels_checkbox.get()
    total_test_files = int(gui.total_files.get()) - 1
    ###

    if model_name == "yolov11":
        path = Path("./saved_models/yolo11s.pt")
    elif model_name == "yolov12":
        path = Path("./saved_models/yolo12s.pt")

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
        total_processing_time = time.time() - start_time
        processed_count += 1

        ### PROGRESS VISUALIZER UPDATE ###
        gui.update_progress("Total Time", f"{total_processing_time:.1f} s")
        gui.update_progress("Processing Media", f"{processed_count}/{total_test_files}")
        ###

        if enable_visualization:
            annotated_frame = results[0].plot(
                boxes=show_boxes, labels=show_labels, conf=show_scores
            )
            cv2.imshow(model_name, annotated_frame)
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
    gui._close_detection_screen()
    print(results_data)

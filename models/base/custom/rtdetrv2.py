import time

import cv2
import torch
from PIL import Image
from transformers import AutoModelForObjectDetection, AutoProcessor

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

                current_video_time = time.time() - start_time
                total_processing_time = time.time() - global_start_time

                ### PROGRESS VISUALIZER UPDATE ###
                current_fps = 1 / frame_time if frame_time > 0 else 0
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
        total_processing_time = time.time() - start_time
        processed_count += 1

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
    gui._close_detection_screen()
    print(results_data)

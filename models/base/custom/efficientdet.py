import json
import time

import cv2
import torch
from effdet import DetBenchPredict, create_model
from torchvision.ops import nms
from torchvision.transforms.functional import to_tensor
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from pycocotools.coco import COCO

from models.utils.metrics import *

from models.utils.helpers import get_correct_custom_model, visualize_effdet_ssd


def run_efficientdet_custom_videos(media_path, device, sport_type, gui):

    ### VISUALIZATION CHECKBOXES ###
    enable_visualization = gui.enable_visualization_var.get()
    show_boxes = gui.show_bounding_boxes_checkbox.get()
    show_scores = gui.show_confidence_scores_checkbox.get()
    show_labels = gui.show_labels_checkbox.get()
    total_test_files = gui.total_files.get()
    files_counter = 0
    ###

    path = get_correct_custom_model(sport_type, "efficientdet")

    with open(path.parents[1] / "class_names.json", "r") as f:
        sport_class_map = json.load(f)
    class_names = {int(k): v for k, v in sport_class_map[sport_type].items()}

    model = create_model(
        "tf_efficientdet_d0", pretrained=False, num_classes=len(class_names) - 1
    )
    if device != "cuda":
        checkpoint = torch.load(
            path, map_location=torch.device("cpu"), weights_only=False
        )
    else:
        checkpoint = torch.load(path, weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    model = DetBenchPredict(model).to(device)
    model.eval()

    torch.backends.cudnn.benchmark = True

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

                original_height, original_width = frame.shape[:2]
                resized_frame = cv2.resize(frame, (512, 512))
                rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
                input_tensor = (
                    to_tensor(rgb_frame).unsqueeze(0).to(device, non_blocking=True)
                )

                pure_start_time = time.time()
                with torch.no_grad():
                    detections = model(input_tensor)
                    detections = detections[0].cpu()

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

                boxes = detections[:, :4].cpu()
                scores = detections[:, 4].cpu()
                labels = detections[:, 5].cpu()

                confidence_threshold = 0.25
                keep = scores > confidence_threshold
                boxes = boxes[keep]
                scores = scores[keep]
                labels = labels[keep]

                keep = nms(boxes, scores, iou_threshold=0.5)
                boxes = boxes[keep]
                scores = scores[keep]
                labels = labels[keep]

                scale_x = original_width / 512
                scale_y = original_height / 512
                boxes[:, [0, 2]] *= scale_x
                boxes[:, [1, 3]] *= scale_y

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
                    cv2.imshow("efficientdet", frame)
                    cv2.waitKey(1)
                ###

                full_time = time.time() - full_start_time
                full_frame_times.append(full_time)

            else:
                break
        cap.release()
        cv2.destroyAllWindows()

        if device == "cuda":
            del detections
            del input_tensor
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
            results_data, "efficientdet", sport_type, "videos", "custom"
        )
    gui._close_detection_screen()
    print(results_data)


def run_efficientdet_custom_images(media_path, device, sport_type, gui):

    ### VISUALIZATION CHECKBOXES ###
    enable_visualization = gui.enable_visualization_var.get()
    show_boxes = gui.show_bounding_boxes_checkbox.get()
    show_scores = gui.show_confidence_scores_checkbox.get()
    show_labels = gui.show_labels_checkbox.get()
    total_test_files = int(gui.total_files.get()) - 1
    ###

    path = get_correct_custom_model(sport_type, "efficientdet")

    with open(path.parents[1] / "class_names.json", "r") as f:
        sport_class_map = json.load(f)
    class_names = {int(k): v for k, v in sport_class_map[sport_type].items()}

    model = create_model(
        "tf_efficientdet_d0", pretrained=False, num_classes=len(class_names) - 1
    )
    if device != "cuda":
        checkpoint = torch.load(
            path, map_location=torch.device("cpu"), weights_only=False
        )
    else:
        checkpoint = torch.load(path, weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    model = DetBenchPredict(model).to(device)
    model.eval()

    if device == "cuda":
        torch.cuda.empty_cache()

    ### FOR METRICS CALCULATION ###
    coco = COCO(str(media_path / "_annotations.coco.json"))
    id_to_filename = {img["id"]: img["file_name"] for img in coco.dataset["images"]}
    filename_to_id = {v: k for k, v in id_to_filename.items()}
    metric = MeanAveragePrecision(
        iou_type="bbox", class_metrics=True, extended_summary=True
    )
    ###

    results_data = []
    pure_frame_times = []
    full_frame_times = []
    processed_count = 0
    start_time = time.time()

    for image_path in media_path.glob("*.jpg"):
        img_name = image_path.name
        image = cv2.imread(image_path)

        full_start_time = time.time()

        original_height, original_width = image.shape[:2]
        resized_frame = cv2.resize(image, (512, 512))
        rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
        input_tensor = to_tensor(rgb_frame).unsqueeze(0).to(device, non_blocking=True)

        pure_start_time = time.time()
        with torch.no_grad():
            detections = model(input_tensor)
            detections = detections[0].cpu()

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

        boxes = detections[:, :4]
        scores = detections[:, 4]
        labels = detections[:, 5].int()

        # confidence_threshold = 0.1
        # keep = scores > confidence_threshold
        # boxes = boxes[keep]
        # scores = scores[keep]
        # labels = labels[keep]

        # keep = nms(boxes, scores, iou_threshold=0.1)
        # boxes = boxes[keep]
        # scores = scores[keep]
        # labels = labels[keep]

        scale_x = original_width / 512
        scale_y = original_height / 512
        boxes[:, [0, 2]] *= scale_x
        boxes[:, [1, 3]] *= scale_y

        ### METRICS CALCULATION ###
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
        ###

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
            cv2.imshow("efficientdet", image)
            cv2.waitKey(1)
        ###

        full_time = time.time() - full_start_time
        full_frame_times.append(full_time)

    cv2.destroyAllWindows()
    if device == "cuda":
        del detections
        del input_tensor
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
        calculate_torch_results(metric, class_names, "efficientdet", sport_type, True)
        save_performence_metrics(
            results_data, "efficientdet", sport_type, "images", "custom"
        )
    else:
        calculate_torch_results(metric, class_names, "efficientdet", sport_type, False)
    gui._close_detection_screen()
    print(results_data)

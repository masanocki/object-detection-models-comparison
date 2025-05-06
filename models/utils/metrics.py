from pathlib import Path
import csv
import json
import torch
import pandas as pd


def calculate_fps_and_time(frame_times, total_time, frame_count):
    if not frame_times or frame_count == 0:
        return {"min_fps": 0, "avg_fps": 0, "max_fps": 0, "avg_frame_time": 0}

    inference_fps = [1 / t for t in frame_times if t > 0]

    return {
        "min_fps": min(inference_fps),
        "avg_fps": sum(inference_fps) / len(inference_fps),
        "max_fps": max(inference_fps),
        "avg_frame_time": total_time / len(frame_times),
    }


def calculate_yolo_results(model, model_name, sport_type, save):
    save_path = Path(f"results/{model_name}/{sport_type}/eval_metrics.csv")
    save_path.parent.mkdir(parents=True, exist_ok=True)

    path = (
        Path(__file__).resolve().parents[2]
        / "dataset/yolo_format"
        / sport_type
        / "data.yaml"
    )

    results = model.val(data=path)

    if save:
        box = results.box
        names = results.names

        rows = []

        rows.append(
            {
                "Class": "all",
                "P": round(box.mp, 3),
                "R": round(box.mr, 3),
                "F1": round(box.f1.mean(), 3),
                "mAP@0.50": round(box.map50, 3),
                "mAP@0.75": round(box.map75, 3),
                "mAP@0.50:0.95": round(box.map, 3),
            }
        )

        for cls_id, cls_name in names.items():
            p, r, map50, map95 = box.class_result(cls_id)
            f1 = box.f1[cls_id]

            rows.append(
                {
                    "Class": cls_name,
                    "P": round(p, 3),
                    "R": round(r, 3),
                    "F1": round(f1, 3),
                    "mAP@0.50": round(map50, 3),
                    "mAP@0.75": "-",
                    "mAP@0.50:0.95": round(map95, 3),
                }
            )

        with open(save_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)


def calculate_torch_results(metric, class_names, model_name, sport_type, save):
    save_path = Path(f"results/{model_name}/{sport_type}/eval_metrics.csv")
    save_path.parent.mkdir(parents=True, exist_ok=True)

    results = metric.compute()

    for k in ["precision", "recall", "map", "map_50", "map_75", "map_per_class"]:
        if k in results:
            results[k] = torch.nan_to_num(results[k], nan=0.0, posinf=0.0, neginf=0.0)
            if "precision" in k or "recall" in k:
                results[k] = results[k].clamp(min=0.0, max=1.0)

    all_map = results["map"].item()
    all_map50 = results["map_50"].item()
    all_map75 = results["map_75"].item()
    all_precision = results["precision"].mean().item()
    all_recall = results["recall"].mean().item()
    all_f1 = (
        2 * all_precision * all_recall / (all_precision + all_recall)
        if (all_precision + all_recall) > 0
        else 0.0
    )

    rows = [
        {
            "Class": "all",
            "P": round(all_precision, 3),
            "R": round(all_recall, 3),
            "F1": round(all_f1, 3),
            "mAP@0.50": round(all_map50, 3),
            "mAP@0.75": round(all_map75, 3),
            "mAP@0.50:0.95": round(all_map, 3),
        }
    ]

    precision_matrix = results["precision"].mean(dim=(0, 1, 3, 4))
    recall_matrix = results["recall"].mean(dim=(0, 2, 3))
    map_per_class = results["map_per_class"]
    class_ids = results["classes"]

    per_class_f1 = (
        2 * precision_matrix * recall_matrix / (precision_matrix + recall_matrix + 1e-6)
    )

    for idx, class_id in enumerate(class_ids):
        class_name = class_names.get(int(class_id), f"Class_{class_id}")
        p = precision_matrix[idx].item()
        r = recall_matrix[idx].item()
        f1 = per_class_f1[idx].item()
        m = map_per_class[idx].item()

        row = {
            "Class": class_name,
            "P": round(p, 3),
            "R": round(r, 3),
            "F1": round(f1, 3),
            "mAP@0.50": "-",
            "mAP@0.75": "-",
            "mAP@0.50:0.95": round(m, 3),
        }
        rows.append(row)

    with open(save_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def calculate_rtdetr_results(metric, model, model_name, sport_type, save):
    save_path = Path(f"results/{model_name}/{sport_type}/eval_metrics.csv")
    save_path.parent.mkdir(parents=True, exist_ok=True)

    results = metric.compute()

    for k in ["precision", "recall", "map", "map_50", "map_75", "map_per_class"]:
        if k in results:
            results[k] = torch.nan_to_num(results[k], nan=0.0, posinf=0.0, neginf=0.0)
            if "precision" in k or "recall" in k:
                results[k] = results[k].clamp(min=0.0, max=1.0)

    all_map = results["map"].item()
    all_map50 = results["map_50"].item()
    all_map75 = results["map_75"].item()
    all_precision = results["precision"].mean().item()
    all_recall = results["recall"].mean().item()
    all_f1 = (
        2 * all_precision * all_recall / (all_precision + all_recall)
        if (all_precision + all_recall) > 0
        else 0.0
    )

    rows = [
        {
            "Class": "all",
            "P": round(all_precision, 3),
            "R": round(all_recall, 3),
            "F1": round(all_f1, 3),
            "mAP@0.50": round(all_map50, 3),
            "mAP@0.75": round(all_map75, 3),
            "mAP@0.50:0.95": round(all_map, 3),
        }
    ]

    precision_matrix = results["precision"].mean(dim=(0, 1, 3, 4))
    recall_matrix = results["recall"].mean(dim=(0, 2, 3))
    map_per_class = results["map_per_class"]
    class_ids = results["classes"]

    per_class_f1 = (
        2 * precision_matrix * recall_matrix / (precision_matrix + recall_matrix + 1e-6)
    )

    for idx, class_id in enumerate(class_ids):
        p = precision_matrix[idx].item()
        r = recall_matrix[idx].item()
        f1 = per_class_f1[idx].item()
        m = map_per_class[idx].item()

        row = {
            "Class": model.config.id2label[int(class_id)],
            "P": round(p, 3),
            "R": round(r, 3),
            "F1": round(f1, 3),
            "mAP@0.50": "-",
            "mAP@0.75": "-",
            "mAP@0.50:0.95": round(m, 3),
        }
        rows.append(row)

    with open(save_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def save_performence_metrics(data, model_name, sport_type, media_type, type):
    if type == "coco":
        save_path = Path(
            f"results/{model_name}/coco/{sport_type}/{media_type}_performence_metrics.json"
        )
    elif type == "custom":
        save_path = Path(
            f"results/{model_name}/custom/{sport_type}/{media_type}_performence_metrics.json"
        )
    save_path.parent.mkdir(parents=True, exist_ok=True)

    with open(save_path, "w") as f:
        json.dump(data, f, indent=4)

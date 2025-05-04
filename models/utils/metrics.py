from pathlib import Path
import csv


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
    save_path = Path(f"results/{model_name}_{sport_type}_results.csv")
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

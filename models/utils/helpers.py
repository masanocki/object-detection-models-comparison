from pathlib import Path
import cv2


def get_correct_custom_model(sport_name, model_name):
    if model_name == "yolov11" or model_name == "yolov12":
        return (
            Path(__file__).resolve().parents[2]
            / "custom_models"
            / model_name
            / f"{model_name}_{sport_name}.pt"
        )
    elif model_name == "efficientdet":
        return (
            Path(__file__).resolve().parents[2]
            / "custom_models"
            / model_name
            / f"{model_name}_{sport_name}.pth.tar"
        )
    elif model_name == "rtdetrv2":
        return (
            Path(__file__).resolve().parents[2]
            / "custom_models"
            / model_name
            / f"{model_name}_{sport_name}"
        )
    elif model_name == "ssd":
        return (
            Path(__file__).resolve().parents[2]
            / "custom_models"
            / model_name
            / f"{model_name}_{sport_name}.pth"
        )


def visualize_effdet_ssd(bbox, conf, lab, class_names, boxes, scores, labels, frame):
    for box, score, label in zip(boxes, scores, labels):
        x1, y1, x2, y2 = map(int, box.tolist())
        class_name = class_names.get(int(label.item()), "Unknown")

        if bbox:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        label_text = ""
        if lab:
            label_text = f"{class_name}"
        if conf:
            if label_text:
                label_text += f": {score.item():.2f}"
            else:
                label_text = f"{score.item():.2f}"

        if label_text:
            cv2.putText(
                frame,
                label_text,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )


def visualize_rtdetr(bbox, conf, lab, model, results, frame, threshhold):
    for score, label, box in zip(
        results["scores"], results["labels"], results["boxes"]
    ):
        if score > threshhold:
            box = [int(i) for i in box.tolist()]

            if bbox:
                cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)

            label_text = ""
            if lab:
                label_text = f"{model.config.id2label[label.item()]}"
            if conf:
                if label_text:
                    label_text += f": {score:.2f}"
                else:
                    label_text = f"{score:.2f}"

            if label_text:
                cv2.putText(
                    frame,
                    label_text,
                    (box[0], box[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2,
                )

from pathlib import Path


def get_correct_custom_model(sport_name, model_name):
    if model_name == "yolo11" or model_name == "yolo12":
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

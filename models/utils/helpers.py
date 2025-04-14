from pathlib import Path


def get_correct_custom_model(sport_name, model_name):
    return (
        Path(__file__).resolve().parents[2]
        / "custom_models"
        / model_name
        / f"{model_name}_{sport_name}.pt"
    )

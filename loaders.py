import json
from pathlib import Path
import pandas as pd


def set_model_details(
    selected_model, model_name_var, model_version_var, model_description_var
):
    with open("models_details.json", "r") as file:
        models_details = json.load(file)

    if selected_model in models_details:
        model_details = models_details[selected_model]
        model_name_var.set(model_details["model_name"])
        model_version_var.set(model_details["model_version"])
        model_description_var.set(model_details["model_description"])
    else:
        model_name_var.set("None")
        model_version_var.set("None")
        model_description_var.set("None")


def load_media_files(
    dir_path, loaded_sport, media_type, total_files, total_size, files_table
):
    media_files = []
    total_size_temp = 0
    path = Path(dir_path)
    splitted_path = path.parts
    splitted_path_length = len(splitted_path)

    for file in path.iterdir():
        media_files.append(file.name)
        total_size_temp += file.stat().st_size

    if media_files[1].endswith(".avi"):
        media_type.set("videos")
    elif media_files[1].endswith(".jpg"):
        media_type.set("images")
    total_files.set(len(media_files))
    total_size.set(f"{total_size_temp} bytes")
    if splitted_path[splitted_path_length - 2] == "images":
        loaded_sport.set(splitted_path[splitted_path_length - 3])
    else:
        loaded_sport.set(splitted_path[splitted_path_length - 2])

    # files_table.configure(row=len(media_files))
    files_table.update_values([media_files])
    # print(media_files)
    # print([media_files])
    # files_table = ["elo"]

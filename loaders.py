import json
from pathlib import Path
import pandas as pd
import customtkinter as ctk
from PIL import Image
import cv2

# FOR PREVIEW VIDEO PLAYER
current_video_capture = None
current_after_id = None


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
    row_number = 1

    files_table.update_values([])
    files_table.edit_row(0, value="File Name", font=ctk.CTkFont(size=17, weight="bold"))

    for file in path.iterdir():
        media_files.append(file.name)
        if row_number <= 20 and file.suffix in [".jpg", ".jpeg", ".avi"]:
            files_table.edit_row(row_number, f"     {file.name}")
            row_number += 1
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

    files_table.edit_row(
        21, value=f"First {row_number - 1} files", anchor="c", hover=False
    )


def load_file_details(
    cell_file_name,
    dir_path,
    file_name,
    file_type,
    file_size,
    width,
    height,
    preview_label,
):
    global current_video_capture, current_after_id

    if cell_file_name == " ":
        return

    stripped_name = cell_file_name.strip()
    path = Path(dir_path) / stripped_name
    if "_jpg" in stripped_name:
        file_name.set(stripped_name.split("_jpg", 1)[0])
    else:
        file_name.set(stripped_name)
    file_size.set(f"{path.stat().st_size} bytes")
    file_type.set(path.suffix[1:].upper())

    stop_and_realease_video(preview_label)
    if path.suffix.lower() == ".avi":
        play_video_in_preview_box(path, preview_label, width, height)
    else:
        loaded_image = ctk.CTkImage(Image.open(path), size=(width - 9, height - 9))
        preview_label.configure(text="", image=loaded_image)


def play_video_in_preview_box(video_path, preview_label, width, height):
    global current_video_capture, current_after_id
    stop_and_realease_video(preview_label)

    cap = cv2.VideoCapture(video_path)
    current_video_capture = cap

    def update_frame():
        global current_after_id

        if current_video_capture is None:
            return

        ret, frame = cap.read()

        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame)
            video_frame = ctk.CTkImage(img, size=(width - 9, height - 9))
            preview_label.configure(text="", image=video_frame)
            current_after_id = preview_label.after(30, update_frame)
        else:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            current_after_id = preview_label.after(30, update_frame)

    update_frame()


def stop_and_realease_video(preview_label):
    global current_video_capture, current_after_id

    if current_video_capture is not None:
        current_video_capture.release()
        current_video_capture = None

    if current_after_id is not None:
        preview_label.after_cancel(current_after_id)
        current_after_id = None

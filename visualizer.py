import customtkinter as ctk


def visualize_detection(frame, det_box_label):
    video_frame = ctk.CTkImage(frame, size=(50, 50))
    det_box_label.configure(text="", image=video_frame)

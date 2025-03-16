import random
import sys
import time

import tkinter as tk
from io import StringIO
from threading import Thread

from tkinter import ttk

import customtkinter as ctk
from customtkinter import filedialog
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from PIL import Image

ctk.set_appearance_mode("system")
ctk.set_default_color_theme("themes/rime.json")


class GUI(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("ModelTester")
        self.geometry("1280x720")
        self.load_model_and_files_screen()

    def load_model_and_files_screen(self):

        self.model_load_box = ctk.CTkFrame(
            self, corner_radius=20, border_width=1, border_color="black"
        )
        self.model_load_box.place(relx=0.01, rely=0.01, relwidth=0.5, relheight=0.5)

        self.model_path = ctk.StringVar()
        self.model_path.set("")

        self.model_label = ctk.CTkLabel(
            self.model_load_box,
            text="Browse your computer to load a model file:",
            font=ctk.CTkFont(size=18, weight="bold"),
        )
        self.model_label.pack(pady=(30, 5), padx=30, anchor="w")

        self.model_entry = ctk.CTkEntry(
            self.model_load_box,
            textvariable=self.model_path,
            height=40,
            corner_radius=15,
        )
        self.model_entry.pack(pady=10, padx=30, fill="x")

        self.model_details_label = ctk.CTkLabel(
            self.model_load_box,
            text="Model Details",
            font=ctk.CTkFont(size=24, weight="bold"),
        )
        self.model_details_label.pack(pady=(20, 5), padx=30, anchor="w")

        self.model_details_box = ctk.CTkFrame(self.model_load_box, corner_radius=20)
        self.model_details_box.pack(pady=(10, 0), padx=30, fill="x")

        self.model_name_label = ctk.CTkLabel(
            self.model_details_box,
            text="Model Name:",
            font=ctk.CTkFont(size=15, weight="bold"),
        )
        self.model_name_label.grid(
            row=0, column=0, padx=(20, 10), pady=(10, 0), sticky="w"
        )
        self.model_name_value = ctk.CTkLabel(
            self.model_details_box, text="None", font=ctk.CTkFont(size=15)
        )
        self.model_name_value.grid(row=0, column=1, pady=(10, 0), sticky="w")

        self.model_version_label = ctk.CTkLabel(
            self.model_details_box,
            text="Version:",
            font=ctk.CTkFont(size=15, weight="bold"),
        )
        self.model_version_label.grid(row=1, column=0, padx=(20, 10), sticky="w")
        self.model_version_value = ctk.CTkLabel(
            self.model_details_box, text="None", font=ctk.CTkFont(size=15)
        )
        self.model_version_value.grid(row=1, column=1, sticky="w")

        self.model_description_label = ctk.CTkLabel(
            self.model_details_box,
            text="Description:",
            font=ctk.CTkFont(size=15, weight="bold"),
        )
        self.model_description_label.grid(
            row=2, column=0, sticky="w", padx=(20, 10), pady=(0, 10)
        )
        self.model_description_value = ctk.CTkLabel(
            self.model_details_box, text="None", font=ctk.CTkFont(size=15)
        )
        self.model_description_value.grid(row=2, column=1, sticky="w", pady=(0, 10))

        self.browse_button = ctk.CTkButton(
            self.model_load_box,
            text="Browse",
            corner_radius=15,
            height=40,
            font=ctk.CTkFont(size=15, weight="bold"),
        )
        self.browse_button.pack(pady=(0, 30), side="right", padx=(0, 30), anchor="s")

        self.load_button = ctk.CTkButton(
            self.model_load_box,
            text="Load Model",
            corner_radius=15,
            height=40,
            font=ctk.CTkFont(size=15, weight="bold"),
        )
        self.load_button.pack(pady=(0, 30), side="right", padx=30, anchor="s")

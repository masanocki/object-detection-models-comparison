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
ctk.set_default_color_theme("dark-blue")


class GUI(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Object Detection Model Tester")
        self.geometry("1280x720")

        self.grid_columnconfigure(2, weight=1)
        self.grid_rowconfigure(1, weight=1)
        self.grid_rowconfigure((0, 1, 2), weight=1)

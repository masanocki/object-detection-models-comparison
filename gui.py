import customtkinter as ctk
from customtkinter import filedialog
from functions import set_model_details
from CTkScrollableDropdown import *

ctk.set_appearance_mode("system")
ctk.set_default_color_theme("themes/rime.json")


class GUI(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("ModelTester")
        self._state_before_windows_set_titlebar_color = "zoomed"

        self.load_model_and_files_screen()

    def load_model_and_files_screen(self):

        # MODEL VARIABLES
        self.model_path = ctk.StringVar()
        self.model_path.set("")
        self.model_name = ctk.StringVar()
        self.model_name.set("None")
        self.model_version = ctk.StringVar()
        self.model_version.set("None")
        self.model_description = ctk.StringVar()
        self.model_description.set("None")

        # MODEL LOAD BOX
        self.model_load_box = ctk.CTkFrame(
            self, corner_radius=20, border_width=1, border_color="black"
        )
        self.model_load_box.place(relx=0.01, rely=0.01, relwidth=0.49, relheight=0.3)

        self.model_label = ctk.CTkLabel(
            self.model_load_box,
            text="Select model for object detection:",
            font=ctk.CTkFont(size=18, weight="bold"),
        )
        self.model_label.pack(pady=(30, 5), padx=30, anchor="w")

        self.model_selection_list = ctk.CTkComboBox(
            self.model_load_box,
            corner_radius=15,
            height=40,
        )
        self.model_selection_list.pack(pady=10, padx=30, fill="x")
        CTkScrollableDropdown(
            self.model_selection_list,
            values=[
                "None",
                "YOLOv11",
                "YOLOv12",
                "SSD",
                "EfficientDet",
                "DETR",
                "RetinaNet",
            ],
            command=lambda selected_model: [
                set_model_details(
                    selected_model,
                    self.model_name,
                    self.model_version,
                    self.model_description,
                ),
                self.model_selection_list.set(selected_model),
                # self.model_selection_list.configure(
                #     font=ctk.CTkFont(size=15), dropdown_font=ctk.CTkFont(size=15)
                # ),
            ],
            autocomplete=True,
            button_color="transparent",
            alpha=0.9,
            # font=ctk.CTkFont(size=15),
        )

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
            self.model_details_box,
            textvariable=self.model_name,
            font=ctk.CTkFont(size=15),
        )
        self.model_name_value.grid(row=0, column=1, pady=(10, 0), sticky="w")

        self.model_version_label = ctk.CTkLabel(
            self.model_details_box,
            text="Version:",
            font=ctk.CTkFont(size=15, weight="bold"),
        )
        self.model_version_label.grid(row=1, column=0, padx=(20, 10), sticky="w")
        self.model_version_value = ctk.CTkLabel(
            self.model_details_box,
            textvariable=self.model_version,
            font=ctk.CTkFont(size=15),
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
            self.model_details_box,
            textvariable=self.model_description,
            font=ctk.CTkFont(size=15),
        )
        self.model_description_value.grid(row=2, column=1, sticky="w", pady=(0, 10))

        # FILES LOAD BOX
        # DIRECTORY VARIABLES
        self.loaded_sport = ctk.StringVar()
        self.loaded_sport.set("None")
        self.media_type = ctk.StringVar()
        self.media_type.set("None")
        self.total_files = ctk.StringVar()
        self.total_files.set("0")
        self.total_size = ctk.StringVar()
        self.total_size.set("0 KB")

        # FILES VARIABLES
        self.file_name = ctk.StringVar()
        self.file_name.set("None")
        self.file_type = ctk.StringVar()
        self.file_type.set("None")
        self.file_size = ctk.StringVar()
        self.file_size.set("0 KB")

        self.files_load_box = ctk.CTkFrame(
            self, corner_radius=20, border_width=1, border_color="black"
        )
        self.files_load_box.place(relx=0.51, rely=0.01, relwidth=0.48, relheight=0.98)

        self.directory_details_label = ctk.CTkLabel(
            self.files_load_box,
            text="Directory Details",
            font=ctk.CTkFont(size=22, weight="bold"),
        )
        self.directory_details_label.place(relx=0.02, rely=0.02)

        self.loaded_sport_label = ctk.CTkLabel(
            self.files_load_box,
            text="Loaded Sport:",
            font=ctk.CTkFont(size=15, weight="bold"),
        )
        self.loaded_sport_label.place(relx=0.02, rely=0.06)

        self.loaded_sport_value = ctk.CTkLabel(
            self.files_load_box,
            textvariable=self.loaded_sport,
            font=ctk.CTkFont(size=15),
        )
        self.loaded_sport_value.place(relx=0.25, rely=0.06)

        self.media_type_label = ctk.CTkLabel(
            self.files_load_box,
            text="Media Type:",
            font=ctk.CTkFont(size=15, weight="bold"),
        )
        self.media_type_label.place(relx=0.02, rely=0.085)

        self.media_type_value = ctk.CTkLabel(
            self.files_load_box,
            textvariable=self.media_type,
            font=ctk.CTkFont(size=15),
        )
        self.media_type_value.place(relx=0.25, rely=0.085)

        self.total_files_label = ctk.CTkLabel(
            self.files_load_box,
            text="Total Files:",
            font=ctk.CTkFont(size=15, weight="bold"),
        )
        self.total_files_label.place(relx=0.02, rely=0.11)

        self.total_files_value = ctk.CTkLabel(
            self.files_load_box,
            textvariable=self.total_files,
            font=ctk.CTkFont(size=15),
        )
        self.total_files_value.place(relx=0.25, rely=0.11)

        self.total_size_label = ctk.CTkLabel(
            self.files_load_box,
            text="Total Size:",
            font=ctk.CTkFont(size=15, weight="bold"),
        )
        self.total_size_label.place(relx=0.02, rely=0.135)

        self.total_size_value = ctk.CTkLabel(
            self.files_load_box,
            textvariable=self.total_size,
            font=ctk.CTkFont(size=15),
        )
        self.total_size_value.place(relx=0.25, rely=0.135)

        self.selected_file_details_label = ctk.CTkLabel(
            self.files_load_box,
            text="Selected File Details",
            font=ctk.CTkFont(size=22, weight="bold"),
        )
        self.selected_file_details_label.place(relx=0.02, rely=0.2)

        self.file_name_label = ctk.CTkLabel(
            self.files_load_box,
            text="File Name:",
            font=ctk.CTkFont(size=15, weight="bold"),
        )
        self.file_name_label.place(relx=0.02, rely=0.24)

        self.file_name_value = ctk.CTkLabel(
            self.files_load_box,
            textvariable=self.file_name,
            font=ctk.CTkFont(size=15),
        )
        self.file_name_value.place(relx=0.25, rely=0.24)

        self.file_type_label = ctk.CTkLabel(
            self.files_load_box,
            text="File Type:",
            font=ctk.CTkFont(size=15, weight="bold"),
        )
        self.file_type_label.place(relx=0.02, rely=0.265)

        self.file_type_value = ctk.CTkLabel(
            self.files_load_box,
            textvariable=self.file_type,
            font=ctk.CTkFont(size=15),
        )
        self.file_type_value.place(relx=0.25, rely=0.265)

        self.file_size_label = ctk.CTkLabel(
            self.files_load_box,
            text="File Size:",
            font=ctk.CTkFont(size=15, weight="bold"),
        )
        self.file_size_label.place(relx=0.02, rely=0.29)

        self.file_size_value = ctk.CTkLabel(
            self.files_load_box,
            textvariable=self.file_size,
            font=ctk.CTkFont(size=15),
        )
        self.file_size_value.place(relx=0.25, rely=0.29)

        self.media_preview_box = ctk.CTkFrame(
            self.files_load_box,
            corner_radius=20,
            border_width=1,
            border_color="black",
        )
        self.media_preview_box.place(
            relx=0.39, rely=0.01, relwidth=0.60, relheight=0.38
        )

        self.media_preview_label = ctk.CTkLabel(
            self.media_preview_box,
            text="No media selected",
            font=ctk.CTkFont(size=15, weight="bold"),
        )
        self.media_preview_label.place(relx=0.5, rely=0.5, anchor="center")

        self.file_list_frame = ctk.CTkFrame(self.files_load_box, fg_color="black")
        self.file_list_frame.place(
            relx=0.5, rely=0.67, relwidth=0.99, relheight=0.54, anchor="center"
        )

        self.directory_path_entry = ctk.CTkEntry(
            self.files_load_box,
            corner_radius=15,
            font=ctk.CTkFont(size=15),
            state="disabled",
        )
        self.directory_path_entry.place(relx=0.01, rely=0.98, relwidth=0.8, anchor="sw")

        self.browse_directory_button = ctk.CTkButton(
            self.files_load_box,
            text="Browse",
            corner_radius=15,
            command=self.browse_directory,
            font=ctk.CTkFont(size=15, weight="bold"),
        )
        self.browse_directory_button.place(relx=0.98, rely=0.98, anchor="se")

        # SPECIFICATION BOX
        self.specification_box = ctk.CTkFrame(
            self, corner_radius=20, border_width=1, border_color="black"
        )
        self.specification_box.place(
            relx=0.01, rely=0.32, relwidth=0.49, relheight=0.67
        )

        self.specification_label = ctk.CTkLabel(
            self.specification_box,
            text="Object detection test settings",
            font=ctk.CTkFont(size=22, weight="bold"),
        )
        self.specification_label.pack(pady=(20, 5), padx=(20, 0), anchor="w")

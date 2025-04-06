import customtkinter as ctk
from customtkinter import filedialog
from loaders import *
from evaluation import *
from CTkScrollableDropdown import *
from CTkTable import *

ctk.set_appearance_mode("system")
ctk.set_default_color_theme("themes/rime.json")


class GUI(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("ModelTester")
        self._state_before_windows_set_titlebar_color = "zoomed"

        self.configuration_screen()

    def configuration_screen(self):

        self.initialize_variables()

        # region MODEL_BOX
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
            values=["None", "YOLOv11", "YOLOv12", "EfficientDet", "RT-DETRv2", "SSD"],
            command=lambda selected_model: [
                set_model_details(
                    selected_model,
                    self.model_name,
                    self.model_version,
                    self.model_description,
                ),
                self.model_selection_list.set(selected_model),
            ],
            autocomplete=True,
            button_color="transparent",
            alpha=0.9,
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
        # endregion

        # region MEDIA_BOX
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
        self.loaded_sport_value.place(relx=0.15, rely=0.06)

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
        self.media_type_value.place(relx=0.15, rely=0.085)

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
        self.total_files_value.place(relx=0.15, rely=0.11)

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
        self.total_size_value.place(relx=0.15, rely=0.135)

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
        self.file_name_value.place(relx=0.15, rely=0.24)

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
        self.file_type_value.place(relx=0.15, rely=0.265)

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
        self.file_size_value.place(relx=0.15, rely=0.29)

        self.media_preview_box = ctk.CTkFrame(
            self.files_load_box,
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
        self.media_preview_label.bind(
            "<Button-1>",
            lambda event: stop_and_realease_video(self.media_preview_label),
        )

        self.files_table = CTkTable(
            self.files_load_box,
            row=22,
            column=1,
            orientation="vertical",
            anchor="w",
            values=[["File Name"]],
            color_phase="vertical",
            hover=True,
            command=lambda cell: load_file_details(
                cell["value"],
                self.media_directory_path.get(),
                self.file_name,
                self.file_type,
                self.file_size,
                self.media_preview_box.winfo_width(),
                self.media_preview_box.winfo_height(),
                self.media_preview_label,
            ),
        )
        self.files_table.place(
            relx=0.5, rely=0.67, relwidth=0.99, relheight=0.54, anchor="center"
        )
        self.files_table.edit_row(0, font=ctk.CTkFont(size=17, weight="bold"))

        self.directory_path_entry = ctk.CTkEntry(
            self.files_load_box,
            corner_radius=15,
            font=ctk.CTkFont(size=15),
            state="disabled",
            textvariable=self.media_directory_path,
        )
        self.directory_path_entry.place(relx=0.01, rely=0.98, relwidth=0.8, anchor="sw")

        self.browse_directory_button = ctk.CTkButton(
            self.files_load_box,
            text="Browse",
            corner_radius=15,
            command=lambda: [
                self.media_directory_path.set(filedialog.askdirectory()),
                load_media_files(
                    self.media_directory_path.get(),
                    self.loaded_sport,
                    self.media_type,
                    self.total_files,
                    self.total_size,
                    self.files_table,
                    self.metric_fps_checkbox,
                    self.metric_precision_checkbox,
                    self.metric_recall_checkbox,
                    self.metric_f1_checkbox,
                    self.metric_iou_checkbox,
                    self.metric_map_checkbox,
                ),
            ],
            font=ctk.CTkFont(size=15, weight="bold"),
        )
        self.browse_directory_button.place(relx=0.98, rely=0.98, anchor="se")
        # endregion

        # region SPEC_BOX
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

        self.metrics_label = ctk.CTkLabel(
            self.specification_box,
            text="Evaluation Metrics",
            font=ctk.CTkFont(size=17, weight="bold"),
        )
        self.metrics_label.pack(padx=(20, 0), pady=10, anchor="w")

        self.metrics_checkboxes_first_row = ctk.CTkFrame(
            self.specification_box, fg_color="transparent"
        )
        self.metrics_checkboxes_first_row.pack(padx=10, fill="x", anchor="w")

        self.metric_precision_checkbox = ctk.CTkCheckBox(
            self.metrics_checkboxes_first_row,
            text="Precision",
            font=ctk.CTkFont(size=15),
        )
        self.metric_precision_checkbox.pack(
            padx=(20, 0), pady=5, anchor="w", side=ctk.LEFT
        )

        self.metric_recall_checkbox = ctk.CTkCheckBox(
            self.metrics_checkboxes_first_row, text="Recall", font=ctk.CTkFont(size=15)
        )
        self.metric_recall_checkbox.pack(
            padx=(20, 0), pady=5, anchor="w", side=ctk.LEFT
        )

        self.metric_f1_checkbox = ctk.CTkCheckBox(
            self.metrics_checkboxes_first_row,
            text="F1 Score",
            font=ctk.CTkFont(size=15),
        )
        self.metric_f1_checkbox.pack(padx=(20, 0), pady=5, anchor="w", side=ctk.LEFT)

        self.metric_map_checkbox = ctk.CTkCheckBox(
            self.metrics_checkboxes_first_row,
            text="mAP (mean Average Precision)",
            font=ctk.CTkFont(size=15),
        )
        self.metric_map_checkbox.pack(padx=(20, 0), pady=5, anchor="w", side=ctk.LEFT)

        self.metric_iou_checkbox = ctk.CTkCheckBox(
            self.metrics_checkboxes_first_row,
            text="IoU (Intersection over Union)",
            font=ctk.CTkFont(size=15),
        )
        self.metric_iou_checkbox.pack(padx=(20, 0), pady=5, anchor="w", side=ctk.LEFT)

        self.metrics_checkboxes_second_row = ctk.CTkFrame(
            self.specification_box, fg_color="transparent"
        )
        self.metrics_checkboxes_second_row.pack(padx=10, fill="x", anchor="w")

        self.metric_detection_time_checkbox = ctk.CTkCheckBox(
            self.metrics_checkboxes_second_row,
            text="Detection Time",
            font=ctk.CTkFont(size=15),
        )
        self.metric_detection_time_checkbox.pack(
            padx=(20, 0), pady=5, anchor="w", side=ctk.LEFT
        )

        self.metric_fps_checkbox = ctk.CTkCheckBox(
            self.metrics_checkboxes_second_row,
            text="FPS (Frames Per Second)",
            font=ctk.CTkFont(size=15),
        )
        self.metric_fps_checkbox.pack(padx=(20, 0), pady=5, anchor="w", side=ctk.LEFT)

        self.visualization_label = ctk.CTkLabel(
            self.specification_box,
            text="Visualization",
            font=ctk.CTkFont(size=17, weight="bold"),
        )
        self.visualization_label.pack(padx=(20, 0), pady=(20, 0), anchor="w")

        self.enable_visualization_row = ctk.CTkFrame(
            self.specification_box, fg_color="transparent"
        )
        self.enable_visualization_row.pack(padx=10, fill="x", anchor="w")

        self.enable_visualization_label = ctk.CTkLabel(
            self.enable_visualization_row,
            text="Enable visualization during detection: ",
            font=ctk.CTkFont(size=15),
        )
        self.enable_visualization_label.pack(
            padx=(20, 0), pady=10, anchor="w", side=ctk.LEFT
        )

        self.enable_visualization_yes_option = ctk.CTkRadioButton(
            self.enable_visualization_row,
            text="Yes",
            value=1,
            variable=self.enable_visualization_var,
            font=ctk.CTkFont(size=15),
            command=lambda: enable_visualization_options(
                self.enable_visualization_var.get(),
                self.show_bounding_boxes_checkbox,
                self.show_confidence_scores_checkbox,
                self.save_visualizations_checkbox,
            ),
        )
        self.enable_visualization_yes_option.pack(
            padx=(20, 0), pady=10, anchor="w", side=ctk.LEFT
        )

        self.enable_visualization_no_option = ctk.CTkRadioButton(
            self.enable_visualization_row,
            text="No",
            value=0,
            variable=self.enable_visualization_var,
            font=ctk.CTkFont(size=15),
            command=lambda: enable_visualization_options(
                self.enable_visualization_var.get(),
                self.show_bounding_boxes_checkbox,
                self.show_confidence_scores_checkbox,
                self.save_visualizations_checkbox,
            ),
        )
        self.enable_visualization_no_option.pack(pady=10, anchor="w", side=ctk.LEFT)

        self.additional_visualization_options = ctk.CTkFrame(
            self.specification_box, fg_color="transparent"
        )
        self.additional_visualization_options.pack(
            padx=10, pady=(5, 0), fill="x", anchor="w"
        )

        self.show_bounding_boxes_checkbox = ctk.CTkCheckBox(
            self.additional_visualization_options,
            text="Show Bounding Boxes",
            font=ctk.CTkFont(size=15),
            state="normal" if self.enable_visualization_var.get() == 1 else "disabled",
        )
        self.show_bounding_boxes_checkbox.pack(
            padx=(20, 0), pady=5, anchor="w", side=ctk.LEFT
        )

        self.show_confidence_scores_checkbox = ctk.CTkCheckBox(
            self.additional_visualization_options,
            text="Show Confidence Scores",
            font=ctk.CTkFont(size=15),
            state="normal" if self.enable_visualization_var.get() == 1 else "disabled",
        )
        self.show_confidence_scores_checkbox.pack(
            padx=(20, 0), pady=5, anchor="w", side=ctk.LEFT
        )

        self.save_visualizations_checkbox = ctk.CTkCheckBox(
            self.additional_visualization_options,
            text="Save Visualizations to Disk",
            font=ctk.CTkFont(size=15),
            state="normal" if self.enable_visualization_var.get() == 1 else "disabled",
        )
        self.save_visualizations_checkbox.pack(
            padx=(20, 0), pady=5, anchor="w", side=ctk.LEFT
        )

        self.device_label = ctk.CTkLabel(
            self.specification_box,
            text="Select Device:",
            font=ctk.CTkFont(size=17, weight="bold"),
        )
        self.device_label.pack(padx=(20, 0), pady=(20, 0), anchor="w")

        self.device_cpu_option = ctk.CTkRadioButton(
            self.specification_box,
            text="CPU",
            value="CPU",
            variable=self.device_var,
            font=ctk.CTkFont(size=15),
        )
        self.device_cpu_option.pack(padx=(20, 0), pady=5, anchor="w")

        self.device_gpu_option = ctk.CTkRadioButton(
            self.specification_box,
            text="GPU",
            value="GPU",
            variable=self.device_var,
            font=ctk.CTkFont(size=15),
        )
        self.device_gpu_option.pack(padx=(20, 0), pady=5, anchor="w")
        self.results_saving_label = ctk.CTkLabel(
            self.specification_box,
            text="Results Saving Options",
            font=ctk.CTkFont(size=17, weight="bold"),
        )
        self.results_saving_label.pack(padx=(20, 0), pady=(20, 0), anchor="w")

        self.auto_save_checkbox = ctk.CTkCheckBox(
            self.specification_box,
            text="Automatically save test results",
            font=ctk.CTkFont(size=15),
        )
        self.auto_save_checkbox.pack(padx=(20, 0), pady=5, anchor="w")

        self.save_format_label = ctk.CTkLabel(
            self.specification_box,
            text="Save Format:",
            font=ctk.CTkFont(size=15),
        )
        self.save_format_label.pack(padx=(20, 0), pady=5, anchor="w")

        self.save_format_dropdown = ctk.CTkComboBox(
            self.specification_box,
            variable=self.save_format_var,
            font=ctk.CTkFont(size=15),
        )
        self.save_format_dropdown.pack(padx=(20, 20), pady=5, fill="x")

        CTkScrollableDropdown(
            self.save_format_dropdown,
            values=["CSV", "XLSX"],
            autocomplete=True,
            button_color="transparent",
            alpha=0.9,
        )

        self.start_test_button = ctk.CTkButton(
            self.specification_box,
            text="Start Test",
            font=ctk.CTkFont(size=15, weight="bold"),
            corner_radius=15,
            command=lambda: start_test(
                self.model_name.get(),
                self.media_directory_path.get(),
                self.media_type.get(),
                self.device_var.get(),
            ),
        )
        self.start_test_button.pack(pady=(40, 0), anchor="s")

        self.select_default_checkboxes()

    # endregion

    # region VARIABLES_INIT
    def initialize_variables(self):
        # MODEL VARIABLES
        self.model_name = ctk.StringVar(value="None")
        self.model_version = ctk.StringVar(value="None")
        self.model_description = ctk.StringVar(value="None")

        # DIRECTORY VARIABLES
        self.media_directory_path = ctk.StringVar(value="Directory path")
        self.loaded_sport = ctk.StringVar(value="None")
        self.media_type = ctk.StringVar(value="None")
        self.total_files = ctk.StringVar(value="0")
        self.total_size = ctk.StringVar(value="0 KB")

        # FILE VARIABLES
        self.file_name = ctk.StringVar(value="None")
        self.file_type = ctk.StringVar(value="None")
        self.file_size = ctk.StringVar(value="0 KB")

        # SPECIFICATIONS VARIABLES
        self.enable_visualization_var = ctk.IntVar(value=0)
        self.save_format_var = ctk.StringVar(value="CSV")
        self.device_var = ctk.StringVar(value="CPU")

    # endregion

    # region CHCKBOXES_INIT
    def select_default_checkboxes(self):
        """
        DEFAULT STATE: ALL CHECKBOXES ARE SELECTED
        IT MEANS THAT ALL METRIC WILL BE INCLUDED
        IN THE TESTING PROCESS
        """
        self.metric_precision_checkbox.select()
        self.metric_recall_checkbox.select()
        self.metric_f1_checkbox.select()
        self.metric_map_checkbox.select()
        self.metric_iou_checkbox.select()
        self.metric_detection_time_checkbox.select()
        self.metric_fps_checkbox.select()
        self.auto_save_checkbox.select()

    # endregion

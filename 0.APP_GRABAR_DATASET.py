import cv2
import pykinect_azure as pykinect
import tkinter as tk
from tkinter import filedialog, Toplevel, Label
from PIL import Image, ImageTk
import threading
import numpy as np
import os

class KinectApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Kinect Recorder")
        
        # Configuración de estilo
        self.bg_color = "#f0f0f0"
        self.button_color = "#4a4a4a"
        self.button_text_color = "#ffffff"
        self.button_font = ("Helvetica", 14, "bold")
        self.root.configure(bg=self.bg_color)

        # Maximiza y centra la ventana principal
        self.root.state('zoomed')
        self.center_window(self.root)

        self.running = True
        self.save_dir = None
        self.recording = False
        self.rgb_frames = []
        self.depth_frames = []
        self.original_rgb_frames = []
        self.record_thread = None
        
        # Inicializa Kinect
        pykinect.initialize_libraries(track_body=True)
        self.device_config = pykinect.default_configuration
        self.device_config.color_resolution = pykinect.K4A_COLOR_RESOLUTION_1080P
        self.device_config.color_format = pykinect.K4A_IMAGE_FORMAT_COLOR_BGRA32
        self.device_config.depth_mode = pykinect.K4A_DEPTH_MODE_WFOV_2X2BINNED
        
        self.device = pykinect.start_device(config=self.device_config)
        self.bodyTracker = pykinect.start_body_tracker()

        # Botones
        self.select_dir_button = tk.Button(root, text="Select Save Directory", command=self.select_save_directory,
                                           bg=self.button_color, fg=self.button_text_color, font=self.button_font, height=2)
        self.select_dir_button.pack(side=tk.TOP, fill=tk.X, padx=20, pady=10)

        self.start_button = tk.Button(root, text="Start Recording", command=self.start_recording,
                                      bg=self.button_color, fg=self.button_text_color, font=self.button_font, height=2, state=tk.DISABLED)
        self.start_button.pack(side=tk.TOP, fill=tk.X, padx=20, pady=10)

        self.stop_button = tk.Button(root, text="Stop Recording", command=self.stop_recording, state=tk.DISABLED,
                                     bg=self.button_color, fg=self.button_text_color, font=self.button_font, height=2)
        self.stop_button.pack(side=tk.TOP, fill=tk.X, padx=20, pady=10)

        self.save_button = tk.Button(root, text="Save Data", command=self.save_data, state=tk.DISABLED,
                                     bg=self.button_color, fg=self.button_text_color, font=self.button_font, height=2)
        self.save_button.pack(side=tk.TOP, fill=tk.X, padx=20, pady=10)

        self.exit_button = tk.Button(root, text="Exit", command=self.close_app,
                                     bg=self.button_color, fg=self.button_text_color, font=self.button_font, height=2)
        self.exit_button.pack(side=tk.TOP, fill=tk.X, padx=20, pady=10)

        # Frames for video display
        self.video_frame = tk.Frame(root, bg=self.bg_color)
        self.video_frame.pack(side=tk.TOP, padx=10, pady=10, expand=True, fill=tk.BOTH)

        self.rgb_label = tk.Label(self.video_frame, bg=self.bg_color)
        self.rgb_label.pack(side=tk.LEFT, padx=10, expand=True, fill=tk.BOTH)

        self.depth_label = tk.Label(self.video_frame, bg=self.bg_color)
        self.depth_label.pack(side=tk.RIGHT, padx=10, expand=True, fill=tk.BOTH)

        # Start displaying video feed
        self.display_thread = threading.Thread(target=self.display)
        self.display_thread.start()

        # Bind resize event
        self.root.bind('<Configure>', self.on_resize)

    def center_window(self, window):
        window.update_idletasks()
        width = window.winfo_width()
        height = window.winfo_height()
        x = (window.winfo_screenwidth() // 2) - (width // 2)
        y = (window.winfo_screenheight() // 2) - (height // 2)
        window.geometry(f'{width}x{height}+{x}+{y}')

    def on_resize(self, event):
        self.redraw_frames()

    def redraw_frames(self):
        if hasattr(self, 'rgb_frame') and hasattr(self, 'depth_frame'):
            window_width = self.root.winfo_width()
            window_height = self.root.winfo_height()

            new_width = int(window_width * 0.45)
            new_height = int(window_height * 0.45)

            rgb_display = cv2.resize(self.rgb_frame, (new_width, new_height))
            depth_display = cv2.resize(self.depth_frame, (new_width, new_height))

            rgb_img = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(rgb_display, cv2.COLOR_BGRA2RGB)))
            self.rgb_label.config(image=rgb_img)
            self.rgb_label.image = rgb_img

            depth_img = ImageTk.PhotoImage(image=Image.fromarray(depth_display))
            self.depth_label.config(image=depth_img)
            self.depth_label.image = depth_img

    def select_save_directory(self):
        self.save_dir = filedialog.askdirectory()
        if self.save_dir:
            self.start_button.config(state=tk.NORMAL)

    def get_next_folder_name(self):
        base_name = "subject_video_"
        existing_folders = [d for d in os.listdir(self.save_dir) if os.path.isdir(os.path.join(self.save_dir, d)) and d.startswith(base_name)]
        numbers = [int(folder_name.replace(base_name, "")) for folder_name in existing_folders if folder_name.replace(base_name, "").isdigit()]
        next_number = max(numbers) + 1 if numbers else 1
        return os.path.join(self.save_dir, f"{base_name}{next_number}")

    def start_recording(self):
        self.recording = True
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.save_button.config(state=tk.DISABLED)
        self.select_dir_button.config(state=tk.DISABLED)
        self.rgb_frames = []
        self.depth_frames = []
        self.original_rgb_frames = []

    def stop_recording(self):
        self.recording = False
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.save_button.config(state=tk.NORMAL)
        self.select_dir_button.config(state=tk.NORMAL)

    def save_data(self):
        if self.save_dir:
            save_folder = self.get_next_folder_name()
            os.makedirs(save_folder, exist_ok=True)

            rgb_dir = os.path.join(save_folder, "rgb")
            depth_dir = os.path.join(save_folder, "depth")
            os.makedirs(rgb_dir, exist_ok=True)
            os.makedirs(depth_dir, exist_ok=True)

            saving_window = Toplevel(self.root)
            saving_window.title("Saving Data")

            saving_label = Label(saving_window, text="Guardando datos...")
            saving_label.pack(padx=20, pady=20)

            self.center_window(saving_window)

            def perform_save():
                # Guardar frames RGB originales como imágenes
                for i, frame in enumerate(self.original_rgb_frames):
                    rgb_image_path = os.path.join(rgb_dir, f"rgb_frame_{i}.png")
                    cv2.imwrite(rgb_image_path, cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR))

                # Guardar frames Depth como PNG de 16 bits
                for i, frame in enumerate(self.depth_frames):
                    depth_image_path = os.path.join(depth_dir, f"depth_frame_{i}.png")
                    # frame ya es uint16 directamente desde el Kinect
                    cv2.imwrite(depth_image_path, frame.astype(np.uint16))

                saving_label.config(text="Datos guardados")
                self.root.after(2000, saving_window.destroy)

            threading.Thread(target=perform_save).start()
            self.save_button.config(state=tk.DISABLED)

    def close_app(self):
        self.running = False
        if self.display_thread is not None:
            self.display_thread.join()
        self.release_resources()
        self.root.destroy()

    def release_resources(self):
        if self.device:
            self.device.stop_cameras()
            self.device.close()
        if self.bodyTracker:
            self.bodyTracker.shutdown()
        print("Resources released and device closed successfully")

    def display(self):
        while self.running:
            capture = self.device.update()
            self.bodyTracker.update()  # Mantiene el tracker activo sin usar el resultado

            ret_color, color_image = capture.get_color_image()
            ret_depth, depth_image = capture.get_depth_image()

            if not ret_color or not ret_depth:
                continue

            # Almacenar frames si se está grabando
            if self.recording:
                self.original_rgb_frames.append(color_image.copy())
                self.rgb_frames.append(color_image)
                self.depth_frames.append(depth_image.copy())  # uint16 sin modificar

            # Mostrar imagen RGB limpia, sin landmarks superpuestos
            self.rgb_frame = color_image
            # Normalizar solo para visualización (no afecta lo guardado)
            self.depth_frame = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

            self.redraw_frames()

# Crear la ventana principal
root = tk.Tk()
app = KinectApp(root)
root.mainloop()
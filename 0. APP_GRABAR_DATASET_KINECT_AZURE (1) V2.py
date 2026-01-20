import cv2
import pykinect_azure as pykinect
import tkinter as tk
from tkinter import filedialog, Toplevel, Label
from PIL import Image, ImageTk
import threading
import numpy as np
import os
import xml.etree.ElementTree as ET

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
        self.skeleton_data_pixel = []  # Inicializa la lista para coordenadas en píxeles
        self.skeleton_data_real = []   # Inicializa la lista para coordenadas en el mundo real
        self.original_rgb_frames = []  # Almacenar los frames RGB originales
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
        # Redraw the frames on resize to maintain proportionality
        self.redraw_frames()

    def redraw_frames(self):
        if hasattr(self, 'rgb_frame') and hasattr(self, 'depth_frame'):
            # Get current window size
            window_width = self.root.winfo_width()
            window_height = self.root.winfo_height()

            # Calculate new sizes for the images
            new_width = int(window_width * 0.45)
            new_height = int(window_height * 0.45)

            # Resize images for display
            rgb_display = cv2.resize(self.rgb_frame, (new_width, new_height))
            depth_display = cv2.resize(self.depth_frame, (new_width, new_height))

            # Convert images to display
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
        self.skeleton_data = []

    def stop_recording(self):
        self.recording = False
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.save_button.config(state=tk.NORMAL)
        self.select_dir_button.config(state=tk.NORMAL)

    def save_data(self):
        if self.save_dir:
            # Crear una carpeta única para guardar los datos
            save_folder = self.get_next_folder_name()
            os.makedirs(save_folder, exist_ok=True)

            # Crear directorios para RGB y Depth dentro de la carpeta única
            rgb_dir = os.path.join(save_folder, "rgb")
            depth_dir = os.path.join(save_folder, "depth")
            os.makedirs(rgb_dir, exist_ok=True)
            os.makedirs(depth_dir, exist_ok=True)

            # Crear y mostrar la ventana "saving data"
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

                # Guardar frames Depth como archivos XML
                for i, frame in enumerate(self.depth_frames):
                    depth_image_path = os.path.join(depth_dir, f"depth_frame_{i}.xml")
                    self.save_depth_to_xml(depth_image_path, frame, i)

                # Save skeleton data (2D and 3D coordinates)
                skeleton_pixel_path = os.path.join(save_folder, "skeleton_data_pixel.txt")
                skeleton_real_path = os.path.join(save_folder, "skeleton_data_real.txt")

                # Save pixel coordinates (2D)
                with open(skeleton_pixel_path, "w") as f:
                    for frame_data in self.skeleton_data_pixel:
                        f.write(f"{frame_data}\n")

                # Save real-world coordinates (3D)
                with open(skeleton_real_path, "w") as f:
                    for frame_data in self.skeleton_data_real:
                        f.write(f"{frame_data}\n")

                # Actualizar la etiqueta y cerrar la ventana después de un retraso
                saving_label.config(text="Datos guardados")
                self.root.after(2000, saving_window.destroy)  # Cerrar después de 2 segundos

            threading.Thread(target=perform_save).start()
            self.save_button.config(state=tk.DISABLED)

    def save_depth_to_xml(self, file_path, depth_frame, frame_index):
        # Crear la estructura XML
        root = ET.Element("lsp_storage")
        depth_img_tag = ET.SubElement(root, f"depthImg{frame_index}")

        width = ET.SubElement(depth_img_tag, "width")
        width.text = str(depth_frame.shape[1])

        height = ET.SubElement(depth_img_tag, "height")
        height.text = str(depth_frame.shape[0])

        origin = ET.SubElement(depth_img_tag, "origin")
        origin.text = "top-left"

        layout = ET.SubElement(depth_img_tag, "layout")
        layout.text = "interleaved"

        dt = ET.SubElement(depth_img_tag, "dt")
        dt.text = "w"

        data = ET.SubElement(depth_img_tag, "data")
        # Convertir la matriz de profundidad en una cadena separada por espacios
        depth_data_str = ' '.join(map(str, depth_frame.flatten()))
        data.text = depth_data_str

        # Guardar el XML en el archivo
        tree = ET.ElementTree(root)
        tree.write(file_path, encoding='utf-8', xml_declaration=True)

    def close_app(self):
        self.running = False
        if self.display_thread is not None:
            self.display_thread.join()  # Asegurar que el hilo termine antes de continuar
        self.release_resources()
        self.root.destroy()

    def release_resources(self):
        # Asegurarse de liberar los recursos de Kinect
        if self.device:
            self.device.stop_cameras()
            self.device.close()
        if self.bodyTracker:
            self.bodyTracker.shutdown()
        print("Resources released and device closed successfully")

    def display(self):
        while self.running:
            capture = self.device.update()
            body_frame = self.bodyTracker.update()

            ret_color, color_image = capture.get_color_image()
            ret_depth, depth_image = capture.get_depth_image()

            if not ret_color or not ret_depth:
                continue

            # Almacenar frames si se está grabando
            if self.recording:
                self.original_rgb_frames.append(color_image.copy())  # Guardar la imagen RGB original
                self.rgb_frames.append(color_image)  # Esta versión puede ser modificada para visualización
                self.depth_frames.append(depth_image)

                skeleton_data_pixel_frame = []
                skeleton_data_real_frame = []
                for body_id in range(body_frame.get_num_bodies()):
                    skeleton_2d = body_frame.get_body2d(body_id, pykinect.K4A_CALIBRATION_TYPE_COLOR).numpy()
                    skeleton_data_pixel_frame.append(skeleton_2d.flatten())
                    #print("Aqui:",skeleton_2d)
                    skeleton_3d = body_frame.get_body(body_id).numpy()
                    skeleton_data_real_frame.append(skeleton_3d.flatten())

                self.skeleton_data_pixel.append(" ".join(map(str, np.concatenate(skeleton_data_pixel_frame))))
                self.skeleton_data_real.append(" ".join(map(str, np.concatenate(skeleton_data_real_frame))))
            # Dibujar esqueleto en la imagen RGB para visualización
            self.rgb_frame = body_frame.draw_bodies(color_image, pykinect.K4A_CALIBRATION_TYPE_COLOR)
            self.depth_frame = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

            self.redraw_frames()

# Crear la ventana principal
root = tk.Tk()
app = KinectApp(root)
root.mainloop()


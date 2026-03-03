import tkinter as tk
from tkinter import ttk, messagebox
import os, re, shutil, json
from PIL import Image, ImageTk
import numpy as np
import cv2

class LabelingApp:

    def __init__(self, root):
        self.root = root
        self.root.title("Programa de Etiquetado")
        
        # Ruta principal de los datos
        self.data_path = r"C:\Users\20214\Documents\prueba"
        
        # Variables de selección
        self.subject_folders = []
        self.current_subject = None
        self.sample_folders = []
        self.current_sample = None

        
        # Imagenes
        self.rgb_files = []
        self.current_frame_index = 0
        
        # Etiquetado: lista de intervalos marcados (start, end, label)
        self.labels = []  
        
        # Variables para marcar inicio y fin
        self.mark_start = None
        self.mark_end = None
        
        # Lista de clases (para expresiones faciales)
        self.facial_classes = [
            "lets see",
            "bored",
            "tired",
            "disgust",
            "happy",
            "smells bad",
            "thief",
            "cry",
            "annoyed",
            "no",
            "i dont know",
            "yes",
            "surprise",
            "sad"
            ]




# CAMBIAR CLASES       
        
        # Construir la interfaz
        self.build_gui()
        
        # Cargar los sujetos
        self.load_subjects()
    
    def build_gui(self):
        # Panel superior: selección de sujeto y muestra
        top_frame = ttk.Frame(self.root)
        top_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
        
        ttk.Label(top_frame, text="Sujeto:").pack(side=tk.LEFT, padx=5)
        self.subject_var = tk.StringVar()
        self.subject_menu = ttk.OptionMenu(top_frame, self.subject_var, None)
        self.subject_menu.pack(side=tk.LEFT, padx=5)
        self.subject_var.trace("w", lambda *args: self.subject_changed())
        
        ttk.Label(top_frame, text="Muestra:").pack(side=tk.LEFT, padx=5)
        self.sample_var = tk.StringVar()
        self.sample_menu = ttk.OptionMenu(top_frame, self.sample_var, None)
        self.sample_menu.pack(side=tk.LEFT, padx=5)
        self.sample_var.trace("w", lambda *args: self.sample_changed())
        
        # Panel de imágenes
        image_frame = ttk.Frame(self.root)
        image_frame.pack(side=tk.TOP, padx=5, pady=5)

        # Imagen original (sin puntos)
        self.image_panel_original = ttk.Label(image_frame)
        self.image_panel_original.pack(side=tk.LEFT, padx=5)

        # Imagen con puntos
        self.image_panel_keypoints = ttk.Label(image_frame)
        self.image_panel_keypoints.pack(side=tk.LEFT, padx=5)
        
        # Panel de navegación
        nav_frame = ttk.Frame(self.root)
        nav_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
        self.prev_button = ttk.Button(nav_frame, text="Anterior", command=self.prev_frame)
        self.prev_button.pack(side=tk.LEFT, padx=5)
        self.next_button = ttk.Button(nav_frame, text="Siguiente", command=self.next_frame)
        self.next_button.pack(side=tk.LEFT, padx=5)
        
        # Nueva funcionalidad: Ir a frame específico
        ttk.Label(nav_frame, text="Ir a frame:").pack(side=tk.LEFT, padx=(20, 5))
        self.frame_entry = ttk.Entry(nav_frame, width=8)
        self.frame_entry.pack(side=tk.LEFT, padx=5)
        self.go_button = ttk.Button(nav_frame, text="Ir", command=self.go_to_frame)
        self.go_button.pack(side=tk.LEFT, padx=5)
        
        # Bind Enter key to go to frame
        self.frame_entry.bind("<Return>", lambda event: self.go_to_frame())
        
        # Panel para marcar inicio y fin de etiqueta y seleccionar clase
        mark_frame = ttk.Frame(self.root)
        mark_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
        self.start_button = ttk.Button(mark_frame, text="Marcar inicio", command=self.mark_start_frame)
        self.start_button.pack(side=tk.LEFT, padx=5)
        self.end_button = ttk.Button(mark_frame, text="Marcar fin", command=self.mark_end_frame)
        self.end_button.pack(side=tk.LEFT, padx=5)
        
        ttk.Label(mark_frame, text="Clase:").pack(side=tk.LEFT, padx=5)
        self.class_var = tk.StringVar()
        self.class_var.set(self.facial_classes[0])
        self.class_menu = ttk.OptionMenu(mark_frame, self.class_var, self.facial_classes[0], *self.facial_classes)
        self.class_menu.pack(side=tk.LEFT, padx=5)
        
        self.add_label_button = ttk.Button(mark_frame, text="Agregar etiqueta", command=self.add_label)
        self.add_label_button.pack(side=tk.LEFT, padx=5)
        
        # Lista para mostrar los intervalos etiquetados
        self.labels_listbox = tk.Listbox(self.root, height=28) ##AQUI SE PUEDE MOFICIAR LA ALTURA DE LA LISTA
        self.labels_listbox.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
        
        # Botón para eliminar etiqueta seleccionada
        self.delete_label_button = ttk.Button(self.root, text="Eliminar etiqueta seleccionada", command=self.delete_label)
        self.delete_label_button.pack(side=tk.TOP, padx=5, pady=5)
        
        # Botón para procesar y guardar las etiquetas
        self.process_button = ttk.Button(self.root, text="Procesar", command=self.process_labels)
        self.process_button.pack(side=tk.TOP, padx=5, pady=5)
        
        # Bind de teclas para navegación (flechas izquierda y derecha)
        self.root.bind("<Left>", lambda event: self.prev_frame())
        self.root.bind("<Right>", lambda event: self.next_frame())
    
    def go_to_frame(self):
        """Nueva función para ir directamente a un frame específico"""
        try:
            frame_num = int(self.frame_entry.get())
            if self.rgb_files and 0 <= frame_num < len(self.rgb_files):
                self.show_frame(frame_num)
            else:
                if self.rgb_files:
                    messagebox.showerror("Error", f"Frame debe estar entre 0 y {len(self.rgb_files)-1}")
                else:
                    messagebox.showerror("Error", "No hay imágenes cargadas")
        except ValueError:
            messagebox.showerror("Error", "Ingrese un número válido")
    
    def load_subjects(self):
        # Lista las carpetas que comienzan con "SUJETO" en la ruta de datos
        if os.path.exists(self.data_path):
            self.subject_folders = [f for f in os.listdir(self.data_path)
                                    if os.path.isdir(os.path.join(self.data_path, f)) and f.startswith("SUJETO")]
            self.subject_folders.sort()
            menu = self.subject_menu["menu"]
            menu.delete(0, "end")
            for subj in self.subject_folders:
                menu.add_command(label=subj, command=lambda value=subj: self.subject_var.set(value))
            if self.subject_folders:
                self.subject_var.set(self.subject_folders[0])
        else:
            messagebox.showerror("Error", f"No se encuentra la ruta: {self.data_path}")
    
    def subject_changed(self):
        self.current_subject = self.subject_var.get()
        subject_path = os.path.join(self.data_path, self.current_subject)
        # Lista las subcarpetas de muestra (excluye la carpeta de salida si existe)
        self.sample_folders = [f for f in os.listdir(subject_path)
                               if os.path.isdir(os.path.join(subject_path, f)) and f != "señas_procesadas"]
        self.sample_folders.sort()
        menu = self.sample_menu["menu"]
        menu.delete(0, "end")
        for samp in self.sample_folders:
            menu.add_command(label=samp, command=lambda value=samp: self.sample_var.set(value))
        if self.sample_folders:
            self.sample_var.set(self.sample_folders[0])
    
    def sample_changed(self):
        self.current_sample = self.sample_var.get()
        # Al cambiar la muestra se cargan las imágenes RGB
        self.load_rgb_images()
        # Cargar etiquetas persistentes si existen
        self.load_labels_from_file()
    
    def load_rgb_images(self):
        self.rgb_files = []
        self.current_frame_index = 0
        if self.current_subject and self.current_sample:
            rgb_folder = os.path.join(self.data_path, self.current_subject, self.current_sample, "rgb")
            if os.path.exists(rgb_folder):
                files = [f for f in os.listdir(rgb_folder) if f.startswith("rgb_frame_") and f.endswith(".png")]
                def get_frame_num(filename):
                    m = re.search(r"rgb_frame_(\d+)", filename)
                    return int(m.group(1)) if m else -1
                files.sort(key=get_frame_num)
                self.rgb_files = [os.path.join(rgb_folder, f) for f in files]
                if self.rgb_files:
                    self.show_frame(0)
                else:
                    messagebox.showwarning("Aviso", "No hay imágenes RGB en la muestra seleccionada.")
            else:
                messagebox.showwarning("Aviso", f"No existe la carpeta 'rgb' en la muestra: {self.current_sample}")
    
    def show_frame(self, index):
        if self.rgb_files and 0 <= index < len(self.rgb_files):
            self.current_frame_index = index
            img_path = self.rgb_files[index]
            img = Image.open(img_path)
            img = img.resize((640, 480))  # Ajusta el tamaño si es necesario
            self.photo = ImageTk.PhotoImage(img)
            self.show_frame_keypoints(index)  # <- Directamente llamar aquí
            #self.root.title(f"Frame {index} de {len(self.rgb_files)}")

    def show_frame_keypoints(self, index):
        subject_path = os.path.join(self.data_path, self.current_subject)
        sample_dir = os.path.join(subject_path, self.current_sample)
        skeleton_pixel_src = os.path.join(sample_dir, "skeleton_data_pixel.txt")
        skeleton_real_src = os.path.join(sample_dir, "skeleton_data_real.txt")
        skeleton_data_src = os.path.join(sample_dir, "skeleton_data.txt")

        if self.rgb_files and 0 <= index < len(self.rgb_files):
            self.current_frame_index = index
            img_path = self.rgb_files[index]
            
            # Cargar imagen base (sin puntos)
            img = Image.open(img_path)
            original_width, original_height = img.size
            img_resized = img.resize((640, 480))
            img_np_base = np.array(img_resized)

            # Copia para dibujar puntos
            img_np_keypoints = img_np_base.copy()

            if os.path.exists(skeleton_pixel_src):
                data = np.loadtxt(skeleton_pixel_src, usecols=range(64))
                points = data[index]
                x_coords = points[::2]
                y_coords = points[1::2]
                x_norm = x_coords / original_width * 640
                y_norm = y_coords / original_height * 480
                # Dibujar puntos
                for x, y in zip(x_norm, y_norm):
                    x_int = int(round(x))
                    y_int = int(round(y))
                    if 0 <= x_int < 640 and 0 <= y_int < 480:
                        cv2.circle(img_np_keypoints, (x_int, y_int), radius=2, color=(0, 255, 0), thickness=-1)
                img_pil_keypoints = Image.fromarray(img_np_keypoints)
            else:
                img_pil_keypoints = Image.fromarray(img_np_base)

            # Dibujar puntos sobre img_np_keypoints
            # for x, y in zip(x_coords, y_coords):
            #     x_int = int(round(x))
            #     y_int = int(round(y))
            #     if 0 <= x_int < 640 and 0 <= y_int < 480:
            #         cv2.circle(img_np_keypoints, (x_int, y_int), radius=1, color=(0, 255, 0), thickness=-1)
            
            # Convertir imágenes a PIL
            img_pil_base = Image.fromarray(img_np_base)
            

            # Mostrar ambas imágenes
            self.photo_base = ImageTk.PhotoImage(img_pil_base)
            self.photo_keypoints = ImageTk.PhotoImage(img_pil_keypoints)

            self.image_panel_original.config(image=self.photo_base)
            self.image_panel_original.image = self.photo_base

            self.image_panel_keypoints.config(image=self.photo_keypoints)
            self.image_panel_keypoints.image = self.photo_keypoints

            self.root.title(f"Frame {index} de {len(self.rgb_files)}")
    
    def prev_frame(self):
        if self.current_frame_index > 0:
            self.show_frame(self.current_frame_index - 1)
    
    def next_frame(self):
        if self.rgb_files and self.current_frame_index < len(self.rgb_files) - 1:
            self.show_frame(self.current_frame_index + 1)
    
    def mark_start_frame(self):
        self.mark_start = self.current_frame_index
        messagebox.showinfo("Marcado", f"Inicio marcado en el frame {self.mark_start}")
    
    def mark_end_frame(self):
        self.mark_end = self.current_frame_index
        messagebox.showinfo("Marcado", f"Fin marcado en el frame {self.mark_end}")
    
    def add_label(self):
        if self.mark_start is None or self.mark_end is None:
            messagebox.showerror("Error", "Debe marcar ambos, inicio y fin, para agregar una etiqueta.")
            return
        start = min(self.mark_start, self.mark_end)
        end = max(self.mark_start, self.mark_end)
        label = self.class_var.get()
        self.labels.append({"start": start, "end": end, "label": label})
        self.labels_listbox.insert(tk.END, f"{label}: {start} - {end}")
        self.mark_start = None
        self.mark_end = None
        self.save_labels_to_file()
    
    def delete_label(self):
        selection = self.labels_listbox.curselection()
        if not selection:
            messagebox.showwarning("Aviso", "Seleccione una etiqueta para eliminar.")
            return
        index = selection[0]
        del self.labels[index]
        self.labels_listbox.delete(index)
        self.save_labels_to_file()
    
    def update_labels_listbox(self):
        self.labels_listbox.delete(0, tk.END)
        for entry in self.labels:
            self.labels_listbox.insert(tk.END, f"{entry['label']}: {entry['start']} - {entry['end']}")
    
    def save_labels_to_file(self):
        sample_dir = os.path.join(self.data_path, self.current_subject, self.current_sample)
        labels_file = os.path.join(sample_dir, "etiquetas.json")
        try:
            with open(labels_file, "w") as f:
                json.dump(self.labels, f)
        except Exception as e:
            messagebox.showerror("Error", f"Error al guardar etiquetas: {e}")
    
    def load_labels_from_file(self):
        sample_dir = os.path.join(self.data_path, self.current_subject, self.current_sample)
        labels_file = os.path.join(sample_dir, "etiquetas.json")
        if os.path.exists(labels_file):
            try:
                with open(labels_file, "r") as f:
                    self.labels = json.load(f)
                self.update_labels_listbox()
            except Exception as e:
                messagebox.showerror("Error", f"Error al cargar etiquetas: {e}")
        else:
            self.labels = []
            self.labels_listbox.delete(0, tk.END)
    
    def process_labels(self):
        if not self.labels:
            messagebox.showwarning("Aviso", "No hay etiquetas para procesar.")
            return
        
        subject_path = os.path.join(self.data_path, self.current_subject)
        output_dir = os.path.join(subject_path, "señas_procesadas")
        os.makedirs(output_dir, exist_ok=True)
        
        sample_dir = os.path.join(subject_path, self.current_sample)
        
        # Crear y mostrar barra de progreso
        progress = ttk.Progressbar(self.root, orient="horizontal", mode="determinate", maximum=len(self.labels))
        progress.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
        
        for i, entry in enumerate(self.labels):
            label = entry["label"]
            start = entry["start"]
            end = entry["end"]
            
            label_folder = os.path.join(output_dir, label)
            os.makedirs(label_folder, exist_ok=True)
            existing = [d for d in os.listdir(label_folder) if os.path.isdir(os.path.join(label_folder, d)) and d.startswith("muestra_")]
            next_index = len(existing) + 1
            muestra_folder = os.path.join(label_folder, f"muestra_{next_index}")
            os.makedirs(muestra_folder, exist_ok=True)
            
            rgb_src_folder = os.path.join(sample_dir, "rgb")
            rgb_dest_folder = os.path.join(muestra_folder, "rgb")
            os.makedirs(rgb_dest_folder, exist_ok=True)
            for i_frame in range(start, end + 1):
                filename = f"rgb_frame_{i_frame}.png"
                src = os.path.join(rgb_src_folder, filename)
                if os.path.exists(src):
                    shutil.copy(src, os.path.join(rgb_dest_folder, filename))
            
            depth_src_folder = os.path.join(sample_dir, "depth")
            depth_dest_folder = os.path.join(muestra_folder, "depth")
            if os.path.exists(depth_src_folder):
                os.makedirs(depth_dest_folder, exist_ok=True)
                for i_frame in range(start, end + 1):
                    for ext in [".png", ".xml"]:
                        filename = f"depth_frame_{i_frame}{ext}"
                        src = os.path.join(depth_src_folder, filename)
                        if os.path.exists(src):
                            shutil.copy(src, os.path.join(depth_dest_folder, filename))
            
            skeleton_pixel_src = os.path.join(sample_dir, "skeleton_data_pixel.txt")
            if os.path.exists(skeleton_pixel_src):
                with open(skeleton_pixel_src, "r") as f:
                    lines = f.readlines()
                selected_lines = lines[start:end+1]
                skeleton_pixel_dest = os.path.join(muestra_folder, "skeleton_data_pixel.txt")
                with open(skeleton_pixel_dest, "w") as f:
                    f.writelines(selected_lines)
            
            skeleton_real_src = os.path.join(sample_dir, "skeleton_data_real.txt")
            if os.path.exists(skeleton_real_src):
                with open(skeleton_real_src, "r") as f:
                    lines = f.readlines()
                selected_lines = lines[start:end+1]
                skeleton_real_dest = os.path.join(muestra_folder, "skeleton_data_real.txt")
                with open(skeleton_real_dest, "w") as f:
                    f.writelines(selected_lines)
            
            # Actualizar barra de progreso
            progress['value'] = i + 1
            self.root.update_idletasks()
        
        progress.destroy()
        messagebox.showinfo("Procesado", "Etiquetado procesado y guardado en la carpeta 'señas_procesadas'.")
        # Se limpian las etiquetas tras procesar
        self.labels = []
        self.labels_listbox.delete(0, tk.END)
        self.save_labels_to_file()

if __name__ == "__main__":
    root = tk.Tk()
    app = LabelingApp(root)
    root.mainloop()
import os
import cv2
import numpy as np
import random
from tqdm import tqdm

# ======= CONFIGURACIÓN PRINCIPAL =======
base_path = r"D:\SALIDA\SALIDA"
log_path = os.path.join(base_path, "errores_lectura.txt")
open(log_path, "w", encoding="utf-8").close()  # Reiniciar log

# Listas de sujetos
sujetos_H = [1, 10, 11, 12, 14]
sujetos_M = [2, 3, 4, 6, 7]
todos_sujetos = sujetos_H + sujetos_M

# ======= FUNCIONES DE AUMENTACIÓN =======
def apply_flip(image):
    """Invierte horizontalmente."""
    return cv2.flip(image, 1)

def apply_rotation(image, angle):
    """Rota con un ángulo fijo (misma dirección para todos los frames de la muestra)."""
    h, w = image.shape[:2]
    M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1)
    return cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REFLECT)

def apply_grayscale(image):
    """Convierte a escala de grises."""
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def apply_stretch(image, factor):
    """Estira horizontal o verticalmente con un factor más grande (1.35–1.55)."""
    h, w = image.shape[:2]
    if random.choice([True, False]):
        new_w = int(w * factor)
        stretched = cv2.resize(image, (new_w, h))
    else:
        new_h = int(h * factor)
        stretched = cv2.resize(image, (w, new_h))
    return cv2.resize(stretched, (w, h))

# ======= RECORRER TODOS LOS SUJETOS =======
for num in todos_sujetos:
    sujeto_nombre = f"SUJETO {num}"
    sujeto_path = os.path.join(base_path, sujeto_nombre, "señas_procesadas")

    if not os.path.isdir(sujeto_path):
        print(f" No se encontró la carpeta: {sujeto_path}")
        continue

    print(f"\n Procesando {sujeto_nombre}\n")

    # ======= Recorrer clases =======
    for clase in os.listdir(sujeto_path):
        clase_path = os.path.join(sujeto_path, clase)
        if not os.path.isdir(clase_path):
            continue

        for muestra in os.listdir(clase_path):
            muestra_path = os.path.join(clase_path, muestra)
            rgb_path = os.path.join(muestra_path, "rgb")

            if not os.path.isdir(rgb_path):
                continue

            frames = [f for f in os.listdir(rgb_path) if f.lower().endswith(".png")]
            if not frames:
                continue

            print(f" - Clase: {clase} | Muestra: {muestra} | Frames: {len(frames)}")

            # Definir ángulo fijo y factor fijo por muestra
            rotation_angle = random.choice([-12, -8, 8, 12])  # una dirección fija (suave)
            stretch_factor = random.uniform(1.35, 1.55)  # más notorio

            # Definir aumentaciones con parámetros por muestra
            augmentations = {
                "FLIP": lambda img: apply_flip(img),
                "ROTATION": lambda img: apply_rotation(img, rotation_angle),
                "GRAY": lambda img: apply_grayscale(img),
                "STRETCH": lambda img: apply_stretch(img, stretch_factor)
            }

            # ======= Aplicar todas las aumentaciones =======
            for aug_name, aug_func in augmentations.items():
                output_dir = os.path.join(muestra_path, f"rgb_{aug_name}")
                os.makedirs(output_dir, exist_ok=True)

                for frame_file in tqdm(frames, desc=f"{sujeto_nombre} - {aug_name} ({clase}/{muestra})", leave=False):
                    frame_path = os.path.join(rgb_path, frame_file)

                    try:
                        # Leer imagen de forma segura
                        image = cv2.imdecode(np.fromfile(frame_path, dtype=np.uint8), cv2.IMREAD_COLOR)
                    except Exception:
                        image = None

                    if image is None:
                        with open(log_path, "a", encoding="utf-8") as f:
                            f.write(f"No se pudo leer: {frame_path}\n")
                        continue

                    try:
                        augmented = aug_func(image)
                        if augmented is None:
                            with open(log_path, "a", encoding="utf-8") as f:
                                f.write(f"Error en augmentation {aug_name} para {frame_path}\n")
                            continue

                        # Si es gris (1 canal), convertir a BGR antes de guardar
                        if len(augmented.shape) == 2:
                            augmented = cv2.cvtColor(augmented, cv2.COLOR_GRAY2BGR)

                        output_path = os.path.join(output_dir, frame_file)
                        cv2.imencode(".png", augmented)[1].tofile(output_path)

                    except Exception as e:
                        with open(log_path, "a", encoding="utf-8") as f:
                            f.write(f"Error en {frame_path}: {e}\n")
                        continue

print("\nProceso completado para todos los sujetos (H y M).")
print(f" Revisa el archivo de errores (si los hubo): {log_path}")
import cv2
import mediapipe as mp
import numpy as np
import os
import re
from pathlib import Path
from PIL import Image

# ===================== CONFIG =====================
BASE_ROOT = r"C:\Users\20214\Documents\prueba"
MAX_NUM_FACES = 1
REFINE_LANDMARKS = False
MIN_DET_CONF = 0.5
MIN_TRK_CONF = 0.5

SKIP_IF_EXISTS = False

FILL_ZEROS_ON_FAIL = True
# ==================================================

RGB_FRAME_RE = re.compile(r"^rgb_frame_(\d+)\.(png|jpg|jpeg|bmp)$", re.IGNORECASE)

def sorted_image_files(rgb_folder: Path):
    imgs = []
    for fn in os.listdir(rgb_folder):
        if fn.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
            m = RGB_FRAME_RE.match(fn)
            idx = int(m.group(1)) if m else None
            imgs.append((idx, fn))
    imgs.sort(key=lambda x: (x[0] is None, x[0] if x[0] is not None else x[1]))
    return [fn for _idx, fn in imgs]

def extract_face_landmarks_from_rgb(rgb_folder: Path, out_muestra_folder: Path):
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=MAX_NUM_FACES,
        refine_landmarks=REFINE_LANDMARKS,
        min_detection_confidence=MIN_DET_CONF,
        min_tracking_confidence=MIN_TRK_CONF
    )

    image_files = sorted_image_files(rgb_folder)
    if not image_files:
        print(f"[SKIP] No se encontraron imágenes en {rgb_folder}")
        face_mesh.close()
        return False

    n_landmarks = 478 if REFINE_LANDMARKS else 468

    landmarks_3d = []
    landmarks_2d = []
    successful_frames = 0
    failed_frames = 0

    for idx, image_file in enumerate(image_files):
        image_path = rgb_folder / image_file

        try:
            pil_image = Image.open(image_path)
            image = np.array(pil_image)

            if image.ndim == 3 and image.shape[-1] == 4:
                image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
            elif image.ndim == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

            height, width = image.shape[:2]
            results = face_mesh.process(image)

            if results.multi_face_landmarks:
                face_landmarks = results.multi_face_landmarks[0]

                frame_landmarks_3d = np.zeros(n_landmarks * 3, dtype=np.float32)
                frame_landmarks_2d = np.zeros(n_landmarks * 2, dtype=np.int32)

                for j, landmark in enumerate(face_landmarks.landmark):
                    frame_landmarks_3d[j*3 + 0] = landmark.x
                    frame_landmarks_3d[j*3 + 1] = landmark.y
                    frame_landmarks_3d[j*3 + 2] = landmark.z

                    frame_landmarks_2d[j*2 + 0] = int(landmark.x * width)
                    frame_landmarks_2d[j*2 + 1] = int(landmark.y * height)

                landmarks_3d.append(frame_landmarks_3d)
                landmarks_2d.append(frame_landmarks_2d)
                successful_frames += 1

            else:
                if FILL_ZEROS_ON_FAIL:
                    landmarks_3d.append(np.zeros(n_landmarks * 3, dtype=np.float32))
                    landmarks_2d.append(np.zeros(n_landmarks * 2, dtype=np.int32))
                else:
                    landmarks_3d.append(None)
                    landmarks_2d.append(None)
                failed_frames += 1

            if (idx + 1) % 50 == 0:
                print(f"  Procesados {idx + 1}/{len(image_files)} frames")

        except Exception as e:
            print(f"  Error al procesar {image_file}: {e}")
            if FILL_ZEROS_ON_FAIL:
                landmarks_3d.append(np.zeros(n_landmarks * 3, dtype=np.float32))
                landmarks_2d.append(np.zeros(n_landmarks * 2, dtype=np.int32))
            else:
                landmarks_3d.append(None)
                landmarks_2d.append(None)
            failed_frames += 1

    face_mesh.close()

    out_real = out_muestra_folder / "skeleton_data_real.txt"
    out_pix  = out_muestra_folder / "skeleton_data_pixel.txt"

    with open(out_real, "w", encoding="utf-8", newline="") as f:
        f.write(f"{len(image_files)}\n")
        for v in landmarks_3d:
            if v is None:
                f.write("\n")
            else:
                f.write(" ".join(map(str, v.tolist())) + "\n")

    with open(out_pix, "w", encoding="utf-8", newline="") as f:
        f.write(f"{len(image_files)}\n")
        for v in landmarks_2d:
            if v is None:
                f.write("\n")
            else:
                f.write(" ".join(map(str, v.tolist())) + "\n")

    print(f"  Guardado en: {out_muestra_folder}")
    print(f"     - {out_pix.name} ({len(image_files)} frames x {n_landmarks*2} vals)")
    print(f"     - {out_real.name} ({len(image_files)} frames x {n_landmarks*3} vals)")
    print(f"     Frames exitosos: {successful_frames}/{len(image_files)} | fallidos: {failed_frames}/{len(image_files)}")

    return True

def find_all_rgb_folders(base_root: Path):
    rgb_folders = []
    for root, dirs, _files in os.walk(base_root):
        for d in dirs:
            if d.lower() == "rgb":
                rgb_path = Path(root) / d
                if rgb_path.parent.name.lower().startswith("muestra_"):
                    rgb_folders.append(rgb_path)
    rgb_folders.sort()
    return rgb_folders

def main():
    base = Path(BASE_ROOT)
    if not base.exists():
        raise FileNotFoundError(base)

    rgb_folders = find_all_rgb_folders(base)
    print("Carpetas rgb encontradas:", len(rgb_folders))
    if not rgb_folders:
        return

    done = 0
    skipped = 0
    failed = 0

    for i, rgb_folder in enumerate(rgb_folders, 1):
        muestra_folder = rgb_folder.parent

        out_real = muestra_folder / "skeleton_data_real.txt"
        out_pix  = muestra_folder / "skeleton_data_pixel.txt"

        if SKIP_IF_EXISTS and out_real.exists() and out_pix.exists():
            skipped += 1
            continue

        print(f"\n[{i}/{len(rgb_folders)}] Procesando: {rgb_folder}")
        ok = extract_face_landmarks_from_rgb(rgb_folder, muestra_folder)
        if ok:
            done += 1
        else:
            failed += 1

    print("\n=== RESUMEN ===")
    print("Procesados OK:", done)
    print("Saltados:", skipped)
    print("Fallidos:", failed)

if __name__ == "__main__":
    main()
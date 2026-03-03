import os
import cv2
import lmdb
import pickle
import random
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import imageio
import torch

# ===================== CONFIGURACION =====================
ROOT      = r"D:\NUEVO\NUEVO"
LMDB_DIR  = r"D:\NUEVO\Dataset_Preprocesado"
os.makedirs(LMDB_DIR, exist_ok=True)

TRAIN_SUBJECTS = [1, 10, 11, 12, 14, 2, 3, 4, 6, 7]
VAL_SUBJECTS   = [16, 17, 13]
TEST_SUBJECTS  = [8, 9, 15]

AUG_DIRS_TRAIN = ["rgb", "rgb_FLIP", "rgb_GRAY", "rgb_ROTATION", "rgb_STRETCH"]

JPEG_QUALITY = 90
MAP_SIZE_GB  = 200

SEED = 42
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)
random.seed(SEED)


# ===================== UTILIDADES =====================
def imread_unicode(path):
    try:
        arr = np.fromfile(path, dtype=np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except Exception:
        return None


def jpeg_encode(img_bgr, quality=JPEG_QUALITY):
    ok, buf = cv2.imencode('.jpg', img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    return buf.tobytes() if ok else None


def decode_frames(frames_bytes):
    frames = []
    for b in frames_bytes:
        arr = np.frombuffer(b, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is not None:
            frames.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    return frames


def collect_samples(root, subjects, for_train=False):
    """
    Recorre el dataset en disco y devuelve lista de tuplas:
      (rgb_dir, class_name, subj_name, muestra, augment)

    Para train recoge todas las variantes de augmentation.
    Para val/test solo recoge 'rgb'.
    """
    samples = []
    for s in subjects:
        subj_name = f"SUJETO {s}"
        subj_path = os.path.join(root, subj_name, "señas_procesadas")
        if not os.path.isdir(subj_path):
            continue

        for class_name in os.listdir(subj_path):
            class_path = os.path.join(subj_path, class_name)
            if not os.path.isdir(class_path):
                continue

            for muestra in os.listdir(class_path):
                muestra_path = os.path.join(class_path, muestra)
                if not os.path.isdir(muestra_path):
                    continue

                aug_list = AUG_DIRS_TRAIN if for_train else ["rgb"]
                for aug in aug_list:
                    rgb_dir = os.path.join(muestra_path, aug)
                    has_png = any(f.lower().endswith(".png") for f in os.listdir(rgb_dir)) if os.path.isdir(rgb_dir) else False
                    if has_png:
                        samples.append((rgb_dir, class_name, subj_name, muestra, aug))
    return samples


# ===================== LMDB: ESCANEO =====================
def scan_lmdb(lmdb_path):
    """
    Devuelve el siguiente indice libre y el conjunto de tuplas
    (sujeto, clase, muestra, augment) ya existentes en el LMDB.
    """
    if not os.path.exists(lmdb_path):
        return 0, set()

    env = lmdb.open(lmdb_path, readonly=True, lock=False)
    existing = set()
    max_idx = -1

    with env.begin() as txn:
        with txn.cursor() as cur:
            for k, v in cur:
                try:
                    idx = int(k.decode())
                    if idx > max_idx:
                        max_idx = idx
                except Exception:
                    pass
                try:
                    data = pickle.loads(v)
                    existing.add((
                        data.get('sujeto', ''),
                        data.get('class', ''),
                        data.get('muestra', ''),
                        data.get('augment', '')
                    ))
                except Exception:
                    continue

    env.close()
    return max_idx + 1, existing


# ===================== LMDB: ESCRITURA =====================
def write_samples_to_lmdb(lmdb_path, samples, append=False):
    """
    Escribe samples en el LMDB indicado.
      append=False : crea o sobreescribe desde cero.
      append=True  : agrega solo samples nuevos, evita duplicados por tupla.
    """
    MAP_SIZE = MAP_SIZE_GB * 1024 ** 3
    env = lmdb.open(lmdb_path, map_size=MAP_SIZE)

    if append:
        next_idx, existing = scan_lmdb(lmdb_path)
        to_write = [s for s in samples if (s[2], s[1], s[3], s[4]) not in existing]
        print(f"  Candidatas: {len(samples)} | Nuevas: {len(to_write)}")
    else:
        next_idx = 0
        to_write = samples

    written = 0
    with env.begin(write=True) as txn:
        for rgb_dir, class_name, subj_name, muestra, augment in tqdm(to_write, desc=os.path.basename(lmdb_path)):
            files = sorted(f for f in os.listdir(rgb_dir) if f.lower().endswith(".png"))
            if not files:
                continue

            frames_bytes = []
            for f in files:
                img = imread_unicode(os.path.join(rgb_dir, f))
                if img is None:
                    continue
                enc = jpeg_encode(img)
                if enc is not None:
                    frames_bytes.append(enc)

            if not frames_bytes:
                continue

            data = {
                'frames':  frames_bytes,
                'class':   class_name,
                'sujeto':  subj_name,
                'muestra': muestra,
                'augment': augment
            }
            txn.put(str(next_idx).encode(), pickle.dumps(data))
            next_idx += 1
            written += 1

    env.close()
    print(f"  Guardado: {lmdb_path} ({written} entradas)")
    return written


# ===================== LMDB: VERIFICACION =====================
def verify_lmdb(lmdb_path):
    """
    Recorre el LMDB, valida estructura de cada entrada y
    reporta distribucion por clase y errores encontrados.
    """
    print(f"\nVerificando: {lmdb_path}")
    env = lmdb.open(lmdb_path, readonly=True, lock=False)
    total  = 0
    errors = {}
    dist   = defaultdict(int)

    with env.begin() as txn:
        for key, value in tqdm(txn.cursor(), desc="Leyendo"):
            try:
                entry = pickle.loads(value)
                if not isinstance(entry, dict):
                    raise TypeError("Entrada no es un diccionario")
                frames = entry.get("frames", [])
                if not isinstance(frames, list) or len(frames) == 0:
                    raise ValueError("Sin frames validos")
                dist[entry.get("class", "Desconocida")] += 1
                total += 1
            except Exception as e:
                try:
                    errors[int(key.decode())] = str(e)
                except Exception:
                    pass

    env.close()

    print(f"  Correctas : {total}")
    print(f"  Con error : {len(errors)}")
    print("  Distribucion por clase:")
    for c, n in sorted(dist.items()):
        print(f"    {c:<22} {n:4d}")

    if errors:
        print("  Errores (max 10):")
        for k, v in list(errors.items())[:10]:
            print(f"    idx {k}: {v}")

    return total, errors, dist


# ===================== LMDB: EJEMPLOS GIF =====================
def save_example_gifs(lmdb_path, output_dir, n=3, fps=10):
    """Selecciona n entradas aleatorias y guarda un GIF de preview por cada una."""
    env = lmdb.open(lmdb_path, readonly=True, lock=False)
    all_keys = []
    with env.begin() as txn:
        for key, _ in txn.cursor():
            all_keys.append(int(key.decode()))
    env.close()

    random.shuffle(all_keys)
    selected = all_keys[:n]

    env = lmdb.open(lmdb_path, readonly=True, lock=False)
    for k in selected:
        with env.begin() as txn:
            entry = pickle.loads(txn.get(str(k).encode()))
        frames = decode_frames(entry["frames"])
        if not frames:
            continue
        gif_path = os.path.join(output_dir, f"preview_{k}.gif")
        imageio.mimsave(gif_path, frames, fps=fps)
        print(f"  GIF: {gif_path} | clase={entry['class']} sujeto={entry['sujeto']} aug={entry['augment']} frames={len(frames)}")
    env.close()


# ===================== PIPELINE PRINCIPAL =====================
def build_dataset():
    """Construye los tres splits desde cero."""
    print("Recolectando samples...")
    train_samples = collect_samples(ROOT, TRAIN_SUBJECTS, for_train=True)
    val_samples   = collect_samples(ROOT, VAL_SUBJECTS,   for_train=False)
    test_samples  = collect_samples(ROOT, TEST_SUBJECTS,  for_train=False)
    print(f"  Train: {len(train_samples)} | Val: {len(val_samples)} | Test: {len(test_samples)}")

    print("\nEscribiendo LMDBs...")
    write_samples_to_lmdb(os.path.join(LMDB_DIR, "train.lmdb"), train_samples, append=False)
    write_samples_to_lmdb(os.path.join(LMDB_DIR, "val.lmdb"),   val_samples,   append=False)
    write_samples_to_lmdb(os.path.join(LMDB_DIR, "test.lmdb"),  test_samples,  append=False)


def append_subjects(subjects):
    """Agrega sujetos nuevos al train.lmdb sin reconstruirlo. Solo inserta muestras nuevas."""
    print(f"Agregando sujetos {subjects} a train.lmdb...")
    new_samples = collect_samples(ROOT, subjects, for_train=True)
    write_samples_to_lmdb(os.path.join(LMDB_DIR, "train.lmdb"), new_samples, append=True)


def verify_all(show_examples=True, n_examples=3):
    """Verifica integridad de los tres splits y genera GIFs de preview."""
    for split in ["train", "val", "test"]:
        path = os.path.join(LMDB_DIR, f"{split}.lmdb")
        if os.path.exists(path):
            verify_lmdb(path)
        else:
            print(f"\n{split}.lmdb no encontrado, omitiendo.")

    if show_examples:
        print("\nGenerando GIFs de ejemplo desde train.lmdb...")
        train_path = os.path.join(LMDB_DIR, "train.lmdb")
        if os.path.exists(train_path):
            save_example_gifs(train_path, LMDB_DIR, n=n_examples)


# ===================== PUNTO DE ENTRADA =====================
if __name__ == "__main__":
    # Paso 1: construir dataset desde cero
    build_dataset()

    # Paso 2 (opcional): agregar sujetos nuevos sin reconstruir
    # append_subjects([1])

    # Paso 3: verificar integridad y generar GIFs de preview
    verify_all(show_examples=True, n_examples=3)
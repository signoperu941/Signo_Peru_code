import os
import cv2
import numpy as np
import mediapipe as mp
from collections import deque

# Configuración
base_path = r'C:\Users\20214\Documents\prueba'
target_height = 640
target_width = 640

# Inicializar detectores
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(
    model_selection=1,
    min_detection_confidence=0.3
)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

class FaceTracker:
    def __init__(self, window_size=10):
        self.positions = deque(maxlen=window_size)
        self.last_valid_bbox = None
        self.frames_without_detection = 0
        self.max_frames_interpolate = 15
    
    def update(self, bbox):
        if bbox is not None:
            self.positions.append(bbox)
            self.last_valid_bbox = bbox
            self.frames_without_detection = 0
        else:
            self.frames_without_detection += 1
            if self.last_valid_bbox is not None and self.frames_without_detection <= self.max_frames_interpolate:
                return tuple(self.last_valid_bbox)
        
        if len(self.positions) == 0:
            return None
        
        avg_bbox = np.mean(list(self.positions), axis=0).astype(int)
        return tuple(avg_bbox)

def detect_face_mediapipe(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_detection.process(image_rgb)
    
    if results.detections:
        best_detection = max(results.detections, key=lambda d: d.score[0])
        bboxC = best_detection.location_data.relative_bounding_box
        h, w, _ = image.shape
        
        x1 = int(bboxC.xmin * w)
        y1 = int(bboxC.ymin * h)
        x2 = int((bboxC.xmin + bboxC.width) * w)
        y2 = int((bboxC.ymin + bboxC.height) * h)
        
        return [x1, y1, x2, y2]
    return None

def detect_face_haar(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30))
    
    if len(faces) > 0:
        largest_face = max(faces, key=lambda f: f[2] * f[3])
        x, y, w, h = largest_face
        return [x, y, x + w, y + h]
    return None

def detect_face_hybrid(image):
    bbox = detect_face_mediapipe(image)
    if bbox is None:
        bbox = detect_face_haar(image)
    return bbox

def process_image(input_path, tracker):
    try:
        img_data = np.fromfile(input_path, dtype=np.uint8)
        image = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
        if image is None:
            return False
        
        detected_face = detect_face_hybrid(image)
        smoothed_face = tracker.update(detected_face)
        
        if smoothed_face is None:
            return False
        
        x1, y1, x2, y2 = smoothed_face
        face_width = x2 - x1
        face_height = y2 - y1
        
        if face_width < 30 or face_height < 30:
            return False
        
        expansion_factor = 0.40
        expand_x = int(face_width * expansion_factor)
        expand_y = int(face_height * expansion_factor)
        
        new_x1 = max(0, x1 - expand_x)
        new_x2 = min(image.shape[1], x2 + expand_x)
        new_y1 = max(0, y1 - expand_y)
        new_y2 = min(image.shape[0], y2 + expand_y)
        
        crop_width = new_x2 - new_x1
        crop_height = new_y2 - new_y1
        
        if crop_width > crop_height:
            diff = crop_width - crop_height
            new_y1 = max(0, new_y1 - diff // 2)
            new_y2 = min(image.shape[0], new_y2 + (diff - diff // 2))
        elif crop_height > crop_width:
            diff = crop_height - crop_width
            new_x1 = max(0, new_x1 - diff // 2)
            new_x2 = min(image.shape[1], new_x2 + (diff - diff // 2))
        
        cropped = image[new_y1:new_y2, new_x1:new_x2]
        if cropped.size == 0 or cropped.shape[0] < 50 or cropped.shape[1] < 50:
            return False
        
        resized = cv2.resize(cropped, (target_width, target_height), interpolation=cv2.INTER_LANCZOS4)
        
        # Sobreescribir la imagen en el mismo path
        success = cv2.imwrite(input_path, resized)
        if not success:
            _, buffer = cv2.imencode('.png', resized)
            with open(input_path, 'wb') as f:
                f.write(buffer)
        
        return os.path.exists(input_path)
    except:
        return False

# Procesar dataset completo
processed_count = 0
total_count = 0
failed_count = 0

print("Procesando dataset completo...")
print(f"Entrada/Salida: {base_path}")

for sujeto in sorted(os.listdir(base_path)):
    sujeto_path = os.path.join(base_path, sujeto)
    
    if os.path.isdir(sujeto_path):
        print(f"\n{sujeto}")
        
        for carpeta_intermedia in sorted(os.listdir(sujeto_path)):
            carpeta_intermedia_path = os.path.join(sujeto_path, carpeta_intermedia)
            
            if os.path.isdir(carpeta_intermedia_path):
                
                for expresion in sorted(os.listdir(carpeta_intermedia_path)):
                    expresion_path = os.path.join(carpeta_intermedia_path, expresion)
                    
                    if os.path.isdir(expresion_path):
                        
                        for muestra in sorted(os.listdir(expresion_path)):
                            muestra_path = os.path.join(expresion_path, muestra)
                            
                            if os.path.isdir(muestra_path):
                                rgb_path = os.path.join(muestra_path, 'rgb')
                                
                                if os.path.exists(rgb_path):
                                    tracker = FaceTracker(window_size=10)
                                    
                                    image_files = sorted([f for f in os.listdir(rgb_path) 
                                                         if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
                                    
                                    muestra_processed = 0
                                    muestra_total = len(image_files)
                                    
                                    for imagen in image_files:
                                        input_img = os.path.join(rgb_path, imagen)
                                        
                                        total_count += 1
                                        if process_image(input_img, tracker):
                                            processed_count += 1
                                            muestra_processed += 1
                                        else:
                                            failed_count += 1
                                    
                                    if muestra_processed == muestra_total:
                                        print(f"  {expresion}/{muestra}: {muestra_processed}/{muestra_total}")
                                    else:
                                        print(f"  {expresion}/{muestra}: {muestra_processed}/{muestra_total} ({muestra_total - muestra_processed} fallos)")

print(f"\nResumen:")
print(f"Total: {total_count}")
print(f"Procesadas: {processed_count}")
print(f"Fallaron: {failed_count}")
print(f"Tasa de exito: {processed_count/total_count*100:.1f}%")
# SIGNO PERU - Peruvian Sign Language Recognition

Research project for Peruvian Sign Language recognition using deep learning and computer vision techniques. This repository contains the complete pipeline from data collection to model training.

## 📋 Overview

This project provides a comprehensive framework for creating and processing a Peruvian Sign Language dataset, including:
- Data collection using Microsoft Kinect Azure sensor
- Manual labeling of 14 expression categories
- Face detection using hybrid MediaPipe + Haar Cascade approach
- Facial landmark extraction (468 points) using MediaPipe Face Mesh
- Data augmentation techniques
- LMDB serialization for efficient data access
- Training implementations for three neural network architectures

## 🚀 Installation

### Using Anaconda (Recommended)
```bash
# Clone the repository
git clone https://github.com/signoperu941/Signo_Peru_code.git
cd Signo_Peru_code

# Create conda environment
conda create -n signo_peru python=3.8
conda activate signo_peru

# Install PyTorch (adjust CUDA version as needed)
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Install remaining dependencies
pip install -r requirements.txt
```

### Using pip (Alternative)
```bash
# Clone the repository
git clone https://github.com/signoperu941/Signo_Peru_code.git
cd Signo_Peru_code

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Verify Installation
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

## 📁 Project Structure
```
SIGNO_PERU/
├── 0.APP_GRABAR_DATASET.py                       # Kinect data capture
├── 1.ETIQUETADO.py                               # Manual labeling tool
├── 2.RECORTE.py                                  # Face detection & cropping
├── 3.LANDMARKS.PY                                # Facial landmark extraction
├── 4.AUMENTACION_DE_DATOS.py                     # Data augmentation
├── 5.CONVERSION_LMDB.py                          # LMDB serialization
├── ARQUITECTURAS/                                # Model training implementations
│   ├── 2+1D.ipynb                                # R(2+1)D model training
│   ├── 2+1D TwoStream.ipynb                      # R(2+1)D Two-Stream model
│   └── S2D_GCN_Two stream.ipynb                  # Spatial-Temporal GCN Two-Stream
├── README.md                                     # Project documentation
└── requirements.txt                              # Python dependencies
```

## 💻 Usage

### Step 1: Data Collection (Requires Kinect Azure)
```bash
python "0.APP_GRABAR_DATASET_KINECT_AZURE_(1)_V2.py"
```
**Description:** A GUI tool built with Tkinter to record synchronized RGB and Depth video sequences frame-by-frame using the Azure Kinect sensor.

### Step 2: Data Labeling
```bash
python 1.ETIQUETADO.py
```
**Description:** A GUI application to manually segment and label the recorded signs. Users mark the start and end frames of a sign, and the script automatically extracts and saves the corresponding RGB and Depth frames into structured señas_procesadas folders. Progress is saved via a JSON file to allow pausing/resuming.

### Step 3-6: Preprocessing Pipeline
Run the scripts in sequential order:
```bash
python 2.RECORTE.py
```
**Description:** Automates facial detection and cropping to remove background noise. It uses a temporal tracker to ensure smooth bounding boxes, expands the area by 40% to capture head/neck movements, and resizes frames uniformly to 640x640 pixels.

```bash
python 3.LANDMARKS.PY
```
**Description:** Extracts 468 facial landmarks from the cropped frames using MediaPipe Face Mesh. It handles failed detections with zero-padding to maintain temporal consistency and exports the sequences as 2D pixel (skeleton_data_pixel.txt) and 3D normalized (skeleton_data_real.txt) coordinates.

```bash
python 4.AUMENTACION_DE_DATOS.py
```
**Description:** Performs on-disk data augmentation (Flip, Rotation, Grayscale, Stretch) to prevent overfitting. Transformation parameters are fixed per video sequence to preserve natural movement dynamics.

```bash
python 5.CONVERSION_LMDB.py
```
**Description:** (Optional) Converts the processed .png frames into LMDB databases (Train, Val, Test). This maps data directly into memory, eliminating disk I/O bottlenecks and drastically speeding up the model's training time.

### Step 7: Model Training
Navigate to the `ARQUITECTURAS/` folder and select the desired architecture:

**Available architectures:**
- `2+1D.ipynb` - R(2+1)D Convolutional Network
- `2+1D TwoStream.ipynb` - R(2+1)D Two-Stream Network
- `S2D_GCN_Two stream.ipynb` - Spatial-Temporal Graph Convolutional Network (Two-Stream)
```bash
jupyter notebook ARQUITECTURAS/
# Open and run the desired architecture notebook
```
**Important Note on Custom Landmarks:** If you plan to conduct experiments that modify the extracted facial landmarks (e.g., isolating specific points like eyebrows or changing the total number of tracked coordinates in `3.LANDMARKS.PY`), you will need to adapt the training notebooks accordingly. Specifically, the data loading sections and the input layers of the landmark branches (such as in the Spatial-Temporal GCN Two-Stream architecture) must be updated to match the new topological dimensions of your custom data.

## 🏗️ Model Architectures

This repository implements three deep learning architectures for sign language recognition:

### 1. R(2+1)D Network
Standard R(2+1)D architecture for spatiotemporal feature learning from video sequences.

### 2. R(2+1)D Two-Stream Network
Dual-pathway architecture processing RGB and motion information independently before fusion.

### 3. Spatial-Temporal GCN Two-Stream
Graph Convolutional Network leveraging facial landmark topology with two-stream fusion for enhanced performance.

## 🛠️ Technologies

- **Hardware:** Microsoft Kinect Azure
- **Deep Learning:** PyTorch, TorchVision, Timm
- **Computer Vision:** OpenCV, MediaPipe
- **Data Processing:** NumPy, Pandas, LMDB
- **Visualization:** Matplotlib, Seaborn

## 📊 Dataset

The dataset consists of **14 Peruvian Sign Language expression categories** with:
- RGB video capture from Kinect Azure
- Hybrid face detection (MediaPipe + Haar Cascade fallback)
- 468 facial landmarks per frame
- Data augmentation for robustness
- LMDB format for efficient training

## 🔬 Research

This project is part of academic research on Peruvian Sign Language recognition. The complete methodology and experimental results will be available in an upcoming publication.

## 📦 Requirements

See `requirements.txt` for full dependencies. Main requirements:
- Python 3.8+
- PyTorch
- OpenCV
- MediaPipe
- Kinect Azure SDK (for data collection only)


## ✉️ Contact

For questions or collaboration inquiries, please contact:
eescobed@ulima.edu.pe

## 🙏 Acknowledgments

We thank the native LSP signers who volunteered their time to record the dataset and the assistants who supported the data collection sessions, and we also acknowledge the Artificial Intelligence Laboratory (IALAB) for providing its facilities for the recording sessions.

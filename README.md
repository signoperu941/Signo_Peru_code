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
├── 0.APP_GRABAR_DATASET_KINECT_AZURE_(1)_V2.py  # Kinect data capture
├── 1.ETIQUETADO.py                               # Manual labeling tool
├── 2.RECORTE.ipynb                               # Face detection & cropping
├── 3.AUMENTACION_DE_DATOS.ipynb                  # Data augmentation
├── 4.AUMENTACION.ipynb                           # Additional augmentation
├── 5.CONVERSION_LMDB.ipynb                       # LMDB serialization
├── 6.LANDMARKS.ipynb                             # Facial landmark extraction
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
**Note:** Microsoft Kinect Azure sensor is required for this step.

### Step 2: Data Labeling
```bash
python 1.ETIQUETADO.py
```

### Step 3-6: Preprocessing Pipeline
Run the Jupyter notebooks in sequential order:
```bash
jupyter notebook
# Execute in order:
# 1. 2.RECORTE.ipynb
# 2. 3.AUMENTACION_DE_DATOS.ipynb
# 3. 4.AUMENTACION.ipynb
# 4. 5.CONVERSION_LMDB.ipynb
# 5. 6.LANDMARKS.ipynb
```

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

### Citation
```
[Paper citation will be added upon publication]
```

## 📦 Requirements

See `requirements.txt` for full dependencies. Main requirements:
- Python 3.8+
- PyTorch
- OpenCV
- MediaPipe
- Kinect Azure SDK (for data collection only)

## 📝 License

[Specify your license]

## ✉️ Contact

For questions or collaboration inquiries, please contact:
[Your contact information]

## 🙏 Acknowledgments

This research was conducted as part of [Institution/Project name].
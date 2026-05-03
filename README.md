# 🏏 Cricket Shot Detection System  
### Batter Detection • Pose Extraction • Shot Classification • Shot Segment Export • GUI

This project is a complete end-to-end cricket analytics system that:

1. Detects the **batter** in cricket videos using a custom YOLOv8 model  
2. Extracts **pose landmarks (33 joints)** using MediaPipe Pose  
3. Classifies shots into **Cover Drive, Pull Shot, or Other** using an LSTM model  
4. Automatically extracts only the selected shots from a **full match video**  
5. Provides a **Graphical User Interface** for non-technical users  
6. Demonstrates dataset creation, annotation, pose modeling, and ML deployment  

---

# 🚀 Features

- Custom-trained **YOLOv8 batter detector**
- Smooth crop stabilization with EMA
- Pose extraction using **MediaPipe Pose**
- Pose-based LSTM shot classifier
- Full match shot extraction (MP4 output)
- Shot classification:
  - `cover_drive`
  - `pull_shot`
  - `other`
- Tkinter GUI for easy use
- Debugging, visualization, and dataset auditing tools

---

# 📂 Project Structure

```
cricket_shot_detection/
│
├── data/
│   ├── raw/                  
│   ├── batter_label_images/  
│   ├── batter_detector/      
│   ├── crops/                
│   ├── pose/                 
│   └── segments/             
│
├── models/
│   ├── batter_detector.pt    
│   └── shot_lstm.pt          
│
├── scripts/
│   ├── extract_pose_batter.py
│   ├── train_shot_lstm.py
│   ├── detect_shots.py
│   ├── ui_detect_shots.py
│   ├── draw_skeleton_video.py
│   ├── viz_pose_samples.py
│   ├── audit_pose_dataset.py
│   ├── utils.py
│   └── build_batter_dataset.py
│
├── data/batter.yaml
└── src/config.yaml
```

---

# 🛠 Installation

### 1️⃣ Create Virtual Environment

**Windows**
```cmd
python -m venv .venv
.venv\Scripts\activate
```

**Linux/Mac**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2️⃣ Install Dependencies

```cmd
pip install -r requirements.txt
```

If YOLO is missing:
```cmd
pip install ultralytics
```

---

# 📸 1. Dataset Preparation

## Step 1 — Extract Frames (Optional)
```cmd
python scripts/extract_frames.py video.mp4 --out data/batter_label_images
```
all images

Get-ChildItem data/raw/train -Recurse -Include *.mp4 | ForEach-Object {
    python scripts/extract_frames.py $_.FullName --out data/batter_label_images/$($_.BaseName)
}


## Step 2 — Label the Batter Using LabelImg

Install LabelImg:
```cmd
pip install labelImg
```

Run:
```cmd
labelImg data/batter_label_images
```

Label only the batter using class name:
```
batter
```

---

# 🧠 2. Train Batter Detector (YOLOv8)

### Build YOLO Dataset
```cmd
python scripts/build_batter_dataset.py --src data/batter_label_images --dst data/batter_detector
```

### Train YOLO
```cmd
yolo detect train model=yolov8n.pt data=data/batter.yaml epochs=50 imgsz=640
```

Output:
```
models/batter_detector.pt
```

---

# 🏋️ 3. Extract Pose From Training Clips

Extract for all:
```cmd
Get-ChildItem -Recurse data/raw/train -Include *.mp4 | ForEach-Object {
    python -m scripts.extract_pose_batter $_.FullName --det_model models/batter_detector.pt
}
```

Outputs:
```
data/crops/<video>_batter.mp4
data/pose/<video>_pose.npy
```
👉 “For each training video, run extract_pose_batter.py using our custom-trained batter detector YOLO model.”
---

# 🔥 4. Train Shot Classifier (LSTM)

```cmd
python -m scripts.train_shot_lstm --pose_root data/pose/train --epochs 80
```

Produces:
```
models/shot_lstm.pt
```

---

# 🎯 5. Full Match Shot Detection & Export

### Detect ALL shots
```cmd
python -m scripts.detect_shots full_match.mp4 --cls all
```

### Only pull shots
```cmd
python -m scripts.detect_shots full_match.mp4 --cls pull_shot
```

### Only cover drives
```cmd
python -m scripts.detect_shots full_match.mp4 --cls cover_drive
```

Saved to:
```
data/segments/
```

---

# 🖥 6. Graphical UI (User-Friendly)

Run GUI:
```cmd
python scripts/ui_detect_shots.py
```

Allows user to:

- Choose a match video  
- Choose shot type  
- Choose output folder  
- Click “Process”  

---

# 🧪 Debugging & Visualization Tools

### 1️⃣ Draw Skeleton on Crop
```cmd
python scripts/draw_skeleton_video.py video.mp4 --pose data/pose/video_pose.npy --out debug.mp4
```

### 2️⃣ Visualize Pose Samples
```cmd
python scripts/viz_pose_samples.py --root data/pose/train --per_class 4
```

### 3️⃣ Audit Pose Dataset
```cmd
python scripts/audit_pose_dataset.py data/pose/train
```

---

# 🧠 Technologies Used

- **YOLOv8 (Ultralytics)** — batter detection  
- **OpenCV** — video processing  
- **MediaPipe Pose** — pose estimation (33 joints)  
- **NumPy** — data preprocessing  
- **PyTorch (LSTM)** — shot classification  
- **MoviePy** — MP4 exporting  
- **Tkinter + ttkbootstrap** — GUI  
- **YAML** — system configuration  

---

# ✨ Future Work

- Better batter detector (more data)
- More shot classes (cut, sweep, lofted)
- 1D-CNN or Transformer shot classifier
- Ball and bat tracking
- Multi-camera support

---

# 📬 Support

If you need improvements:
- Accuracy optimization  
- Adding new shot types  
- GUI improvements  
- Model tuning  

Open an issue or contact the contributor.

## Visit to the GUI
[Open CricShotVid GUI](https://github.com/DahamAdikari/CricShotVidGUI)

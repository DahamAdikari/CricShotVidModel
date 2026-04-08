import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk

import cv2
import numpy as np
import torch
import torch.nn as nn
from ultralytics import YOLO
import mediapipe as mp

# ---------------------------
# LSTM model definition (must match train_shot_lstm.py)
# ---------------------------
CLASSES = ["cover_drive", "pull_shot", "other"]
POSE_LANDMARKS = 33
T_DEFAULT = 24

class LSTMShot(nn.Module):
    def __init__(self, in_dim=3*33, hidden=128, num_classes=3):
        super().__init__()
        self.lstm = nn.LSTM(in_dim, hidden, num_layers=1, batch_first=True)
        self.fc = nn.Linear(hidden, num_classes)

    def forward(self, x):
        # x: (B,T,33,3)
        B,T,N,D = x.shape
        x = x.view(B, T, N*D)       # (B,T,99)
        out,_ = self.lstm(x)        # (B,T,H)
        logits = self.fc(out[:, -1, :])  # last timestep
        return logits

# ---------------------------
# Pose helpers
# ---------------------------
LEFT  = [11,13,15,23,25,27,29,31]
RIGHT = [12,14,16,24,26,28,30,32]

def mirror_sequence(seq):
    """Mirror horizontally + swap left/right joints. seq: (T,33,3)"""
    mir = seq.copy()
    mir[:,:,0] = 1.0 - mir[:,:,0]
    for l, r in zip(LEFT, RIGHT):
        mir[:,[l,r],:] = mir[:,[r,l],:]
    return mir

def norm_frame(frame_33x3):
    """
    Normalize per frame:
    - center at mid-hip
    - scale by shoulder width
    Keep visibility as third channel.
    """
    midhip = (frame_33x3[23,:2] + frame_33x3[24,:2]) / 2
    shoulder = np.linalg.norm(frame_33x3[11,:2] - frame_33x3[12,:2]) + 1e-6
    xy = (frame_33x3[:,:2] - midhip) / shoulder
    vis = frame_33x3[:,2:3]
    return np.concatenate([xy, vis], axis=1).astype(np.float32)  # (33,3)

def softmax_np(x):
    x = x - np.max(x)
    e = np.exp(x)
    return e / np.sum(e)

# ---------------------------
# GUI App
# ---------------------------
class LiveShotPreviewGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Live Shot + Pose Preview (YOLO + MediaPipe + LSTM)")

        # ---- Models ----
        self.det = YOLO("models/batter_detector.pt")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = torch.load("models/shot_lstm.pt", map_location=self.device)
        self.T = ckpt.get("T", T_DEFAULT)

        self.net = LSTMShot().to(self.device)
        self.net.load_state_dict(ckpt["state_dict"])
        self.net.eval()

        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(static_image_mode=False, model_complexity=1, enable_segmentation=False)
        self.drawer = mp.solutions.drawing_utils

        # ---- State ----
        self.cap = None
        self.playing = False
        self.frame_index = 0
        self.pose_buffer = []  # list of (33,3)
        self.last_pred = ("-", 0.0)

        # ---- Controls / Settings ----
        self.conf = 0.35
        self.iou = 0.5
        self.skip = 2  # process every Nth frame (speed). change to 1 for max accuracy.

        # ---- UI ----
        top = tk.Frame(root)
        top.pack(fill="x", padx=10, pady=8)

        tk.Button(top, text="Open Video", command=self.open_video).pack(side="left")
        tk.Button(top, text="Play", command=self.play).pack(side="left", padx=6)
        tk.Button(top, text="Pause", command=self.pause).pack(side="left")

        # Two preview panels
        panel = tk.Frame(root)
        panel.pack(padx=10, pady=8)

        self.lbl_full = tk.Label(panel, text="Full Frame Preview")
        self.lbl_full.grid(row=0, column=0, padx=6)

        self.lbl_crop = tk.Label(panel, text="Batter Crop + Skeleton Preview")
        self.lbl_crop.grid(row=0, column=1, padx=6)

        # Status text
        self.status = tk.Label(root, text="Open a video to start.", anchor="w")
        self.status.pack(fill="x", padx=10, pady=(0,10))

    def open_video(self):
        path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4;*.avi;*.mov")])
        if not path:
            return
        self.cap = cv2.VideoCapture(path)
        self.playing = False
        self.frame_index = 0
        self.pose_buffer = []
        self.last_pred = ("-", 0.0)
        self.status.config(text=f"Loaded: {path}")

    def play(self):
        if self.cap is None:
            self.status.config(text="No video loaded.")
            return
        self.playing = True
        self.update()

    def pause(self):
        self.playing = False

    def update(self):
        if not self.playing or self.cap is None:
            return

        ret, frame = self.cap.read()
        if not ret:
            self.playing = False
            self.status.config(text="Video ended.")
            return

        self.frame_index += 1

        # Speed control: skip frames
        if self.frame_index % self.skip != 0:
            self.show_full_frame(frame, None, None)
            self.root.after(10, self.update)
            return

        H, W = frame.shape[:2]

        # ---------- YOLO detect batter ----------
        res = self.det.predict(frame, conf=self.conf, iou=self.iou, verbose=False)
        bbox = None
        conf_val = None

        if res and res[0].boxes is not None and len(res[0].boxes) > 0:
            xyxy = res[0].boxes.xyxy.cpu().numpy()
            confs = res[0].boxes.conf.cpu().numpy() if res[0].boxes.conf is not None else np.ones(len(xyxy))
            idx = int(np.argmax(confs))
            x1, y1, x2, y2 = map(int, xyxy[idx])
            bbox = (max(0,x1), max(0,y1), min(W-1,x2), min(H-1,y2))
            conf_val = float(confs[idx])

        # If no batter -> predict OTHER & show full only
        if bbox is None:
            self.pose_buffer = []
            self.last_pred = ("other", 0.0)
            self.show_full_frame(frame, None, "No batter detected")
            self.show_crop(None, label="other (no batter)")
            self.root.after(10, self.update)
            return

        # ---------- Crop batter ----------
        x1, y1, x2, y2 = bbox
        crop = frame[y1:y2, x1:x2].copy()
        if crop.size == 0:
            self.root.after(10, self.update)
            return

        # ---------- MediaPipe pose on crop ----------
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        pres = self.pose.process(crop_rgb)

        lm = np.zeros((POSE_LANDMARKS, 3), dtype=np.float32)
        if pres.pose_landmarks:
            for i, p in enumerate(pres.pose_landmarks.landmark):
                if i < POSE_LANDMARKS:
                    lm[i,0] = p.x
                    lm[i,1] = p.y
                    lm[i,2] = p.visibility

        # draw skeleton for preview
        crop_draw = crop.copy()
        if pres.pose_landmarks:
            self.drawer.draw_landmarks(
                crop_draw, pres.pose_landmarks, self.mp_pose.POSE_CONNECTIONS
            )

        # ---------- Update pose buffer ----------
        self.pose_buffer.append(lm)
        if len(self.pose_buffer) > self.T:
            self.pose_buffer.pop(0)

        # ---------- Predict if we have T frames ----------
        pred_text = None
        if len(self.pose_buffer) == self.T:
            seq = np.stack(self.pose_buffer, axis=0)  # (T,33,3)

            # normalize per-frame
            seq_n = np.stack([norm_frame(f) for f in seq], axis=0)  # (T,33,3)

            # mirrored version (handedness invariance)
            seq_m = mirror_sequence(seq)
            seq_mn = np.stack([norm_frame(f) for f in seq_m], axis=0)

            # run model on both
            with torch.no_grad():
                a = self.net(torch.tensor(seq_n[None], dtype=torch.float32, device=self.device)).cpu().numpy().squeeze()
                b = self.net(torch.tensor(seq_mn[None], dtype=torch.float32, device=self.device)).cpu().numpy().squeeze()
                pA = softmax_np(a)
                pB = softmax_np(b)
                p = np.maximum(pA, pB)

            top = int(np.argmax(p))
            label = CLASSES[top]
            prob = float(p[top])
            self.last_pred = (label, prob)
            pred_text = f"{label} ({prob:.2f})"
        else:
            pred_text = f"buffering... ({len(self.pose_buffer)}/{self.T})"

        # ---------- Show UI ----------
        self.show_full_frame(frame, bbox, f"det_conf={conf_val:.2f} | pred={pred_text}")
        self.show_crop(crop_draw, label=pred_text)

        self.root.after(10, self.update)

    def show_full_frame(self, frame_bgr, bbox, text):
        frame = frame_bgr.copy()
        if bbox is not None:
            x1,y1,x2,y2 = bbox
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
        if text:
            cv2.putText(frame, text, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        img.thumbnail((640, 360))
        imgtk = ImageTk.PhotoImage(img)
        self.lbl_full.config(image=imgtk)
        self.lbl_full.image = imgtk

    def show_crop(self, crop_bgr, label=""):
        if crop_bgr is None:
            blank = np.zeros((360, 640, 3), dtype=np.uint8)
            cv2.putText(blank, "No crop", (20,60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2)
            crop_bgr = blank

        crop = crop_bgr.copy()
        if label:
            cv2.putText(crop, label, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(crop_rgb)
        img.thumbnail((640, 360))
        imgtk = ImageTk.PhotoImage(img)
        self.lbl_crop.config(image=imgtk)
        self.lbl_crop.image = imgtk


if __name__ == "__main__":
    root = tk.Tk()
    app = LiveShotPreviewGUI(root)
    root.mainloop()

# scripts/extract_pose_batter.py
# Detect batter per frame (YOLO), smooth bbox, crop, run MediaPipe Pose, save crop MP4 + pose .npy

import cv2, argparse, numpy as np
from pathlib import Path
from ultralytics import YOLO
import mediapipe as mp

from scripts.utils import Cfg, ensure_dir, crop_box_from_bbox

mp_pose = mp.solutions.pose
POSE_LANDMARKS = 33

def downsample_reader(path, fps_target):
    cap = cv2.VideoCapture(str(path))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    step = max(1, int(round(src_fps / fps_target)))
    return cap, int(src_fps), step

def clip_box(x1, y1, x2, y2, W, H):
    x1 = max(0, min(int(x1), W-1))
    y1 = max(0, min(int(y1), H-1))
    x2 = max(0, min(int(x2), W-1))
    y2 = max(0, min(int(y2), H-1))
    return x1, y1, x2, y2

def ema(prev, curr, alpha=0.6):
    if prev is None:
        return np.array(curr, dtype=np.float32)
    return (1 - alpha) * prev + alpha * np.array(curr, dtype=np.float32)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('video', help='input video (.mp4)')
    ap.add_argument('--det_model', default='models/batter_detector.pt', help='YOLO batter detector path')
    ap.add_argument('--pose_out', default='data/pose', help='folder to write pose .npy')
    ap.add_argument('--crop_out', default='data/crops', help='folder to write cropped batter .mp4')
    ap.add_argument('--config', default='src/config.yaml', help='config (fps_target, crop_size, thresholds)')
    ap.add_argument('--conf', type=float, default=0.35, help='YOLO confidence threshold')
    ap.add_argument('--iou', type=float, default=0.5, help='YOLO NMS IoU')
    ap.add_argument('--smooth', type=float, default=0.6, help='EMA smoothing factor for bbox (0..1)')
    ap.add_argument('--draw', action='store_true', help='draw pose on the crop video')
    args = ap.parse_args()

    cfg = Cfg(args.config)
    ensure_dir(args.pose_out); ensure_dir(args.crop_out)

    # IO
    video = Path(args.video)
    crop_path = Path(args.crop_out) / f"{video.stem}_batter.mp4"
    pose_path = Path(args.pose_out) / f"{video.stem}_pose.npy"

    # Open video + prepare downsampling
    cap, src_fps, step = downsample_reader(video, cfg.fps_target)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    crop_w, crop_h = cfg.crop_size
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(crop_path), fourcc, cfg.fps_target, (crop_w, crop_h))

    # Models
    det = YOLO(args.det_model)
    pose = mp_pose.Pose(static_image_mode=False, model_complexity=1, enable_segmentation=False)

    # State
    keypoints = []
    ema_box = None             # smoothed [x1,y1,x2,y2]
    miss_streak = 0
    MAX_MISS = 8               # tolerate short gaps

    frame_idx = -1
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1
        if frame_idx % step != 0:
            continue

        # 1) DETECT BATTER IN THIS FRAME
        # Ultralytics returns a list of Results; we take first batch element
        res = det.predict(frame, conf=args.conf, iou=args.iou, verbose=False)
        boxes = []
        if len(res) > 0 and res[0].boxes is not None:
            xyxy = res[0].boxes.xyxy.cpu().numpy()
            confs = res[0].boxes.conf.cpu().numpy() if res[0].boxes.conf is not None else np.ones(len(xyxy))
            # sort by confidence desc, pick top-1
            if len(xyxy) > 0:
                idx = int(np.argmax(confs))
                boxes = [xyxy[idx]]

        if boxes:
            x1,y1,x2,y2 = boxes[0]
            x1,y1,x2,y2 = clip_box(x1,y1,x2,y2,W,H)
            ema_box = ema(ema_box, (x1,y1,x2,y2), alpha=args.smooth)
            miss_streak = 0
        else:
            miss_streak += 1
            # hold last box for a few frames to avoid flicker
            if miss_streak > MAX_MISS and ema_box is None:
                # nothing to crop; append empty pose and continue
                keypoints.append(np.zeros((POSE_LANDMARKS,3), dtype=np.float32))
                # write a blank frame to keep lengths consistent
                blank = np.zeros((crop_h, crop_w, 3), dtype=np.uint8)
                writer.write(blank)
                continue

        # 2) CROP USING SMOOTHED BOX (or last known)
        if ema_box is None:
            # no detection yet: write blank (keeps timing consistent)
            blank = np.zeros((crop_h, crop_w, 3), dtype=np.uint8)
            keypoints.append(np.zeros((POSE_LANDMARKS,3), dtype=np.float32))
            writer.write(blank)
            continue

        bx1,by1,bx2,by2 = ema_box.tolist()
        left, top, right, bottom = crop_box_from_bbox(W,H,(bx1,by1,bx2,by2),crop_w,crop_h)
        crop = frame[top:bottom, left:right]
        if crop.shape[0] != crop_h or crop.shape[1] != crop_w:
            crop = cv2.resize(crop, (crop_w, crop_h), interpolation=cv2.INTER_LINEAR)

        # 3) POSE
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        pres = pose.process(rgb)
        lm = np.zeros((POSE_LANDMARKS,3), dtype=np.float32)
        if pres.pose_landmarks:
            for i, p in enumerate(pres.pose_landmarks.landmark):
                if i < POSE_LANDMARKS:
                    lm[i,0] = p.x; lm[i,1] = p.y; lm[i,2] = p.visibility
        keypoints.append(lm)

        # 4) OPTIONAL DRAW
        if args.draw and pres.pose_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(
                crop, pres.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        writer.write(crop)

    # finalize
    writer.release(); cap.release()
    kps = np.stack(keypoints) if keypoints else np.zeros((0,POSE_LANDMARKS,3), np.float32)
    np.save(pose_path, kps)
    print(f"[OK] Saved crop: {crop_path}")
    print(f"[OK] Saved pose: {pose_path}  frames={len(kps)}")

if __name__ == '__main__':
    main()

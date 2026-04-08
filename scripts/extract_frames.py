# scripts/extract_frames.py
# Extract frames from a video to prepare dataset for LabelImg or inspection

import cv2
import argparse
from pathlib import Path

def extract_frames(video_path, out_dir, fps=5):
    video_path = Path(video_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {video_path}")
        return
    
    src_fps = cap.get(cv2.CAP_PROP_FPS)
    step = max(1, int(round(src_fps / fps)))
    
    frame_idx = 0
    saved_idx = 0

    print(f"[INFO] Extracting frames from: {video_path.name}")
    print(f"[INFO] Source FPS: {src_fps}  |  Target FPS: {fps} | Step: {step}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_idx % step == 0:
            saved_idx += 1
            save_path = out_dir / f"frame_{saved_idx:04d}.jpg"
            cv2.imwrite(str(save_path), frame)

        frame_idx += 1

    cap.release()
    print(f"[OK] Extracted {saved_idx} frames → {out_dir}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video", help="Input video (.mp4)")
    ap.add_argument("--out", default="frames", help="Output folder")
    ap.add_argument("--fps", type=int, default=5, help="Frames per second to extract")
    args = ap.parse_args()

    extract_frames(args.video, args.out, args.fps)

if __name__ == "__main__":
    main()

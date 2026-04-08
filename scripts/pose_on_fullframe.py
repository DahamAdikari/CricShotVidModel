# scripts/pose_on_fullframe.py
# YOLO-guided pose on full frame + save pose .npy + optional debug video.
#
# Usage examples:
#   Live preview only:
#     python -m scripts.pose_on_fullframe data/raw/train/cover_drive/cd1.mp4
#
#   Save debug video + pose array:
#     python -m scripts.pose_on_fullframe data/raw/train/cover_drive/cd1.mp4 ^
#       --save data/pose_vis_clean/cover_drive/cd1_pose_full.mp4 ^
#       --pose_out data/pose_clean/cover_drive/cd1_pose.npy
#
# Notes:
# - The pose saved is in FULL-FRAME normalized coordinates (x,y in [0..1] wrt original frame).
# - If YOLO misses for some frames, we append zeros for pose (keeps length consistent).

import argparse
from pathlib import Path

import cv2
import numpy as np
import mediapipe as mp
from ultralytics import YOLO

mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils
POSE_LANDMARKS = 33


def clamp_box(x1, y1, x2, y2, W, H):
    x1 = max(0, min(int(x1), W - 1))
    y1 = max(0, min(int(y1), H - 1))
    x2 = max(0, min(int(x2), W - 1))
    y2 = max(0, min(int(y2), H - 1))
    if x2 <= x1:
        x2 = min(W - 1, x1 + 1)
    if y2 <= y1:
        y2 = min(H - 1, y1 + 1)
    return x1, y1, x2, y2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video", help="Input video path (.mp4)")
    ap.add_argument("--det_model", default="models/batter_detector.pt", help="YOLO batter detector .pt")
    ap.add_argument("--conf", type=float, default=0.35, help="YOLO confidence threshold")
    ap.add_argument("--iou", type=float, default=0.5, help="YOLO NMS IoU threshold")
    ap.add_argument("--pad", type=float, default=0.10, help="BBox padding ratio (0.0-0.4)")
    ap.add_argument("--save", default="", help="Optional: output debug mp4 path")
    ap.add_argument("--pose_out", default="", help="Optional: output pose .npy path")
    ap.add_argument("--show", action="store_true", help="Show live preview window")
    ap.add_argument("--max_frames", type=int, default=0, help="0=all, else limit frames (debug)")
    args = ap.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        raise SystemExit(f"Video not found: {video_path}")

    # Ensure output dirs exist
    if args.save:
        Path(args.save).parent.mkdir(parents=True, exist_ok=True)
    if args.pose_out:
        Path(args.pose_out).parent.mkdir(parents=True, exist_ok=True)

    det = YOLO(args.det_model)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = None
    if args.save:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(args.save), fourcc, fps, (W, H))

    pose = mp_pose.Pose(static_image_mode=False, model_complexity=1, enable_segmentation=False)

    pose_sequence = []
    frame_idx = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame_idx += 1
            if args.max_frames and frame_idx > args.max_frames:
                break

            # Default: empty pose for this frame
            lm_full = np.zeros((POSE_LANDMARKS, 3), dtype=np.float32)

            # ---- YOLO detect ----
            res = det.predict(frame, conf=args.conf, iou=args.iou, verbose=False)
            if res and res[0].boxes is not None and len(res[0].boxes) > 0:
                xyxy = res[0].boxes.xyxy.cpu().numpy()
                confs = res[0].boxes.conf.cpu().numpy()
                idx = int(np.argmax(confs))
                x1, y1, x2, y2 = xyxy[idx]

                # pad bbox a little (helps include arms/bat)
                bw = x2 - x1
                bh = y2 - y1
                x1 = x1 - args.pad * bw
                x2 = x2 + args.pad * bw
                y1 = y1 - args.pad * bh
                y2 = y2 + args.pad * bh

                x1, y1, x2, y2 = clamp_box(x1, y1, x2, y2, W, H)

                # Draw bbox
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    f"batter {float(confs[idx]):.2f}",
                    (x1, max(0, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )

                # ---- Pose on ROI ----
                roi = frame[y1:y2, x1:x2].copy()
                if roi.size > 0:
                    roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
                    pres = pose.process(roi_rgb)

                    if pres.pose_landmarks:
                        # Convert ROI-normalized landmarks to FULL-FRAME normalized landmarks
                        roi_w = max(1, (x2 - x1))
                        roi_h = max(1, (y2 - y1))
                        for i, p in enumerate(pres.pose_landmarks.landmark[:POSE_LANDMARKS]):
                            # p.x, p.y are normalized inside ROI (0..1)
                            fx = (x1 + p.x * roi_w) / W
                            fy = (y1 + p.y * roi_h) / H
                            lm_full[i, 0] = float(fx)
                            lm_full[i, 1] = float(fy)
                            lm_full[i, 2] = float(p.visibility)

                        # Draw skeleton on FULL frame:
                        # We can draw directly using the MediaPipe object, but it expects normalized
                        # coordinates relative to the image. We already set them by mapping.
                        # So we can temporarily overwrite the landmark coords with full-frame normalized.
                        for i, p in enumerate(pres.pose_landmarks.landmark[:POSE_LANDMARKS]):
                            p.x = lm_full[i, 0]
                            p.y = lm_full[i, 1]
                        mp_draw.draw_landmarks(frame, pres.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            pose_sequence.append(lm_full)

            if writer:
                writer.write(frame)

            if args.show:
                cv2.imshow("YOLO + MediaPipe Pose (Full Frame)", frame)
                # ESC to quit
                if cv2.waitKey(1) & 0xFF == 27:
                    break

    finally:
        cap.release()
        pose.close()
        if writer:
            writer.release()
        if args.show:
            cv2.destroyAllWindows()

    # ---- Save pose sequence ----
    if args.pose_out:
        pose_arr = np.stack(pose_sequence) if pose_sequence else np.zeros((0, POSE_LANDMARKS, 3), np.float32)
        np.save(args.pose_out, pose_arr)
        print(f"[OK] Saved pose: {args.pose_out}  frames={len(pose_arr)}")

    if args.save:
        print(f"[OK] Saved debug video: {args.save}")


if __name__ == "__main__":
    main()

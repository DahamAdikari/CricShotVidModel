
#!/usr/bin/env python
# preview_batter_tracks.py (v1.1)
# Adds robust key handling for Windows arrow keys + WASD fallbacks.

import argparse
import csv
import json
from pathlib import Path
import cv2

def load_tracks(tracks_path: Path):
    with open(tracks_path, "r") as f:
        raw = json.load(f)
    tracks = {int(k): v for k, v in raw.items()}
    ptrs = {tid: 0 for tid in tracks.keys()}
    return tracks, ptrs

def advance_ptrs(ptrs, tracks):
    for tid in ptrs.keys():
        if ptrs[tid] < len(tracks[tid]) - 1:
            ptrs[tid] += 1

def draw_box(frame, box, tid, highlight=False):
    x1, y1, x2, y2 = [int(v) for v in box]
    color = (0, 255, 255) if highlight else (0, 255, 0)
    thickness = 3 if highlight else 1
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    label = f"ID {tid}"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 6, y1), color, -1)
    cv2.putText(frame, label, (x1 + 3, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2, cv2.LINE_AA)

def write_label(labels_csv: Path, video_name: str, track_id: int):
    labels_csv.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    if labels_csv.exists():
        with open(labels_csv, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                rows.append(r)
    updated = False
    for r in rows:
        if r.get("video") == video_name:
            r["track_id"] = str(track_id)
            updated = True
            break
    if not updated:
        rows.append({"video": video_name, "track_id": str(track_id)})
    with open(labels_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["video", "track_id"])
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, help="Path to the original video file")
    ap.add_argument("--tracks", help="Path to corresponding *_tracks.json (default: auto from data/tracks)")
    ap.add_argument("--labels", default="data/labels.csv", help="CSV file to write (video,track_id)")
    args = ap.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        raise SystemExit(f"Video not found: {video_path}")

    tracks_path = Path(args.tracks) if args.tracks else Path("data/tracks") / f"{video_path.stem}_tracks.json"
    if not tracks_path.exists():
        raise SystemExit(f"Tracks JSON not found: {tracks_path}\nRun tracking first for this video.")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {video_path}")
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    tracks, ptrs = load_tracks(tracks_path)
    if not tracks:
        raise SystemExit("No tracks in JSON.")
    ids = sorted(tracks.keys())
    sel_idx = 0
    selected_id = ids[sel_idx]

    paused = False
    win = "Pick Batter (B=save, arrows/WASD switch, P=pause, Q=quit)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, min(1280, W), min(720, H))

    # Common OpenCV keycodes for arrows:
    LEFT_CODES  = {81, 2424832}   # Linux/Qt, Windows
    RIGHT_CODES = {83, 2555904}
    UP_CODES    = {82, 2490368}
    DOWN_CODES  = {84, 2621440}

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                for tid in ptrs: ptrs[tid] = 0
                ret, frame = cap.read()
                if not ret: break

            for tid in ids:
                if ptrs[tid] < len(tracks[tid]):
                    draw_box(frame, tracks[tid][ptrs[tid]], tid, highlight=(tid == selected_id))
            advance_ptrs(ptrs, tracks)

        hud = f"Selected ID: {selected_id}    Video: {video_path.name}"
        cv2.putText(frame, hud, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2, cv2.LINE_AA)
        cv2.imshow(win, frame)

        key = cv2.waitKey(1) & 0xFFFFFFFF  # keep full code on Windows
        if key in (ord('q'), 27):
            break
        elif key == ord('p'):
            paused = not paused
        elif key == ord('b'):
            write_label(Path(args.labels), video_path.name, selected_id)
            print(f"[SAVED] {video_path.name},{selected_id} -> {args.labels}")
        elif key in LEFT_CODES or key == ord('a'):
            sel_idx = (sel_idx - 1) % len(ids)
            selected_id = ids[sel_idx]
        elif key in RIGHT_CODES or key == ord('d'):
            sel_idx = (sel_idx + 1) % len(ids)
            selected_id = ids[sel_idx]
        elif key in UP_CODES or key == ord('w'):
            sel_idx = (sel_idx + 1) % len(ids)
            selected_id = ids[sel_idx]
        elif key in DOWN_CODES or key == ord('s'):
            sel_idx = (sel_idx - 1) % len(ids)
            selected_id = ids[sel_idx]
        # else: ignore

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

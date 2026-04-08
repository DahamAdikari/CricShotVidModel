#1

import cv2, json, argparse
from pathlib import Path
from ultralytics import YOLO
import supervision as sv
from scripts.utils import Cfg, ensure_dir, norm_roi_to_xyxy, load_overrides

def pick_batter_track(frame_size, tracks, roi_norm, overrides, video_name):
    W,H = frame_size
    rx1, ry1, rx2, ry2 = norm_roi_to_xyxy(roi_norm, W, H)
    if video_name in overrides:
        return overrides[video_name]
    # score by time inside ROI and proximity
    cx0 = (rx1+rx2)//2; cy0 = ry1
    scores = {}
    for tid, boxes in tracks.items():
        inside = 0; prox_acc = 0; n = 0
        for (x1,y1,x2,y2) in boxes:
            cx = (x1+x2)/2; cy = (y1+y2)/2
            if rx1<=cx<=rx2 and ry1<=cy<=ry2:
                inside += 1
            prox_acc += abs(cx-cx0) + 2*max(0, cy0-cy)
            n += 1
        if n>0:
            scores[int(tid)] = inside / max(1, prox_acc/n)
    if not scores:
        scores = {int(tid): len(b) for tid,b in tracks.items()}
    return max(scores, key=scores.get)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('video')
    ap.add_argument('--out', default='data/crops')
    ap.add_argument('--tracks', default='data/tracks')
    ap.add_argument('--config', default='src/config.yaml')
    args = ap.parse_args()

    cfg = Cfg(args.config)
    ensure_dir(args.out); ensure_dir(args.tracks)
    model = YOLO('models/batter_detector.pt')

    video_path = Path(args.video)
    results = model.track(source=str(video_path), save=False, tracker=cfg.tracker)

    frame_w = frame_h = None
    tracks = {}
    for r in results:
        if frame_w is None:
            frame_h, frame_w = r.orig_shape
        if r.boxes is None or r.boxes.id is None: continue
        ids = r.boxes.id.int().tolist()
        xyxy = r.boxes.xyxy.cpu().numpy().tolist()
        for tid, box in zip(ids, xyxy):
            t = tracks.setdefault(int(tid), [])
            t.append([float(x) for x in box])

    with open(Path(args.tracks)/f"{video_path.stem}_tracks.json", "w") as f:
        json.dump(tracks, f)

    overrides = load_overrides(Path('data/labels.csv'))
    batter_id = pick_batter_track((frame_w,frame_h), tracks, cfg.batter_roi, overrides, video_path.name)
    print(f"Chosen batter track id: {batter_id}")

if __name__ == '__main__':
    main()

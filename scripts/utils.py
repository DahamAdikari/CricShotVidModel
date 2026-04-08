
import cv2, numpy as np, yaml
from pathlib import Path
import pandas as pd

class Cfg:
    def __init__(self, path):
        with open(path, 'r') as f:
            self.__dict__.update(yaml.safe_load(f))

def ensure_dir(p: str|Path):
    Path(p).mkdir(parents=True, exist_ok=True)

def crop_box_from_bbox(frame_w, frame_h, bbox, crop_w, crop_h):
    x1,y1,x2,y2 = bbox
    cx = (x1+x2)/2; cy = (y1+y2)/2
    left = int(max(0, cx - crop_w/2))
    top  = int(max(0, cy - crop_h/2))
    right = int(min(frame_w, left + crop_w))
    bottom= int(min(frame_h, top + crop_h))
    left = right - crop_w
    top  = bottom - crop_h
    return max(0,left), max(0,top), min(frame_w,right), min(frame_h,bottom)

def norm_roi_to_xyxy(roi, w, h):
    return [int(roi['x1']*w), int(roi['y1']*h), int(roi['x2']*w), int(roi['y2']*h)]

def load_overrides(csv_path: Path):
    if not csv_path.exists():
        return {}
    df = pd.read_csv(csv_path)
    return {row.video: int(row.track_id) for _,row in df.iterrows()}

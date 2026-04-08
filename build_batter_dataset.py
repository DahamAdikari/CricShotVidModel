
#!/usr/bin/env python

"""
build_batter_dataset.py
---------------------------------
Create a YOLOv8-ready dataset from LabelImg outputs, ensuring
unique filenames and an 80/20 train/val split.

Assumes your labeled images are under:
    data/batter_label_images/<clip_folder>/*.jpg
with YOLO sidecar labels:
    data/batter_label_images/<clip_folder>/*.txt

Empty .txt files are created automatically for images with no batter.

Usage (run from project root):
    python build_batter_dataset.py \
        --src data/batter_label_images \
        --dst data/batter_detector \
        --train_ratio 0.8 \
        --seed 42
"""

import argparse
from pathlib import Path
import shutil
import random

IMG_EXTS = {'.jpg', '.jpeg', '.png'}

def copy_pair(img_path: Path, lbl_path: Path, out_img: Path, out_lbl: Path):
    out_img.parent.mkdir(parents=True, exist_ok=True)
    out_lbl.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(img_path, out_img)
    if lbl_path and lbl_path.exists():
        shutil.copy2(lbl_path, out_lbl)
    else:
        out_lbl.write_text("", encoding="utf-8")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/batter_label_images", help="Root folder containing labeled images subfolders")
    ap.add_argument("--dst", default="data/batter_detector", help="Output dataset root (YOLO layout will be created)")
    ap.add_argument("--train_ratio", type=float, default=0.8, help="Train split ratio, rest goes to val")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = ap.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)
    img_train = dst / "images" / "train"
    img_val   = dst / "images" / "val"
    lbl_train = dst / "labels" / "train"
    lbl_val   = dst / "labels" / "val"
    for d in (img_train, img_val, lbl_train, lbl_val):
        d.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)

    all_imgs = [p for p in src.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS]
    n = len(all_imgs)
    if n == 0:
        print(f"[WARN] No images found under {src}.")
        return

    copied_train = copied_val = 0
    for img in all_imgs:
        parent = img.parent.name  # e.g., cd1
        base = img.stem           # e.g., frame_0001
        ext = img.suffix          # e.g., .jpg
        lbl = img.with_suffix(".txt")

        new_img_name = f"{parent}_{base}{ext}"
        new_lbl_name = f"{parent}_{base}.txt"

        split = "train" if random.random() < args.train_ratio else "val"
        if split == "train":
            out_img = img_train / new_img_name
            out_lbl = lbl_train / new_lbl_name
            copied_train += 1
        else:
            out_img = img_val / new_img_name
            out_lbl = lbl_val / new_lbl_name
            copied_val += 1

        copy_pair(img, lbl, out_img, out_lbl)

    print(f"[DONE] Dataset created at {dst}")
    print(f"  Train images: {copied_train}")
    print(f"  Val images:   {copied_val}")
    print("\nNext: create data/batter.yaml with:")
    print('''train: data/batter_detector/images/train
val: data/batter_detector/images/val
names: ["batter"]''')

if __name__ == "__main__":
    main()

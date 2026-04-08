# scripts/eval_batter_detector.py
# Evaluate trained YOLO batter detector on validation split
#
# Usage:
#   python -m scripts.eval_batter_detector --model models/batter_detector.pt --data data/batter.yaml
#
# Outputs:
#   reports/yolo_batter_eval/
#       args.yaml
#       confusion_matrix.png
#       confusion_matrix_normalized.png
#       F1_curve.png
#       P_curve.png
#       PR_curve.png
#       R_curve.png
#       val_batch*_pred.jpg
#       val_batch*_labels.jpg
#       metrics_summary.txt

import argparse
from pathlib import Path
from ultralytics import YOLO

def safe_get(obj, name, default=None):
    return getattr(obj, name, default) if hasattr(obj, name) else default

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/batter_detector.pt", help="trained YOLO model")
    ap.add_argument("--data", default="data/batter.yaml", help="YOLO data yaml")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--project", default="reports")
    ap.add_argument("--name", default="yolo_batter_eval")
    args = ap.parse_args()

    Path(args.project).mkdir(parents=True, exist_ok=True)

    model = YOLO(args.model)

    # Run validation
    metrics = model.val(
        data=args.data,
        imgsz=args.imgsz,
        split="val",
        plots=True,
        save_json=True,
        project=args.project,
        name=args.name,
        exist_ok=True,
        verbose=True,
    )

    out_dir = Path(args.project) / args.name
    out_dir.mkdir(parents=True, exist_ok=True)

    # Try to extract the most common YOLO metrics
    box = safe_get(metrics, "box", None)

    precision = safe_get(box, "mp", None)
    recall = safe_get(box, "mr", None)
    map50 = safe_get(box, "map50", None)
    map5095 = safe_get(box, "map", None)
    map75 = safe_get(box, "map75", None)

    summary_lines = [
        "YOLO Batter Detector Validation Summary",
        "======================================",
        f"Model: {args.model}",
        f"Data : {args.data}",
        "",
        f"Precision (mean): {precision}",
        f"Recall (mean)   : {recall}",
        f"mAP@0.5         : {map50}",
        f"mAP@0.75        : {map75}",
        f"mAP@0.5:0.95    : {map5095}",
        "",
        "Generated plots/files usually include:",
        "- confusion_matrix.png",
        "- confusion_matrix_normalized.png",
        "- PR_curve.png",
        "- P_curve.png",
        "- R_curve.png",
        "- F1_curve.png",
        "- val_batch*_pred.jpg",
        "- val_batch*_labels.jpg",
    ]

    summary_path = out_dir / "metrics_summary.txt"
    summary_path.write_text("\n".join(summary_lines), encoding="utf-8")

    print(f"\n[OK] YOLO evaluation finished.")
    print(f"[OK] Reports saved to: {out_dir}")
    print(f"[OK] Summary saved to: {summary_path}")

if __name__ == "__main__":
    main()
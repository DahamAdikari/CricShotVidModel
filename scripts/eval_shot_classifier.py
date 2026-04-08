# scripts/eval_shot_classifier.py
# Evaluate trained shot classifier on pose .npy files
#
# Expected folder structure:
#   data/pose_clean/
#       cover_drive/*.npy
#       pull_shot/*.npy
#       other/*.npy   (optional)
#
# Usage:
#   python -m scripts.eval_shot_classifier --pose_root data/pose_clean --model models/shot_lstm.pt
#
# Outputs:
#   reports/classification_report.txt
#   reports/predictions.csv
#   reports/confusion_matrix.png

import argparse
from pathlib import Path
import csv

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


CLASSES = ["cover_drive", "pull_shot", "other"]

LEFT  = [11,13,15,23,25,27,29,31]
RIGHT = [12,14,16,24,26,28,30,32]


# ---------------- Model (must match training) ----------------
class BiLSTMShot(nn.Module):
    def __init__(self, in_dim=33 * 5, hidden=256, num_layers=2, num_classes=3, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=in_dim,
            hidden_size=hidden,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
        )
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden * 2, num_classes),
        )

    def forward(self, x):
        B, T, N, D = x.shape
        x = x.view(B, T, N * D)
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


# ---------------- Feature helpers ----------------
def mirror_sequence(seq):
    mir = seq.copy()
    mir[:, :, 0] = 1.0 - mir[:, :, 0]
    for l, r in zip(LEFT, RIGHT):
        mir[:, [l, r], :] = mir[:, [r, l], :]
    return mir

def norm_frame(frame):
    midhip = (frame[23, :2] + frame[24, :2]) / 2
    shoulder = np.linalg.norm(frame[11, :2] - frame[12, :2]) + 1e-6
    xy = (frame[:, :2] - midhip) / shoulder
    vis = frame[:, 2:3]
    return np.concatenate([xy, vis], axis=1).astype(np.float32)

def add_velocity(seq3):
    xy = seq3[:, :, :2]
    vis = seq3[:, :, 2:3]
    dxy = np.zeros_like(xy, dtype=np.float32)
    dxy[1:] = xy[1:] - xy[:-1]
    return np.concatenate([xy, vis, dxy], axis=2).astype(np.float32)

def pad_or_trim(seq, T):
    if len(seq) >= T:
        return seq[-T:]
    pad = np.repeat(seq[:1], T - len(seq), axis=0)
    return np.concatenate([pad, seq], axis=0)

def softmax_np(logits):
    x = logits - logits.max()
    e = np.exp(x)
    return e / (e.sum() + 1e-9)


# ---------------- Data loading ----------------
def load_eval_files(root):
    files = []
    labels = []
    for ci, cls in enumerate(CLASSES):
        cls_dir = Path(root, cls)
        if not cls_dir.exists():
            continue
        for npy in cls_dir.rglob("*.npy"):
            files.append(str(npy))
            labels.append(ci)
    return files, labels


def build_input(seq_raw, T):
    """
    seq_raw: (frames,33,3)
    returns:
      xA: original features  (1,T,33,5)
      xB: mirrored features  (1,T,33,5)
    """
    seq_raw = pad_or_trim(seq_raw, T)

    seq_n = np.stack([norm_frame(f) for f in seq_raw], axis=0)
    seq_f = add_velocity(seq_n)

    seq_m = mirror_sequence(seq_raw)
    seq_mn = np.stack([norm_frame(f) for f in seq_m], axis=0)
    seq_mf = add_velocity(seq_mn)

    return seq_f[None], seq_mf[None]


# ---------------- Main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pose_root", default="data/pose_clean")
    ap.add_argument("--model", default="models/shot_lstm.pt")
    ap.add_argument("--report_dir", default="reports")
    args = ap.parse_args()

    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(args.model, map_location="cpu")
    T = int(ckpt.get("T", 32))

    net = BiLSTMShot()
    net.load_state_dict(ckpt["state_dict"])
    net.eval()

    files, y_true = load_eval_files(args.pose_root)
    if not files:
        raise RuntimeError("No .npy files found under pose_root")

    y_pred = []
    all_probs = []

    for fpath in files:
        seq = np.load(fpath)  # (frames,33,3)

        xA, xB = build_input(seq, T)
        xA = torch.tensor(xA, dtype=torch.float32)
        xB = torch.tensor(xB, dtype=torch.float32)

        with torch.no_grad():
            pA = softmax_np(net(xA).numpy().squeeze())
            pB = softmax_np(net(xB).numpy().squeeze())

        p = np.maximum(pA, pB)
        pred = int(np.argmax(p))

        y_pred.append(pred)
        all_probs.append(p)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    all_probs = np.array(all_probs)

    # Which classes actually exist in evaluation set?
    present_classes = sorted(set(y_true.tolist()))
    target_names = [CLASSES[i] for i in present_classes]

    # Classification report
    report_txt = classification_report(
        y_true,
        y_pred,
        labels=present_classes,
        target_names=target_names,
        digits=4,
        zero_division=0
    )

    print("\n===== CLASSIFICATION REPORT =====\n")
    print(report_txt)

    (report_dir / "classification_report.txt").write_text(report_txt, encoding="utf-8")

    # Save prediction CSV
    csv_path = report_dir / "predictions.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = ["file", "true_label", "pred_label"] + [f"prob_{c}" for c in CLASSES]
        writer.writerow(header)

        for fpath, yt, yp, probs in zip(files, y_true, y_pred, all_probs):
            writer.writerow([
                fpath,
                CLASSES[yt],
                CLASSES[yp],
                float(probs[0]),
                float(probs[1]),
                float(probs[2]),
            ])

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=present_classes)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)

    fig, ax = plt.subplots(figsize=(6, 6))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    plt.title("Shot Classifier Confusion Matrix")
    plt.tight_layout()
    plt.savefig(report_dir / "confusion_matrix.png", dpi=200)
    plt.close(fig)

    print(f"\n[OK] Saved report to: {report_dir / 'classification_report.txt'}")
    print(f"[OK] Saved predictions to: {csv_path}")
    print(f"[OK] Saved confusion matrix to: {report_dir / 'confusion_matrix.png'}")


if __name__ == "__main__":
    main()
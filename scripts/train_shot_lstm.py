# scripts/train_shot_lstm.py
# Accuracy-first version:
# - Adds velocity features (dx, dy)
# - 2-layer BiLSTM + dropout
# - mini-batch training + validation
# Saves: models/shot_lstm.pt

import argparse
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

CLASSES = ["cover_drive", "pull_shot", "other"]

LEFT  = [11,13,15,23,25,27,29,31]
RIGHT = [12,14,16,24,26,28,30,32]


def mirror_sequence(seq):
    """Mirror pose horizontally and swap left/right joints.
    seq: (T, 33, 3) in normalized coords (x,y,vis)"""
    mir = seq.copy()
    mir[:, :, 0] = 1.0 - mir[:, :, 0]  # flip x
    for l, r in zip(LEFT, RIGHT):
        mir[:, [l, r], :] = mir[:, [r, l], :]
    return mir


def norm_frame(frame):
    """Normalize using mid-hip origin and shoulder scale.
    frame: (33,3) => returns (33,3) with xy normalized, vis kept"""
    midhip = (frame[23, :2] + frame[24, :2]) / 2
    shoulder = np.linalg.norm(frame[11, :2] - frame[12, :2]) + 1e-6
    xy = (frame[:, :2] - midhip) / shoulder
    vis = frame[:, 2:3]
    return np.concatenate([xy, vis], axis=1).astype(np.float32)


def add_velocity(seq3):
    """Convert (T,33,3) -> (T,33,5) by adding dx,dy.
    Uses frame-to-frame diff on normalized xy."""
    T = seq3.shape[0]
    xy = seq3[:, :, :2]
    vis = seq3[:, :, 2:3]
    dxy = np.zeros_like(xy, dtype=np.float32)
    dxy[1:] = xy[1:] - xy[:-1]
    return np.concatenate([xy, vis, dxy], axis=2).astype(np.float32)


def pad_or_trim(seq, T=32):
    """Pad at the start by repeating first frame if too short, else keep last T."""
    if len(seq) >= T:
        return seq[-T:]
    pad = np.repeat(seq[:1], T - len(seq), axis=0)
    return np.concatenate([pad, seq], axis=0)


def load_sequences(root):
    """Load all *_pose.npy under class folders and build features."""
    X, y = [], []
    for ci, cls in enumerate(CLASSES):
        cls_dir = Path(root, cls)
        if not cls_dir.exists():
            continue
        for npy in cls_dir.rglob("*_pose.npy"):
            seq = np.load(npy)  # (T,33,3)
            if len(seq) < 12:
                continue

            # normalize each frame
            seq_n = np.stack([norm_frame(f) for f in seq], axis=0)  # (T,33,3)
            seq_f = add_velocity(seq_n)  # (T,33,5)

            X.append(seq_f)
            y.append(ci)

            # augment mirrored (helps left/right handedness invariance)
            seq_m = mirror_sequence(seq)
            seq_mn = np.stack([norm_frame(f) for f in seq_m], axis=0)
            seq_mf = add_velocity(seq_mn)
            X.append(seq_mf)
            y.append(ci)

    return X, y


class PoseDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=torch.float32), torch.tensor(self.y[idx], dtype=torch.long)


class BiLSTMShot(nn.Module):
    def __init__(self, in_dim=33*5, hidden=256, num_layers=2, num_classes=3, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=in_dim,
            hidden_size=hidden,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True
        )
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden * 2, num_classes)  # *2 for bidirectional
        )

    def forward(self, x):
        # x: (B,T,33,5)
        B, T, N, D = x.shape
        x = x.view(B, T, N * D)  # (B,T,165)
        out, _ = self.lstm(x)    # (B,T,2*hidden)
        logits = self.fc(out[:, -1, :])  # last timestep
        return logits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pose_root", default="data/pose_clean", help="Folder with cover_drive/pull_shot/other")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--T", type=int, default=32, help="Window length (frames) for training")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out", default="models/shot_lstm.pt")
    args = ap.parse_args()

    X, y = load_sequences(args.pose_root)
    if not X:
        raise RuntimeError("No pose sequences found. Check your pose_root and file names *_pose.npy")

    # pad/trim
    T = args.T
    X = [pad_or_trim(s, T) for s in X]
    X = np.stack(X).astype(np.float32)
    y = np.array(y, dtype=np.int64)

    # split
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=0, stratify=y)

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = BiLSTMShot().to(dev)

    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    train_loader = DataLoader(PoseDataset(Xtr, ytr), batch_size=args.batch, shuffle=True)
    val_loader = DataLoader(PoseDataset(Xte, yte), batch_size=args.batch, shuffle=False)

    best_acc = -1.0

    for ep in range(args.epochs):
        net.train()
        total_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(dev), yb.to(dev)
            opt.zero_grad()
            logits = net(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()
            total_loss += float(loss.item()) * len(xb)

        # val
        net.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(dev), yb.to(dev)
                logits = net(xb)
                pred = logits.argmax(1)
                correct += int((pred == yb).sum().item())
                total += len(xb)
        acc = correct / max(1, total)
        avg_loss = total_loss / max(1, len(Xtr))
        print(f"epoch {ep+1}/{args.epochs} loss={avg_loss:.3f} val_acc={acc:.3f}")

        # save best
        if acc > best_acc:
            best_acc = acc
            Path("models").mkdir(parents=True, exist_ok=True)
            torch.save({
                "state_dict": net.state_dict(),
                "T": T,
                "classes": CLASSES,
                "in_feat": 5,
                "arch": "BiLSTMShot(hidden=256,layers=2,dropout=0.3)",
            }, args.out)

    print(f"[OK] Saved best model to {args.out} (best val_acc={best_acc:.3f})")


if __name__ == "__main__":
    main()

# scripts/detect_shots.py
# End-to-end:
# PASS 1 (sampled):
#   YOLO batter bbox -> MediaPipe pose on ROI -> map pose to full-frame normalized coords
#   store times, bboxes, confs, pose_arr
#
# CLASSIFY:
#   BiLSTM on pose windows (features: normalized xy+vis + velocity), plus mirrored invariance
#   build per-sampled-frame probs timeline
#
# EXPORT (UPDATED - two-pass timeline trim):
#   1) Create boolean mask where P(class)>=thr AND batter_conf>=BATTER_MIN
#   2) Convert mask -> segments, merge small gaps
#   3) Export only those segments with 1s padding using ffmpeg
#
# PASS 2 (debug overlay, optional):
#   Re-read full FPS video, interpolate bbox/pose/probs and draw smoothly
#
# Usage:
#   python -m scripts.detect_shots data/raw/full_match.mp4 --model models/shot_lstm.pt --cls pull_shot --out_dir data/segments
#   python -m scripts.detect_shots data/raw/full_match.mp4 --model models/shot_lstm.pt --cls all --debug_out data/samples/debug_overlay_interp.mp4
#
# Notes:
# - Requires ffmpeg in PATH.
# - The google protobuf warning is harmless (MediaPipe dependency warning).

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import mediapipe as mp
from ultralytics import YOLO
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip

CLASSES = ["cover_drive", "pull_shot", "other"]


# ------------------ Model (must match training) ------------------
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
        # x: (B,T,33,5)
        B, T, N, D = x.shape
        x = x.view(B, T, N * D)
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


# ------------------ Pose helpers ------------------
LEFT  = [11, 13, 15, 23, 25, 27, 29, 31]
RIGHT = [12, 14, 16, 24, 26, 28, 30, 32]

def mirror_sequence(seq):
    """seq: (T,33,3) -> mirrored horizontally and swapped left/right joints."""
    mir = seq.copy()
    mir[:, :, 0] = 1.0 - mir[:, :, 0]
    for l, r in zip(LEFT, RIGHT):
        mir[:, [l, r], :] = mir[:, [r, l], :]
    return mir

def norm_frame(frame):
    """Normalize pose: origin at mid-hip, scale by shoulder distance; keep visibility."""
    midhip = (frame[23, :2] + frame[24, :2]) / 2
    shoulder = np.linalg.norm(frame[11, :2] - frame[12, :2]) + 1e-6
    xy = (frame[:, :2] - midhip) / shoulder
    vis = frame[:, 2:3]
    return np.concatenate([xy, vis], axis=1).astype(np.float32)  # (33,3)

def add_velocity(seq3):
    """(T,33,3)->(T,33,5): add dx,dy on normalized xy."""
    xy = seq3[:, :, :2]
    vis = seq3[:, :, 2:3]
    dxy = np.zeros_like(xy, dtype=np.float32)
    dxy[1:] = xy[1:] - xy[:-1]
    return np.concatenate([xy, vis, dxy], axis=2).astype(np.float32)

def softmax_np(logits):
    x = logits - logits.max()
    e = np.exp(x)
    return e / (e.sum() + 1e-9)

def entropy(p):
    p = np.clip(p, 1e-8, 1.0)
    return float(-(p * np.log(p)).sum())


# ------------------ Drawing helpers ------------------
mp_pose = mp.solutions.pose
POSE_CONNECTIONS = list(mp_pose.POSE_CONNECTIONS)

def draw_label_box(img, lines, x=10, y=10):
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.55
    thick = 2
    pad = 8
    sizes = [cv2.getTextSize(t, font, scale, thick)[0] for t in lines]
    w = max(s[0] for s in sizes) + pad * 2
    h = sum(s[1] for s in sizes) + pad * 2 + (len(lines) - 1) * 6
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 0), -1)
    cy = y + pad + sizes[0][1]
    for t, (tw, th) in zip(lines, sizes):
        cv2.putText(img, t, (x + pad, cy), font, scale, (255, 255, 255), thick, cv2.LINE_AA)
        cy += th + 6

def draw_skeleton_fullframe(img, lm_full, color_pts=(0, 0, 255), color_lines=(255, 255, 255), vis_thr=0.5):
    H, W = img.shape[:2]
    pts = []
    good = []
    for i in range(lm_full.shape[0]):
        x, y, v = lm_full[i]
        px = int(x * W)
        py = int(y * H)
        pts.append((px, py))
        good.append(v >= vis_thr and 0 <= px < W and 0 <= py < H)

    for i, ok in enumerate(good):
        if ok:
            cv2.circle(img, pts[i], 3, color_pts, -1)

    for a, b in POSE_CONNECTIONS:
        if a < len(good) and b < len(good) and good[a] and good[b]:
            cv2.line(img, pts[a], pts[b], color_lines, 2)


# ------------------ Detection helpers ------------------
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

def pick_top1_box(det_res, W, H, conf_thr):
    """Return (x1,y1,x2,y2,conf) or None."""
    if not det_res or det_res[0].boxes is None or len(det_res[0].boxes) == 0:
        return None
    xyxy = det_res[0].boxes.xyxy.cpu().numpy()
    confs = det_res[0].boxes.conf.cpu().numpy()
    idx = int(np.argmax(confs))
    if float(confs[idx]) < conf_thr:
        return None
    x1, y1, x2, y2 = xyxy[idx]
    x1, y1, x2, y2 = clamp_box(x1, y1, x2, y2, W, H)
    return x1, y1, x2, y2, float(confs[idx])


# ------------------ Export helpers (NEW two-pass timeline trim) ------------------
def mask_to_segments(times, mask, max_gap_s=0.35, min_len_s=0.10):
    """
    Convert a boolean mask over sampled timeline into merged segments.
    - mask[i] True means "this sampled time belongs to the shot"
    - max_gap_s: merge if the gap between segments <= this
    - min_len_s: keep even short segments (fast shots)
    Returns list of (start_idx, end_idx) inclusive.
    """
    S = len(times)
    if S == 0:
        return []

    # 1) raw contiguous True runs
    raw = []
    i = 0
    while i < S:
        if not mask[i]:
            i += 1
            continue
        s = i
        while i + 1 < S and mask[i + 1]:
            i += 1
        e = i
        raw.append((s, e))
        i += 1

    if not raw:
        return []

    # 2) merge small gaps
    merged = [raw[0]]
    for s, e in raw[1:]:
        ps, pe = merged[-1]
        gap = float(times[s] - times[pe])
        if gap <= max_gap_s:
            merged[-1] = (ps, e)
        else:
            merged.append((s, e))

    # 3) drop too-short segments
    out = []
    for s, e in merged:
        if float(times[e] - times[s]) >= min_len_s:
            out.append((s, e))
    return out

def unique_out_path(base_dir: Path, stem: str, cls_name: str, idx: int):
    """
    Ensure we never overwrite the same filename.
    """
    base_dir.mkdir(parents=True, exist_ok=True)
    k = idx
    while True:
        out = base_dir / f"{stem}_{cls_name}_{k:02d}.mp4"
        if not out.exists():
            return out
        k += 1

def export_segments_ffmpeg(video_path, times, segs, out_dir, stem, cls_name, pad_s=1.0):
    """
    Export list of segments (indices on sampled timeline) as MP4 clips.
    Adds pad_s seconds before start and after end.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for idx, (a, b) in enumerate(segs, 1):
        t1 = max(0.0, float(times[a]) - pad_s)
        t2 = float(times[b]) + pad_s
        out = unique_out_path(out_dir, stem, cls_name, idx)
        ffmpeg_extract_subclip(str(video_path), t1, t2, targetname=str(out))
        print(f"[OK] Exported {out} ({t1:.2f}s -> {t2:.2f}s)")
        count += 1
    return count


# ------------------ Interpolation (debug overlay) ------------------
def lerp(a, b, alpha):
    return (1.0 - alpha) * a + alpha * b

def interp_bbox(bb0, bb1, alpha):
    if bb0 is None and bb1 is None:
        return None
    if bb0 is None:
        return bb1
    if bb1 is None:
        return bb0
    a = np.array(bb0, dtype=np.float32)
    b = np.array(bb1, dtype=np.float32)
    c = lerp(a, b, alpha)
    return tuple(int(v) for v in c.tolist())

def interp_pose(p0, p1, alpha):
    if p0 is None and p1 is None:
        return None
    if p0 is None:
        return p1
    if p1 is None:
        return p0
    return lerp(p0, p1, alpha).astype(np.float32)

def interp_probs(q0, q1, alpha):
    if q0 is None and q1 is None:
        return None
    if q0 is None:
        return q1
    if q1 is None:
        return q0
    out = lerp(q0, q1, alpha).astype(np.float32)
    s = float(out.sum())
    if s > 1e-9:
        out /= s
    return out

def find_sample_neighbors(times, t):
    """
    times sorted (S,)
    return k0,k1,alpha where t is between times[k0] and times[k1]
    """
    S = len(times)
    if S == 0:
        return 0, 0, 0.0
    if t <= times[0]:
        return 0, 0, 0.0
    if t >= times[-1]:
        return S - 1, S - 1, 0.0

    k1 = int(np.searchsorted(times, t, side="right"))
    k0 = max(0, k1 - 1)
    k1 = min(S - 1, k1)

    t0 = float(times[k0])
    t1 = float(times[k1])
    if abs(t1 - t0) < 1e-9:
        return k0, k1, 0.0
    alpha = float((t - t0) / (t1 - t0))
    alpha = max(0.0, min(1.0, alpha))
    return k0, k1, alpha


# ------------------ Main ------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--model", default="models/shot_lstm.pt")
    ap.add_argument("--cls", default="all", choices=["cover_drive", "pull_shot", "all"])
    ap.add_argument("--out_dir", default="data/segments")

    ap.add_argument("--det_model", default="models/batter_detector.pt")
    ap.add_argument("--det_conf", type=float, default=0.35)
    ap.add_argument("--det_iou", type=float, default=0.5)
    ap.add_argument("--pad", type=float, default=0.10)

    ap.add_argument("--pose_fps", type=float, default=15.0)
    ap.add_argument("--T", type=int, default=32)
    ap.add_argument("--stride", type=int, default=4)

    ap.add_argument("--entropy_ratio", type=float, default=0.70)

    ap.add_argument("--pad_pre_s", type=float, default=0.4)   # kept for compatibility
    ap.add_argument("--pad_post_s", type=float, default=0.6)  # kept for compatibility

    ap.add_argument("--debug_out", default="")
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        raise SystemExit(f"Video not found: {video_path}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.debug_out:
        Path(args.debug_out).parent.mkdir(parents=True, exist_ok=True)

    # ---------------- Hard requirements ----------------
    COVER_THR = 0.75
    PULL_THR  = 0.80
    BATTER_MIN = 0.50

    # Export policy (your requested logic)
    EXPORT_PAD_S = 1.0     # +1 second before/after
    MAX_GAP_S    = 0.35    # merge short holes in the mask
    MIN_LEN_S    = 0.10    # keep short segments (fast shots)

    # Load models
    det = YOLO(args.det_model)

    ckpt = torch.load(args.model, map_location="cpu")
    net = BiLSTMShot()
    net.load_state_dict(ckpt["state_dict"])
    net.eval()
    T = int(ckpt.get("T", args.T))

    # ---------------- PASS 1: Sampled YOLO + Pose ----------------
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit("Could not open video.")

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    step = max(1, int(round(src_fps / args.pose_fps)))

    pose_solver = mp_pose.Pose(static_image_mode=False, model_complexity=1, enable_segmentation=False)

    times = []
    bboxes = []
    bconfs = []
    poses = []

    frame_idx = -1
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1
        if frame_idx % step != 0:
            continue

        t_sec = frame_idx / src_fps
        times.append(t_sec)

        det_res = det.predict(frame, conf=args.det_conf, iou=args.det_iou, verbose=False)
        box = pick_top1_box(det_res, W, H, args.det_conf)

        lm_full = np.zeros((33, 3), dtype=np.float32)

        if box is None:
            bboxes.append(None)
            bconfs.append(0.0)
            poses.append(lm_full)
            continue

        x1, y1, x2, y2, conf = box
        bconfs.append(conf)

        # pad bbox for pose
        bw = x2 - x1
        bh = y2 - y1
        x1 = x1 - args.pad * bw
        x2 = x2 + args.pad * bw
        y1 = y1 - args.pad * bh
        y2 = y2 + args.pad * bh
        x1, y1, x2, y2 = clamp_box(x1, y1, x2, y2, W, H)
        bboxes.append((x1, y1, x2, y2))

        roi = frame[y1:y2, x1:x2].copy()
        if roi.size > 0:
            roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            pres = pose_solver.process(roi_rgb)
            if pres.pose_landmarks:
                roi_w = max(1, (x2 - x1))
                roi_h = max(1, (y2 - y1))
                for i, p in enumerate(pres.pose_landmarks.landmark[:33]):
                    lm_full[i, 0] = (x1 + p.x * roi_w) / W
                    lm_full[i, 1] = (y1 + p.y * roi_h) / H
                    lm_full[i, 2] = p.visibility

        poses.append(lm_full)

    cap.release()
    pose_solver.close()

    if len(poses) < T:
        print("[INFO] Not enough sampled frames for classification. Exiting.")
        return

    times = np.array(times, dtype=np.float32)
    pose_arr = np.stack(poses, axis=0).astype(np.float32)
    bconfs_arr = np.array(bconfs, dtype=np.float32)
    S = pose_arr.shape[0]

    # ---------------- CLASSIFY: probs timeline ----------------
    probs_timeline = np.zeros((S, 3), dtype=np.float32)
    counts = np.zeros((S,), dtype=np.int32)

    H_max = np.log(len(CLASSES))
    H_thr = args.entropy_ratio * H_max

    for i in range(0, S - T + 1, args.stride):
        win = pose_arr[i:i + T]

        win_n = np.stack([norm_frame(f) for f in win], axis=0)
        win5 = add_velocity(win_n)

        win_m = mirror_sequence(win)
        win_mn = np.stack([norm_frame(f) for f in win_m], axis=0)
        win_m5 = add_velocity(win_mn)

        xA = torch.tensor(win5[None], dtype=torch.float32)
        xB = torch.tensor(win_m5[None], dtype=torch.float32)

        with torch.no_grad():
            pA = softmax_np(net(xA).numpy().squeeze())
            pB = softmax_np(net(xB).numpy().squeeze())

        p = np.maximum(pA, pB)

        # entropy filter to reject uncertain windows
        if entropy(p) > H_thr:
            continue

        for j in range(i, i + T):
            probs_timeline[j] += p
            counts[j] += 1

    # normalize; default to "other" if never covered by any window
    for k in range(S):
        if counts[k] > 0:
            probs_timeline[k] /= counts[k]
        else:
            probs_timeline[k] = np.array([0.0, 0.0, 1.0], dtype=np.float32)

    # ---------------- Gate by batter confidence ----------------
    valid_mask = (bconfs_arr >= BATTER_MIN)
    probs_use = probs_timeline.copy()
    probs_use[~valid_mask] = np.array([0.0, 0.0, 1.0], dtype=np.float32)

    # ---------------- EXPORT (NEW: boolean timeline trim) ----------------
    want_cover = args.cls in ("cover_drive", "all")
    want_pull  = args.cls in ("pull_shot", "all")

    # Build masks only where batter exists (valid_mask)
    total = 0

    if want_cover:
        cover_mask = (probs_use[:, 0] >= COVER_THR) & valid_mask
        cover_segs = mask_to_segments(times, cover_mask, max_gap_s=MAX_GAP_S, min_len_s=MIN_LEN_S)
        total += export_segments_ffmpeg(video_path, times, cover_segs, out_dir, video_path.stem, "cover_drive", pad_s=EXPORT_PAD_S)

    if want_pull:
        pull_mask = (probs_use[:, 1] >= PULL_THR) & valid_mask
        pull_segs = mask_to_segments(times, pull_mask, max_gap_s=MAX_GAP_S, min_len_s=MIN_LEN_S)
        total += export_segments_ffmpeg(video_path, times, pull_segs, out_dir, video_path.stem, "pull_shot", pad_s=EXPORT_PAD_S)

    if total == 0:
        print(f"[RESULT] No '{args.cls}' segments found (after threshold mask).")
    else:
        print(f"[RESULT] Exported {total} segment(s) to {out_dir}")

    # ---------------- PASS 2: Debug overlay (interpolated) ----------------
    if not args.debug_out:
        return

    cap2 = cv2.VideoCapture(str(video_path))
    if not cap2.isOpened():
        print("[WARN] Could not reopen video for debug overlay.")
        return

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    dbg_writer = cv2.VideoWriter(args.debug_out, fourcc, src_fps, (W, H))

    frame_idx = -1
    while True:
        ok, frame = cap2.read()
        if not ok:
            break
        frame_idx += 1
        t = frame_idx / src_fps

        overlay = frame.copy()

        k0, k1, alpha = find_sample_neighbors(times, t)

        bb = interp_bbox(bboxes[k0], bboxes[k1], alpha)
        bc = float(lerp(np.array([bconfs_arr[k0]]), np.array([bconfs_arr[k1]]), alpha)[0])

        lm = interp_pose(pose_arr[k0], pose_arr[k1], alpha)
        pr = interp_probs(probs_use[k0], probs_use[k1], alpha)
        pred = int(np.argmax(pr)) if pr is not None else 2

        # draw bbox
        if bb is not None:
            x1, y1, x2, y2 = bb
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                overlay,
                f"batter_conf {bc:.2f}",
                (x1, max(0, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

        # draw skeleton
        if lm is not None:
            draw_skeleton_fullframe(overlay, lm, color_pts=(0, 0, 255), color_lines=(255, 255, 255), vis_thr=0.5)

        # draw probs
        if pr is not None:
            lines = [
                f"batter gate >= {BATTER_MIN:.2f}  (now {bc:.2f})",
                "Shot probs (gated):",
                f"cover_drive: {pr[0]:.2f} (thr {COVER_THR:.2f})",
                f"pull_shot : {pr[1]:.2f} (thr {PULL_THR:.2f})",
                f"other     : {pr[2]:.2f}",
                f"PRED: {CLASSES[pred]}",
            ]
            draw_label_box(overlay, lines, x=10, y=10)

        dbg_writer.write(overlay)

        if args.show:
            cv2.imshow("Debug Overlay (Interpolated)", overlay)
            if cv2.waitKey(1) & 0xFF == 27:
                break

    cap2.release()
    dbg_writer.release()
    if args.show:
        cv2.destroyAllWindows()

    print(f"[OK] Debug overlay written to: {args.debug_out}")


if __name__ == "__main__":
    main()

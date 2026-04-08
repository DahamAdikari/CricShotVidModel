import argparse, cv2, numpy as np
from pathlib import Path
CLS = ["cover_drive","pull_shot","other"]
EDGES = [(11,12),(11,13),(13,15),(12,14),(14,16),
         (11,23),(12,24),(23,24),(23,25),(25,27),(27,29),(24,26),(26,28),(28,30),
         (15,21),(16,22)]

def draw(frame, lm, vis_thr=0.35):
    h,w = frame.shape[:2]
    pts = (lm[:,:2]*[w,h]).astype(int); vis = lm[:,2]>vis_thr
    for a,b in EDGES:
        if vis[a] and vis[b]:
            cv2.line(frame, tuple(pts[a]), tuple(pts[b]), (255,255,255), 2)
    for i,(x,y) in enumerate(pts):
        if vis[i]: cv2.circle(frame,(x,y),3,(0,255,0),-1)

def find_crop(stem, crops_dir):
    # tries common names: <stem>_batter.mp4 or stem.mp4
    p1 = Path(crops_dir,f"{stem}_batter.mp4")
    p2 = Path(crops_dir,f"{stem}.mp4")
    return p1 if p1.exists() else (p2 if p2.exists() else None)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/pose/train", help="pose root with class subfolders")
    ap.add_argument("--crops", default="data/crops", help="folder with cropped batter mp4s")
    ap.add_argument("--out", default="data/samples/audit_viz", help="output videos")
    ap.add_argument("--per_class", type=int, default=5, help="#samples per class")
    ap.add_argument("--dur", type=int, default=64, help="frames to render per sample")
    args = ap.parse_args()
    Path(args.out).mkdir(parents=True, exist_ok=True)

    for c in CLS:
        npy_list = sorted(Path(args.root,c).glob("*.npy"))[:args.per_class]
        print(f"[{c}] rendering {len(npy_list)} samples")
        for npy in npy_list:
            stem = npy.stem.replace("_pose","")
            crop = find_crop(stem, args.crops)
            if crop is None:
                print(f"  [warn] crop not found for {npy.name}")
                continue
            pose = np.load(npy)  # (T,33,3)
            cap = cv2.VideoCapture(str(crop))
            fps = cap.get(cv2.CAP_PROP_FPS) or 12
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); h=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            outp = Path(args.out,f"{c}_{stem}_viz.mp4")
            wr = cv2.VideoWriter(str(outp), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w,h))
            t=0; n=min(args.dur, len(pose))
            while t<n:
                ok, frame = cap.read()
                if not ok: break
                overlay = frame.copy()
                draw(overlay, pose[t])
                frame = cv2.addWeighted(frame,0.6,overlay,0.4,0)
                cv2.putText(frame,f"{c} | {t+1}/{len(pose)}",(10,26),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2,cv2.LINE_AA)
                wr.write(frame); t+=1
            wr.release(); cap.release()
            print("  wrote", outp)

if __name__ == "__main__":
    main()

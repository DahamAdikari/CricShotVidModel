import argparse, json, numpy as np
from pathlib import Path

CLS = ["cover_drive","pull_shot","other"]

def mean_speed(seq, keys=(11,12,15,16)):
    v=0.0;n=0
    for t in range(1,len(seq)):
        for k in keys:
            v += np.linalg.norm(seq[t][k,:2]-seq[t-1][k,:2])
            n += 1
    return v/max(1,n)

def vis_ratio(seq, thr=0.35):
    return float((seq[:,:,2] > thr).mean())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/pose/train", help="root with class subfolders")
    ap.add_argument("--csv", default="data/pose_audit.csv", help="write metrics CSV")
    args = ap.parse_args()

    root = Path(args.root)
    rows = []
    print("\n[Audit] scanning:", root)
    total=0
    per_class = {c:0 for c in CLS}
    lens = {c:[] for c in CLS}
    speeds = {c:[] for c in CLS}
    visrs  = {c:[] for c in CLS}

    for c in CLS:
        for f in sorted((root/c).glob("*.npy")):
            try:
                x = np.load(f)  # (T,33,3)
                T = len(x)
                per_class[c]+=1; total+=1
                lens[c].append(T)
                speeds[c].append(mean_speed(x))
                visrs[c].append(vis_ratio(x))
                rows.append((c, f.name, T, speeds[c][-1], visrs[c][-1]))
            except Exception as e:
                print(f"[ERR] {f}: {e}")

    print("\n[Summary]")
    for c in CLS:
        if per_class[c]==0:
            print(f"  {c:12}: 0")
            continue
        print(f"  {c:12}: {per_class[c]} clips | len μ={np.mean(lens[c]):.1f}±{np.std(lens[c]):.1f} | speed μ={np.mean(speeds[c]):.4f} | vis μ={np.mean(visrs[c]):.2f}")

    Path(args.csv).parent.mkdir(parents=True, exist_ok=True)
    with open(args.csv,"w",encoding="utf-8") as f:
        f.write("class,file,T,mean_speed,vis_ratio\n")
        for r in rows: f.write(",".join([str(x) for x in r])+"\n")
    print(f"\n[OK] wrote metrics: {args.csv}")

    # flag potential issues
    print("\n[Heuristics]")
    for c in CLS:
        if per_class[c]==0: continue
        low_vis = sum(v<0.25 for v in visrs[c])
        static  = sum(s<0.002 for s in speeds[c])
        short   = sum(L<8 for L in lens[c])
        print(f"  {c:12}: low-vis={low_vis} static={static} very-short={short}")

if __name__ == "__main__":
    main()


import argparse, cv2, numpy as np, mediapipe as mp
from pathlib import Path
mp_pose = mp.solutions.pose
CONNECTIONS = mp_pose.POSE_CONNECTIONS

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('video')
    ap.add_argument('--pose', required=True)
    ap.add_argument('--out', default='data/samples/pose_vis.mp4')
    args = ap.parse_args()

    pose = np.load(args.pose)  # (T,33,3)
    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 12
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); h=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w,h))

    t = 0
    while True:
        ret, frame = cap.read()
        if not ret or t>=len(pose): break
        lm = pose[t]; t += 1
        pts = (lm[:,:2] * [w,h]).astype(int); vis = lm[:,2] > 0.5
        for i,(x,y) in enumerate(pts):
            if vis[i]: cv2.circle(frame,(x,y),3,(0,255,0),-1)
        for a,b in CONNECTIONS:
            if vis[a] and vis[b]:
                cv2.line(frame, tuple(pts[a]), tuple(pts[b]), (255,255,255), 2)
        writer.write(frame)

    writer.release(); cap.release()
    print(f"Wrote {args.out}")

if __name__ == '__main__':
    main()

import cv2
import tkinter as tk
from tkinter import filedialog
from ultralytics import YOLO
from PIL import Image, ImageTk

class YOLODebugGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("YOLO Batter Detection Debugger")

        self.model = YOLO("models/batter_detector.pt")
        self.cap = None
        self.playing = False
        self.conf = 0.35
        self.iou = 0.5

        self.label = tk.Label(root)
        self.label.pack()

        btn_frame = tk.Frame(root)
        btn_frame.pack()

        tk.Button(btn_frame, text="Open Video", command=self.open_video).pack(side=tk.LEFT)
        tk.Button(btn_frame, text="Play", command=self.play).pack(side=tk.LEFT)
        tk.Button(btn_frame, text="Pause", command=self.pause).pack(side=tk.LEFT)

    def open_video(self):
        path = filedialog.askopenfilename(filetypes=[("MP4 files", "*.mp4")])
        if not path:
            return
        self.cap = cv2.VideoCapture(path)

    def play(self):
        self.playing = True
        self.update_frame()

    def pause(self):
        self.playing = False

    def update_frame(self):
        if not self.playing or self.cap is None:
            return

        ret, frame = self.cap.read()
        if not ret:
            self.playing = False
            return

        results = self.model.predict(frame, conf=self.conf, iou=self.iou, verbose=False)

        if results and results[0].boxes is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            confs = results[0].boxes.conf.cpu().numpy()

            if len(boxes) > 0:
                idx = confs.argmax()
                x1, y1, x2, y2 = map(int, boxes[idx])
                cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
                cv2.putText(frame, f"Batter {confs[idx]:.2f}",
                            (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (0,255,0), 2)

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = ImageTk.PhotoImage(Image.fromarray(frame))
        self.label.config(image=img)
        self.label.image = img

        self.root.after(30, self.update_frame)

if __name__ == "__main__":
    root = tk.Tk()
    YOLODebugGUI(root)
    root.mainloop()

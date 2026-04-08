import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import ttkbootstrap as tb
import threading, subprocess, sys, shlex
from pathlib import Path

PROJ_ROOT = Path(__file__).resolve().parent.parent

def which_python():
    return sys.executable

class ShotExtractorUI(tb.Window):
    def __init__(self):
        super().__init__(themename="darkly")
        self.title("Cricket Shot Extractor 🎯")
        self.geometry("880x600")
        self.resizable(False, False)
        self.proc = None
        self._build_menu()
        self._build_ui()

    # ---------------- Menu ----------------
    def _build_menu(self):
        menubar = tk.Menu(self)
        filem = tk.Menu(menubar, tearoff=0)
        filem.add_command(label="Open Video…", command=self.pick_video)
        filem.add_command(label="Set Export Folder…", command=self.pick_output)
        filem.add_separator()
        filem.add_command(label="Exit", command=self.destroy)
        menubar.add_cascade(label="File", menu=filem)
        self.config(menu=menubar)

    # --------------- UI -------------------
    def _build_ui(self):
        pad = {"padx": 14, "pady": 10}

        # Header
        header = ttk.Label(self, text="🏏  Cricket Shot Extractor", font=("Segoe UI", 20, "bold"))
        header.pack(fill="x", pady=(16, 10))

        # Big pickers row
        pickers = ttk.Frame(self)
        pickers.pack(fill="x", **pad)

        # Video picker card
        video_card = ttk.Labelframe(pickers, text=" Full Match Video ")
        video_card.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        self.video_var = tk.StringVar()
        ttk.Button(
            video_card, text="📂 Select Video…", bootstyle="primary-outline", command=self.pick_video
        ).pack(fill="x", padx=12, pady=(16, 8))
        self.video_display = ttk.Label(video_card, text="(none selected)", wraplength=380, justify="left")
        self.video_display.pack(fill="x", padx=12, pady=(0, 16))

        # Export picker card
        out_card = ttk.Labelframe(pickers, text=" Export Folder ")
        out_card.grid(row=0, column=1, sticky="nsew", padx=(10, 0))
        self.out_var = tk.StringVar(value=str(PROJ_ROOT / "data" / "segments"))
        ttk.Button(
            out_card, text="📁 Select Folder…", bootstyle="primary-outline", command=self.pick_output
        ).pack(fill="x", padx=12, pady=(16, 8))
        self.out_display = ttk.Label(out_card, text=self.out_var.get(), wraplength=380, justify="left")
        self.out_display.pack(fill="x", padx=12, pady=(0, 16))

        pickers.columnconfigure(0, weight=1)
        pickers.columnconfigure(1, weight=1)

        # Options row
        opts = ttk.Frame(self)
        opts.pack(fill="x", **pad)

        ttk.Label(opts, text="Shot Type:", font=("Segoe UI", 11)).grid(row=0, column=0, sticky="w")
        self.cls_var = tk.StringVar(value="pull_shot")
        ttk.Combobox(
            opts, textvariable=self.cls_var, state="readonly",
            values=["cover_drive", "pull_shot", "all"], width=18
        ).grid(row=0, column=1, sticky="w", padx=(8, 18))

        self.debug_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(opts, text="Save Debug Crop Video", variable=self.debug_var).grid(row=0, column=2, padx=(0, 18))

        ttk.Label(opts, text="Detector conf:", font=("Segoe UI", 11)).grid(row=0, column=3, sticky="e")
        self.conf_var = tk.StringVar(value="0.45")
        ttk.Entry(opts, textvariable=self.conf_var, width=7).grid(row=0, column=4, sticky="w", padx=(6, 18))

        ttk.Label(opts, text="IoU:", font=("Segoe UI", 11)).grid(row=0, column=5, sticky="e")
        self.iou_var = tk.StringVar(value="0.5")
        ttk.Entry(opts, textvariable=self.iou_var, width=7).grid(row=0, column=6, sticky="w", padx=(6, 0))

        # Run/Stop + progress
        ctrl = ttk.Frame(self)
        ctrl.pack(fill="x", pady=(6, 0), padx=14)

        self.run_btn = ttk.Button(ctrl, text="▶ Run Detection", bootstyle="success", command=self.run, state="disabled")
        self.run_btn.grid(row=0, column=0, padx=(0, 10))
        self.stop_btn = ttk.Button(ctrl, text="⛔ Stop", bootstyle="danger", command=self.stop, state="disabled")
        self.stop_btn.grid(row=0, column=1)

        self.pb = ttk.Progressbar(ctrl, mode="indeterminate", bootstyle="info-striped")
        self.pb.grid(row=0, column=2, sticky="ew", padx=(16, 0))
        ctrl.columnconfigure(2, weight=1)

        self.status_lbl = ttk.Label(self, text="Choose a video and an export folder to begin.", font=("Segoe UI", 10))
        self.status_lbl.pack(fill="x", padx=16, pady=(8, 8))

        # Output log (cleaner, not raw spam—but still useful)
        self.text = tk.Text(self, height=14, wrap="word", font=("Consolas", 10))
        self.text.pack(fill="both", expand=True, padx=16, pady=(0, 14))
        self.text.insert("end", "Ready.\n")
        self.text.configure(state="disabled")

        self._update_run_state()

    # ---------------- Helpers ----------------
    def _update_run_state(self):
        video_ok = Path(self.video_var.get()).exists() if self.video_var.get() else False
        out_ok = bool(self.out_var.get())
        self.run_btn.configure(state=("normal" if (video_ok and out_ok and self.proc is None) else "disabled"))
        self.video_display.configure(text=(self.video_var.get() or "(none selected)"))
        self.out_display.configure(text=(self.out_var.get() or "(none selected)"))

    def append(self, msg):
        self.text.configure(state="normal")
        self.text.insert("end", msg)
        self.text.see("end")
        self.text.configure(state="disabled")

    def set_status(self, msg):
        self.status_lbl.configure(text=msg)
        self.update_idletasks()

    # ---------------- Browse actions ----------------
    def pick_video(self):
        p = filedialog.askopenfilename(
            title="Select Full Match Video",
            filetypes=[("MP4 Files", "*.mp4"), ("All Files", "*.*")]
        )
        if p:
            self.video_var.set(p)
            self._update_run_state()

    def pick_output(self):
        d = filedialog.askdirectory(title="Select Export Folder")
        if d:
            self.out_var.set(d)
            self._update_run_state()

    # ---------------- Run/Stop ----------------
    def run(self):
        if self.proc:
            messagebox.showwarning("Busy", "Detection is already running.")
            return

        video = self.video_var.get().strip()
        outdir = self.out_var.get().strip()
        cls = self.cls_var.get().strip()
        debug = self.debug_var.get()
        conf = self.conf_var.get().strip()
        iou = self.iou_var.get().strip()

        vpath = Path(video)
        if not vpath.exists():
            messagebox.showerror("Error", "Please select a valid video file.")
            return
        Path(outdir).mkdir(parents=True, exist_ok=True)

        # Build the command to call your existing pipeline
        cmd = [
            which_python(), "-m", "scripts.detect_shots", video,
            "--model", "models/shot_lstm.pt",
            "--cls", cls,
            "--out_dir", outdir,
            "--det_model", "models/batter_detector.pt",
            "--config", "src/config.yaml",
            "--conf", conf,
            "--iou", iou
        ]
        if debug:
            cmd.append("--debug_crop")

        # Lock UI
        self.run_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")
        self.pb.start()
        self.set_status("Processing… This may take a few minutes ⏳")
        self.append(f"\n▶ Running: {vpath.name}  |  Shot: {cls}\n")

        def worker():
            try:
                self.proc = subprocess.Popen(
                    cmd, cwd=str(PROJ_ROOT),
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
                )
                # Show only key lines to keep it readable
                for line in self.proc.stdout:
                    if any(tag in line for tag in (
                        "Exported", "[RESULT]", "[INFO]", "Batter detected", "frames", "Saved", "Completed"
                    )):
                        self.append(line)
                self.proc.wait()
                rc = self.proc.returncode
                if rc == 0:
                    self.set_status("✅ Completed successfully.")
                    self.append("✅ Done! Check the export folder.\n")
                else:
                    self.set_status("⚠️ Finished with errors (see console).")
                    self.append(f"⚠️ Exit code: {rc}\n")
            except Exception as e:
                self.append(f"❌ Error: {e}\n")
                self.set_status("❌ Error occurred.")
            finally:
                self.proc = None
                self.pb.stop()
                self.stop_btn.configure(state="disabled")
                self._update_run_state()

        threading.Thread(target=worker, daemon=True).start()

    def stop(self):
        if self.proc:
            try:
                self.proc.terminate()
                self.append("\n🛑 Stopped by user.\n")
                self.set_status("🛑 Stopped.")
            except Exception as e:
                self.append(f"\n[Error stopping] {e}\n")
            finally:
                self.proc = None
                self.pb.stop()
                self.stop_btn.configure(state="disabled")
                self._update_run_state()

# ---------------- entry ----------------
if __name__ == "__main__":
    app = ShotExtractorUI()
    app.mainloop()

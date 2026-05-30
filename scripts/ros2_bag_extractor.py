#!/usr/bin/env python3
"""
ROS2 Bag Image Extractor
Extract image frames or video from a topic in a ROS2 bag file.
Supports both .mcap and .db3 (rosbags) formats.

Dependencies:
    pip install rosbags opencv-python-headless pillow tkinter
    # For mcap support: pip install mcap mcap-ros2-support
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import os
import queue
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional


# ── Lazy imports (reported to user if missing) ──────────────────────────────

def _require(pkg: str, install: str = ""):
    try:
        return __import__(pkg)
    except ImportError:
        install_hint = install or pkg
        messagebox.showerror(
            "Missing dependency",
            f"Package '{pkg}' is not installed.\n\nRun:\n  pip install {install_hint}",
        )
        return None


# ── Data ────────────────────────────────────────────────────────────────────

@dataclass
class ExtractionConfig:
    bag_path: str = ""
    topic: str = ""
    output_dir: str = ""
    mode: str = "frames"          # "frames" | "video"
    image_format: str = "png"     # png | jpg | bmp
    fps: float = 30.0
    frame_skip: int = 0           # 0 = keep every frame, N = keep 1 in N+1
    start_time: float = 0.0       # seconds from bag start (0 = from beginning)
    end_time: float = 0.0         # seconds from bag start (0 = until end)


# ── ROS2 bag reader (rosbags library) ────────────────────────────────────────

def _make_typestore():
    """Return (typestore_or_None) for the newest available rosbags API."""
    try:
        from rosbags.typesys import Stores, get_typestore
        for name in ("ROS2_HUMBLE", "ROS2_IRON", "ROS2_JAZZY", "ROS2_FOXY", "ROS2_GALACTIC"):
            try:
                return get_typestore(getattr(Stores, name))
            except Exception:
                continue
    except ImportError:
        pass
    return None



# Known direct image msgtypes (checked by substring)
_IMAGE_MSGTYPES = ("sensor_msgs/msg/Image", "sensor_msgs/msg/CompressedImage",
                   "sensor_msgs/Image", "sensor_msgs/CompressedImage")


def _is_direct_image_type(msgtype: str) -> bool:
    return any(t in msgtype for t in ("sensor_msgs/msg/Image",
                                      "sensor_msgs/msg/CompressedImage",
                                      "sensor_msgs/Image",
                                      "sensor_msgs/CompressedImage"))


def _find_image_field(msg) -> tuple[object | None, str]:
    """
    Recursively search a deserialized message for the first field that looks
    like a sensor_msgs Image or CompressedImage.
    Returns (sub_message, effective_msgtype_hint) or (None, "").
    """
    if msg is None:
        return None, ""
    # Check direct attributes that look like image fields
    for attr in vars(msg) if hasattr(msg, "__dict__") else []:
        val = getattr(msg, attr, None)
        if val is None:
            continue
        t = type(val).__name__
        # rosbags names the class after the last segment of the msgtype
        if t in ("Image", "CompressedImage"):
            return val, t
        # Recurse one level (avoid infinite loops on primitives)
        if hasattr(val, "__dict__") and not isinstance(val, (str, bytes, bytearray)):
            sub, hint = _find_image_field(val)
            if sub is not None:
                return sub, hint
    return None, ""


def list_image_topics(bag_path: str) -> list[tuple[str, str]]:
    """
    Return [(topic, msg_type), ...] for topics that contain or ARE an image.
    Includes custom message types that embed a sensor_msgs Image field.
    """
    rosbags = _require("rosbags", "rosbags[ros2]")
    if rosbags is None:
        return []

    from rosbags.rosbag2 import Reader
    topics = []
    try:
        typestore = _make_typestore()
        deserialize_cdr = None
        if typestore is not None:
            deserialize_cdr = typestore.deserialize_cdr
        else:
            try:
                from rosbags.serde import deserialize_cdr as _d
                deserialize_cdr = _d
            except ImportError:
                pass

        with Reader(bag_path) as reader:
            # Collect all unique connections (one per topic+type pair)
            seen: set[str] = set()
            for conn in reader.connections:
                if conn.topic in seen:
                    continue
                # Direct image type — always include
                if _is_direct_image_type(conn.msgtype):
                    topics.append((conn.topic, conn.msgtype))
                    seen.add(conn.topic)
                    continue
                # Unknown type — peek at the first message to check for image fields
                if deserialize_cdr is None:
                    continue
                try:
                    for _, _, raw in reader.messages(connections=[conn]):
                        msg = deserialize_cdr(raw, conn.msgtype)
                        sub, hint = _find_image_field(msg)
                        if sub is not None:
                            # Tag it so the extractor knows which field to use
                            topics.append((conn.topic, f"{conn.msgtype}[image_field:{hint}]"))
                            seen.add(conn.topic)
                        break  # only need the first message
                except Exception:
                    pass
    except Exception as exc:
        messagebox.showerror("Read error", str(exc))
    return topics



def _get_deserializer():
    """Return a deserialize_cdr callable, compatible with old and new rosbags APIs."""
    # rosbags < 0.9  (legacy top-level function)
    try:
        from rosbags.serde import deserialize_cdr
        return deserialize_cdr, None
    except ImportError:
        pass

    # rosbags >= 0.9 — typestore-based
    ts = _make_typestore()
    if ts is not None:
        return ts.deserialize_cdr, ts

    raise ImportError(
        "Could not find a deserialize_cdr function in rosbags. "
        "Try: pip install --upgrade 'rosbags[ros2]'"
    )


def extract_frames(cfg: ExtractionConfig, progress_q: queue.Queue, stop_event: threading.Event):
    """Worker: extract frames or build video. Puts progress dicts into progress_q."""
    try:
        import cv2
        import numpy as np
        from rosbags.rosbag2 import Reader
        deserialize_cdr, typestore = _get_deserializer()
    except ImportError as exc:
        progress_q.put({"error": str(exc)})
        return

    bag_path = Path(cfg.bag_path)
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    video_writer: Optional[cv2.VideoWriter] = None
    frame_idx = 0
    saved = 0
    first_stamp: Optional[int] = None

    try:
        with Reader(str(bag_path)) as reader:
            # Find connections matching the chosen topic
            connections = [c for c in reader.connections if c.topic == cfg.topic]
            if not connections:
                progress_q.put({"error": f"Topic '{cfg.topic}' not found in bag."})
                return

            # Count total messages for progress bar
            # Note: rosbags Reader doesn't have statistics attribute
            # We'll count messages as we process them
            total = None

            for conn, stamp_ns, raw in reader.messages(connections=connections):
                if stop_event.is_set():
                    progress_q.put({"status": "Cancelled."})
                    break

                # Time filtering
                if first_stamp is None:
                    first_stamp = stamp_ns
                elapsed_s = (stamp_ns - first_stamp) / 1e9

                if cfg.start_time > 0 and elapsed_s < cfg.start_time:
                    continue
                if cfg.end_time > 0 and elapsed_s > cfg.end_time:
                    break

                # Frame skip
                if cfg.frame_skip > 0 and (frame_idx % (cfg.frame_skip + 1)) != 0:
                    frame_idx += 1
                    continue

                # Deserialise
                msg = deserialize_cdr(raw, conn.msgtype)
                img = _msg_to_bgr(msg, conn.msgtype)
                if img is None:
                    frame_idx += 1
                    continue

                if cfg.mode == "frames":
                    fname = out_dir / f"frame_{saved:06d}.{cfg.image_format}"
                    cv2.imwrite(str(fname), img)
                else:
                    if video_writer is None:
                        h, w = img.shape[:2]
                        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                        video_path = out_dir / "output.mp4"
                        video_writer = cv2.VideoWriter(str(video_path), fourcc, cfg.fps, (w, h))
                    video_writer.write(img)

                saved += 1
                frame_idx += 1

                pct = int(frame_idx / total * 100) if total else -1
                progress_q.put({"saved": saved, "pct": pct, "status": f"Extracted {saved} frames…"})

    except Exception as exc:
        progress_q.put({"error": str(exc)})
        return
    finally:
        if video_writer is not None:
            video_writer.release()

    progress_q.put({"done": True, "saved": saved, "out_dir": str(out_dir)})


def _decode_image_msg(img_msg, is_compressed: bool):
    """Convert a sensor_msgs Image or CompressedImage object to a BGR ndarray."""
    import cv2
    import numpy as np

    if is_compressed:
        buf = np.frombuffer(img_msg.data, dtype=np.uint8)
        return cv2.imdecode(buf, cv2.IMREAD_COLOR)

    enc = img_msg.encoding.lower()
    data = np.frombuffer(bytes(img_msg.data), dtype=np.uint8)
    h, w = img_msg.height, img_msg.width

    if enc in ("rgb8", "bgr8", "mono8"):
        channels = 1 if enc == "mono8" else 3
        img = data.reshape((h, w, channels))
        if enc == "rgb8":
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        elif enc == "mono8":
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif enc == "16uc1":
        img = data.view(np.uint16).reshape((h, w))
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif enc in ("bayer_rggb8", "bayer_bggr8", "bayer_gbrg8", "bayer_grbg8"):
        img = data.reshape((h, w))
        img = cv2.cvtColor(img, cv2.COLOR_BayerRG2BGR)
    else:
        img = data.reshape((h, w, -1))
    return img


def _msg_to_bgr(msg, msgtype: str):
    """
    Convert any ROS message to a BGR numpy array.
    Handles:
      - Direct sensor_msgs/Image
      - Direct sensor_msgs/CompressedImage
      - Custom messages with an embedded Image or CompressedImage field
        (tagged in msgtype as "...[image_field:Image]" or "...[image_field:CompressedImage]")
    """
    try:
        import cv2
        import numpy as np

        # Check for embedded-field tag added by list_image_topics
        if "[image_field:" in msgtype:
            hint = msgtype.split("[image_field:")[-1].rstrip("]")
            is_compressed = "Compressed" in hint
            sub, _ = _find_image_field(msg)
            if sub is None:
                return None
            return _decode_image_msg(sub, is_compressed)

        # Direct compressed image
        if "CompressedImage" in msgtype:
            return _decode_image_msg(msg, is_compressed=True)

        # Direct uncompressed image
        return _decode_image_msg(msg, is_compressed=False)

    except Exception:
        return None


# ── GUI ──────────────────────────────────────────────────────────────────────

BG      = "#f5f5f0"   # warm off-white page
SURFACE = "#ffffff"   # card / input surface
BORDER  = "#ddddd8"   # subtle divider
ACCENT  = "#2563eb"   # blue action colour
ACCENT2 = "#1d4ed8"   # hover
TEXT    = "#1a1a1a"   # primary text
MUTED   = "#6b7280"   # secondary text
SUCCESS = "#16a34a"

FONT_MONO = ("Courier New", 10)
FONT_BODY = ("Segoe UI", 10) if os.name == "nt" else ("Helvetica Neue", 10)
FONT_HEAD = ("Segoe UI Semibold", 11) if os.name == "nt" else ("Helvetica Neue", 11)
FONT_TINY = ("Segoe UI", 9)  if os.name == "nt" else ("Helvetica Neue", 9)


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("ROS2 Bag → Image Extractor")
        self.configure(bg=BG)
        self.resizable(True, True)
        self.minsize(640, 540)

        self.cfg = ExtractionConfig()
        self._stop_event = threading.Event()
        self._progress_q: queue.Queue = queue.Queue()
        self._worker: Optional[threading.Thread] = None

        self._build_styles()
        self._build_ui()
        self._poll_progress()

    # ── styles ───────────────────────────────────────────────────────────────

    def _build_styles(self):
        s = ttk.Style(self)
        s.theme_use("clam")

        s.configure(".", background=BG, foreground=TEXT, font=FONT_BODY,
                    troughcolor=BORDER, selectbackground=ACCENT, selectforeground="white",
                    fieldbackground=SURFACE, borderwidth=0, relief="flat")

        s.configure("TFrame", background=BG)

        s.configure("TLabel", background=BG, foreground=TEXT, font=FONT_BODY)
        s.configure("Head.TLabel", background=BG, foreground=TEXT, font=FONT_HEAD)
        s.configure("Muted.TLabel", background=BG, foreground=MUTED, font=FONT_TINY)

        s.configure("TEntry", fieldbackground=SURFACE, foreground=TEXT, insertcolor=TEXT,
                    borderwidth=1, relief="solid", bordercolor=BORDER, padding=(6, 5))
        s.map("TEntry",
              bordercolor=[("focus", ACCENT)],
              fieldbackground=[("focus", SURFACE)])

        s.configure("TCombobox", fieldbackground=SURFACE, foreground=TEXT,
                    selectbackground=SURFACE, selectforeground=TEXT,
                    arrowcolor=MUTED, bordercolor=BORDER, padding=(6, 5))
        s.map("TCombobox",
              fieldbackground=[("readonly", SURFACE)],
              bordercolor=[("focus", ACCENT)])

        # Default button — outlined style
        s.configure("TButton", background=SURFACE, foreground=ACCENT,
                    padding=(12, 6), font=FONT_BODY, borderwidth=1,
                    relief="solid", bordercolor=BORDER)
        s.map("TButton",
              background=[("active", "#eff6ff"), ("disabled", BG)],
              foreground=[("disabled", BORDER)],
              bordercolor=[("active", ACCENT), ("disabled", BORDER)])

        # Primary action button — filled blue
        s.configure("Accent.TButton", background=ACCENT, foreground="white",
                    padding=(14, 7), font=FONT_HEAD, borderwidth=0, relief="flat")
        s.map("Accent.TButton",
              background=[("active", ACCENT2), ("disabled", BORDER)],
              foreground=[("disabled", "white")])

        s.configure("TProgressbar", troughcolor=BORDER, background=ACCENT,
                    borderwidth=0, thickness=4)

        s.configure("TRadiobutton", background=BG, foreground=TEXT, font=FONT_BODY)
        s.map("TRadiobutton", background=[("active", BG)])

        s.configure("TCheckbutton", background=BG, foreground=TEXT, font=FONT_BODY)
        s.map("TCheckbutton", background=[("active", BG)])

        s.configure("TSeparator", background=BORDER)

    # ── UI layout ─────────────────────────────────────────────────────────────

    def _build_ui(self):
        root = ttk.Frame(self, padding=(28, 20, 28, 20))
        root.pack(fill="both", expand=True)
        root.columnconfigure(0, weight=1)

        # ── Header ────────────────────────────────────────────────────────
        hdr = ttk.Frame(root)
        hdr.grid(row=0, column=0, sticky="ew", pady=(0, 16))
        hdr.columnconfigure(0, weight=1)
        ttk.Label(hdr, text="ROS2 Bag Image Extractor",
                  font=(FONT_HEAD[0], 15, "bold"), foreground=TEXT).grid(
            row=0, column=0, sticky="w")
        ttk.Label(hdr, text="Extract frames or video from a bag topic — no ROS2 required",
                  style="Muted.TLabel").grid(row=1, column=0, sticky="w", pady=(2, 0))

        ttk.Separator(root).grid(row=1, column=0, sticky="ew", pady=(0, 20))

        # ── Bag file ──────────────────────────────────────────────────────
        self._section(root, 2, "Bag folder")
        bag_row = ttk.Frame(root)
        bag_row.grid(row=3, column=0, sticky="ew", pady=(4, 0))
        bag_row.columnconfigure(0, weight=1)
        self._bag_var = tk.StringVar()
        ttk.Entry(bag_row, textvariable=self._bag_var, font=FONT_MONO).grid(
            row=0, column=0, sticky="ew", padx=(0, 8))
        ttk.Button(bag_row, text="Browse…", command=self._browse_bag).grid(row=0, column=1)

        # ── Topic ─────────────────────────────────────────────────────────
        self._section(root, 4, "Image topic", pady_top=14)
        topic_row = ttk.Frame(root)
        topic_row.grid(row=5, column=0, sticky="ew", pady=(4, 0))
        topic_row.columnconfigure(0, weight=1)
        self._topic_var = tk.StringVar()
        self._topic_combo = ttk.Combobox(topic_row, textvariable=self._topic_var,
                                         state="readonly", font=FONT_MONO)
        self._topic_combo.grid(row=0, column=0, sticky="ew", padx=(0, 8))
        ttk.Button(topic_row, text="Refresh", command=self._load_topics).grid(row=0, column=1)
        ttk.Label(root, text="Select a bag folder first, then click Refresh",
                  style="Muted.TLabel").grid(row=6, column=0, sticky="w", pady=(3, 0))

        # ── Output ────────────────────────────────────────────────────────
        self._section(root, 7, "Output folder", pady_top=14)
        out_row = ttk.Frame(root)
        out_row.grid(row=8, column=0, sticky="ew", pady=(4, 0))
        out_row.columnconfigure(0, weight=1)
        self._out_var = tk.StringVar()
        ttk.Entry(out_row, textvariable=self._out_var, font=FONT_MONO).grid(
            row=0, column=0, sticky="ew", padx=(0, 8))
        ttk.Button(out_row, text="Browse…", command=self._browse_out).grid(row=0, column=1)

        # ── Options ───────────────────────────────────────────────────────
        self._section(root, 9, "Options", pady_top=14)

        opts = ttk.Frame(root)
        opts.grid(row=10, column=0, sticky="ew", pady=(6, 0))
        # 6 columns: label | control | label | control | label | control
        for col in (1, 3, 5):
            opts.columnconfigure(col, weight=1)

        # Row 0: Mode / Format / FPS
        ttk.Label(opts, text="Mode", foreground=MUTED, font=FONT_TINY).grid(
            row=0, column=0, sticky="w")
        self._mode_var = tk.StringVar(value="frames")
        mode_frame = ttk.Frame(opts)
        mode_frame.grid(row=0, column=1, sticky="w", padx=(6, 20))
        ttk.Radiobutton(mode_frame, text="Frames", variable=self._mode_var,
                        value="frames", command=self._on_mode).pack(side="left", padx=(0, 10))
        ttk.Radiobutton(mode_frame, text="Video", variable=self._mode_var,
                        value="video", command=self._on_mode).pack(side="left")

        ttk.Label(opts, text="Format", foreground=MUTED, font=FONT_TINY).grid(
            row=0, column=2, sticky="w")
        self._fmt_var = tk.StringVar(value="png")
        self._fmt_combo = ttk.Combobox(opts, textvariable=self._fmt_var, width=7,
                                       values=["png", "jpg", "bmp"], state="readonly")
        self._fmt_combo.grid(row=0, column=3, sticky="w", padx=(6, 20))

        ttk.Label(opts, text="FPS", foreground=MUTED, font=FONT_TINY).grid(
            row=0, column=4, sticky="w")
        self._fps_var = tk.StringVar(value="30")
        self._fps_entry = ttk.Entry(opts, textvariable=self._fps_var, width=6)
        self._fps_entry.grid(row=0, column=5, sticky="w", padx=(6, 0))
        self._fps_entry.configure(state="disabled")

        # Row 1: Skip / Start / End
        ttk.Label(opts, text="Skip frames", foreground=MUTED, font=FONT_TINY).grid(
            row=1, column=0, sticky="w", pady=(10, 0))
        self._skip_var = tk.StringVar(value="0")
        ttk.Entry(opts, textvariable=self._skip_var, width=6).grid(
            row=1, column=1, sticky="w", padx=(6, 20), pady=(10, 0))

        ttk.Label(opts, text="Start (s)", foreground=MUTED, font=FONT_TINY).grid(
            row=1, column=2, sticky="w", pady=(10, 0))
        self._start_var = tk.StringVar(value="0")
        ttk.Entry(opts, textvariable=self._start_var, width=8).grid(
            row=1, column=3, sticky="w", padx=(6, 20), pady=(10, 0))

        ttk.Label(opts, text="End (s)", foreground=MUTED, font=FONT_TINY).grid(
            row=1, column=4, sticky="w", pady=(10, 0))
        self._end_var = tk.StringVar(value="0")
        ttk.Entry(opts, textvariable=self._end_var, width=8).grid(
            row=1, column=5, sticky="w", padx=(6, 0), pady=(10, 0))

        ttk.Label(root, text="Skip 0 = keep every frame   •   Start/End 0 = full bag",
                  style="Muted.TLabel").grid(row=11, column=0, sticky="w", pady=(4, 0))

        # ── Progress ──────────────────────────────────────────────────────
        ttk.Separator(root).grid(row=12, column=0, sticky="ew", pady=(20, 14))

        self._status_var = tk.StringVar(value="Ready")
        ttk.Label(root, textvariable=self._status_var,
                  foreground=MUTED, font=FONT_TINY).grid(row=13, column=0, sticky="w")

        self._progress = ttk.Progressbar(root, orient="horizontal",
                                         mode="determinate", maximum=100)
        self._progress.grid(row=14, column=0, sticky="ew", pady=(5, 0))

        # ── Buttons ───────────────────────────────────────────────────────
        btn_row = ttk.Frame(root)
        btn_row.grid(row=15, column=0, sticky="e", pady=(14, 0))

        self._stop_btn = ttk.Button(btn_row, text="Stop", command=self._stop,
                                    state="disabled")
        self._stop_btn.pack(side="right", padx=(8, 0))

        self._run_btn = ttk.Button(btn_row, text="Extract →", style="Accent.TButton",
                                   command=self._run)
        self._run_btn.pack(side="right")

    def _section(self, parent, row, label, pady_top=0):
        ttk.Label(parent, text=label, foreground=TEXT,
                  font=(FONT_HEAD[0], 10, "bold")).grid(
            row=row, column=0, sticky="w", pady=(pady_top, 0))

    # ── Interactions ──────────────────────────────────────────────────────────

    def _on_mode(self):
        mode = self._mode_var.get()
        self._fps_entry.configure(state="normal" if mode == "video" else "disabled")
        self._fmt_combo.configure(state="disabled" if mode == "video" else "readonly")

    def _browse_bag(self):
        path = filedialog.askdirectory(title="Select ROS2 bag folder")
        if path:
            self._bag_var.set(path)
            # Auto-set output dir
            if not self._out_var.get():
                self._out_var.set(str(Path(path).parent / "extracted"))

    def _browse_out(self):
        path = filedialog.askdirectory(title="Select output directory")
        if path:
            self._out_var.set(path)

    def _load_topics(self):
        bag_path = self._bag_var.get().strip()
        if not bag_path:
            messagebox.showwarning("No bag selected", "Please select a bag file or folder first.")
            return
        self._status_var.set("Scanning topics…")
        self.update_idletasks()
        topics = list_image_topics(bag_path)
        if not topics:
            self._status_var.set("No image topics found.")
            return
        labels = [f"{t}  [{mt.split('/')[-1]}]" for t, mt in topics]
        self._topic_map = {lbl: t for lbl, (t, _) in zip(labels, topics)}
        self._topic_combo["values"] = labels
        self._topic_combo.current(0)
        self._status_var.set(f"Found {len(topics)} image topic(s).")

    def _build_config(self) -> Optional[ExtractionConfig]:
        bag = self._bag_var.get().strip()
        out = self._out_var.get().strip()
        topic_lbl = self._topic_var.get().strip()

        if not bag:
            messagebox.showwarning("Missing", "Select a bag file/folder.")
            return None
        if not topic_lbl:
            messagebox.showwarning("Missing", "Select a topic.")
            return None
        if not out:
            messagebox.showwarning("Missing", "Select an output directory.")
            return None

        topic = getattr(self, "_topic_map", {}).get(topic_lbl, topic_lbl)

        try:
            fps = float(self._fps_var.get())
            skip = int(self._skip_var.get())
            start = float(self._start_var.get())
            end = float(self._end_var.get())
        except ValueError as exc:
            messagebox.showerror("Invalid option", str(exc))
            return None

        return ExtractionConfig(
            bag_path=bag, topic=topic, output_dir=out,
            mode=self._mode_var.get(), image_format=self._fmt_var.get(),
            fps=fps, frame_skip=skip, start_time=start, end_time=end,
        )

    def _run(self):
        cfg = self._build_config()
        if cfg is None:
            return

        self._stop_event.clear()
        self._run_btn.configure(state="disabled")
        self._stop_btn.configure(state="normal")
        self._progress["value"] = 0
        self._progress.configure(mode="indeterminate")
        self._progress.start(12)
        self._status_var.set("Starting…")

        self._worker = threading.Thread(
            target=extract_frames, args=(cfg, self._progress_q, self._stop_event), daemon=True)
        self._worker.start()

    def _stop(self):
        self._stop_event.set()
        self._status_var.set("Stopping…")
        self._stop_btn.configure(state="disabled")

    # ── Progress polling ──────────────────────────────────────────────────────

    def _poll_progress(self):
        try:
            while True:
                msg = self._progress_q.get_nowait()

                if "error" in msg:
                    self._finish()
                    messagebox.showerror("Extraction error", msg["error"])

                elif "done" in msg:
                    self._finish()
                    out = msg.get("out_dir", "")
                    saved = msg.get("saved", 0)
                    self._status_var.set(f"✓  Done — {saved} frame(s) saved to {out}")
                    self._progress.configure(mode="determinate")
                    self._progress["value"] = 100
                    messagebox.showinfo("Done", f"Extracted {saved} frame(s).\n\n→ {out}")

                elif "status" in msg:
                    self._status_var.set(msg["status"])
                    pct = msg.get("pct", -1)
                    if pct >= 0:
                        self._progress.stop()
                        self._progress.configure(mode="determinate")
                        self._progress["value"] = pct

        except queue.Empty:
            pass
        self.after(80, self._poll_progress)

    def _finish(self):
        self._progress.stop()
        self._progress.configure(mode="determinate")
        self._run_btn.configure(state="normal")
        self._stop_btn.configure(state="disabled")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app = App()
    app.mainloop()

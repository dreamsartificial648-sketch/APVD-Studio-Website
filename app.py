"""
APVD v3.0 - AI Pixel Value Determinator
Tkinter GUI for VAE-based image variation generation with mini latent diffusion.
Features: 
- Training controls (Epochs, Save/Load)
- Generation Tools (Unique, Chaos Mode, Auto-Cycle)
- Dream Cycle: Smoothly morphs between latent points using Slerp for constant velocity.
- Mini Diffusion: Iterative latent denoising for more structured generations.
- Memory evolution, latent presets, interactive breeding, and latent map recall.
- Threaded Training with Stop Functionality
"""

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from tkinter import simpledialog
from datetime import datetime
from pathlib import Path
import time
import random
import threading
import re
import math
import tarfile
import zipfile
from io import BytesIO

import torch
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFile
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from memory_system import MemoryBank, breed_latents, parse_selection_indices, summarize_memory
from model import VAE, vae_loss, latent_denoiser_loss, get_device
from model_merger import merge_checkpoints
from reconstruction_video import render_reconstruction_video
from scene_composer import LayeredScene, ParsedScene, generate_scene_from_prompt
from utils import (
    get_image_paths,
    list_image_members,
    list_model_paths,
    select_model_path_for_prompt,
    load_training_images_from_archive_entries,
    load_training_images_from_paths,
    load_training_images_from_videos,
    tensor_to_pil,
)

MAX_TRAINING_PREVIEW_IMAGES = 256
ImageFile.LOAD_TRUNCATED_IMAGES = True
PERSONALITY_PRESETS = {
    "Manual": {
        "intensity": 1.0,
        "blend": False,
        "blend_count": 2,
        "iterations": 3,
        "use_diffusion": True,
        "diffusion_steps": 8,
        "diffusion_strength": 0.85,
    },
    "Dreamy": {
        "intensity": 0.8,
        "blend": True,
        "blend_count": 4,
        "iterations": 2,
        "use_diffusion": True,
        "diffusion_steps": 10,
        "diffusion_strength": 0.55,
    },
    "Chaotic": {
        "intensity": 8.5,
        "blend": False,
        "blend_count": 2,
        "iterations": 0,
        "use_diffusion": True,
        "diffusion_steps": 14,
        "diffusion_strength": 1.25,
    },
    "Nostalgic": {
        "intensity": 0.9,
        "blend": True,
        "blend_count": 3,
        "iterations": 4,
        "use_diffusion": True,
        "diffusion_steps": 7,
        "diffusion_strength": 0.7,
    },
    "Hybrid": {
        "intensity": 2.2,
        "blend": True,
        "blend_count": 6,
        "iterations": 3,
        "use_diffusion": True,
        "diffusion_steps": 12,
        "diffusion_strength": 0.95,
    },
    "Corruption": {
        "intensity": 6.0,
        "blend": False,
        "blend_count": 2,
        "iterations": 10,
        "use_diffusion": True,
        "diffusion_steps": 18,
        "diffusion_strength": 1.35,
    },
}

def slerp(val, low, high):
    """
    Spherical linear interpolation.
    Maintains constant velocity through latent space.
    """
    # Normalize to unit vectors
    low_norm = low / torch.norm(low, dim=1, keepdim=True)
    high_norm = high / torch.norm(high, dim=1, keepdim=True)
    
    # Dot product
    dot = (low_norm * high_norm).sum(1)
    
    # Clamp for numerical stability
    dot = torch.clamp(dot, -1.0, 1.0)
    
    omega = torch.acos(dot)
    so = torch.sin(omega)
    
    # Handle cases where points are very close (division by zero)
    if torch.all(so < 1e-6):
        return (1.0 - val) * low + val * high
        
    res = (torch.sin((1.0 - val) * omega) / so).unsqueeze(1) * low + (torch.sin(val * omega) / so).unsqueeze(1) * high
    return res


class APVDDataset(Dataset):
    def __init__(
        self,
        image_paths: list[Path],
        target_resolution: int,
        archive_entries: list[tuple[Path, str]] | None = None,
        video_paths: list[Path] | None = None,
        video_stride: int = 30,
        video_max_frames: int = 0,
    ):
        self.image_paths = [Path(path) for path in image_paths]
        self.target_size = (int(target_resolution), int(target_resolution))
        self.transform = transforms.Compose(
            [
                transforms.Resize(
                    self.target_size,
                    interpolation=transforms.InterpolationMode.BICUBIC,
                ),
                transforms.ToTensor(),
            ]
        )
        self.fallback = torch.zeros(
            3,
            self.target_size[1],
            self.target_size[0],
            dtype=torch.float32,
        )
        self.video_stride = max(1, int(video_stride))
        self.video_max_frames = None if int(video_max_frames) <= 0 else int(video_max_frames)
        self.samples: list[tuple[str, object]] = [("path", path) for path in self.image_paths]
        if archive_entries:
            self.samples.extend(("archive", (Path(ap), member)) for ap, member in archive_entries)
        if video_paths:
            self.samples.extend(self._build_video_samples(video_paths))

    def __len__(self) -> int:
        return len(self.samples)

    @staticmethod
    def _is_safe_archive_member(member_name: str) -> bool:
        norm = member_name.replace("\\", "/").strip("/")
        if not norm:
            return False
        parts = norm.split("/")
        if ".." in parts:
            return False
        if parts[0].endswith(":"):
            return False
        return True

    @staticmethod
    def _skip_macosx_path(member_name: str) -> bool:
        parts = member_name.replace("\\", "/").split("/")
        return any(part == "__MACOSX" for part in parts)

    @staticmethod
    def _member_is_image_file(member_name: str) -> bool:
        return Path(member_name).suffix.lower() in {
            ".png",
            ".jpg",
            ".jpeg",
            ".bmp",
            ".gif",
            ".webp",
        }

    def _build_video_samples(self, video_paths: list[Path]) -> list[tuple[str, object]]:
        samples: list[tuple[str, object]] = []
        remaining = self.video_max_frames
        for video_path in video_paths:
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                continue
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            if frame_count <= 0:
                continue
            for frame_index in range(0, frame_count, self.video_stride):
                samples.append(("video", (Path(video_path), frame_index)))
                if remaining is not None:
                    remaining -= 1
                    if remaining <= 0:
                        return samples
        return samples

    def _load_archive_member(self, archive_path: Path, member_name: str) -> torch.Tensor:
        if not self._is_safe_archive_member(member_name) or self._skip_macosx_path(member_name):
            return self.fallback.clone()

        try:
            if archive_path.suffix.lower() == ".zip" or archive_path.name.lower().endswith((".zip",)):
                with zipfile.ZipFile(archive_path, "r") as zf:
                    with zf.open(member_name, "r") as f:
                        data = f.read()
            else:
                with tarfile.open(archive_path, "r:*") as tf:
                    info = tf.getmember(member_name)
                    if not info.isfile():
                        return self.fallback.clone()
                    reader = tf.extractfile(info)
                    if reader is None:
                        return self.fallback.clone()
                    data = reader.read()
            with Image.open(BytesIO(data)) as img:
                img = img.convert("RGB")
                return self.transform(img)
        except Exception:
            return self.fallback.clone()

    def _load_video_frame(self, video_path: Path, frame_index: int) -> torch.Tensor:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return self.fallback.clone()
        try:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
            ok, frame = cap.read()
            if not ok or frame is None:
                return self.fallback.clone()
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(np.ascontiguousarray(rgb))
            pil = pil.convert("RGB")
            return self.transform(pil)
        except Exception:
            return self.fallback.clone()
        finally:
            cap.release()

    def __getitem__(self, index: int) -> torch.Tensor:
        kind, payload = self.samples[index]
        try:
            if kind == "path":
                path = Path(payload)
                with Image.open(path) as img:
                    img = img.convert("RGB")
                    return self.transform(img)
            if kind == "archive":
                archive_path, member_name = payload
                return self._load_archive_member(Path(archive_path), str(member_name))
            if kind == "video":
                video_path, frame_index = payload
                return self._load_video_frame(Path(video_path), int(frame_index))
            with Image.open(Path(payload)) as img:
                img = img.convert("RGB")
                return self.transform(img)
        except Exception:
            return self.fallback.clone()


class APVDApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("APVD v3.0 - AI Pixel Value Determinator")
        self.root.geometry("800x1000")
        self.root.minsize(750, 950)

        self.device = get_device()
        self.model: VAE | None = None
        self.loaded_model_path: Path | None = None
        self.training_folder: Path | None = None
        self.training_paths: list[Path] | None = None
        self.batch_training_root: Path | None = None
        self.batch_training_folders: list[Path] = []
        self.archive_entries: list[tuple[Path, str]] = []
        self.video_paths: list[Path] = []
        self.training_tensors: torch.Tensor | None = None
        
        # State Variables
        self.epochs_var = tk.IntVar(value=100)
        self.resolution_var = tk.IntVar(value=256)
        self.video_stride_var = tk.IntVar(value=30)
        self.video_max_frames_var = tk.IntVar(value=0)
        self.iterations_var = tk.IntVar(value=3)
        self.show_iterations_var = tk.BooleanVar(value=True)
        self.auto_cycle_var = tk.BooleanVar(value=False)
        self.dream_cycle_var = tk.BooleanVar(value=False)
        self.blend_mode_var = tk.BooleanVar(value=False)
        self.blend_count_var = tk.IntVar(value=2)
        self.output_count_var = tk.IntVar(value=1)
        self.use_mini_diffusion_var = tk.BooleanVar(value=True)
        self.diffusion_steps_var = tk.IntVar(value=8)
        self.diffusion_strength_var = tk.DoubleVar(value=0.85)
        self.personality_var = tk.StringVar(value="Manual")
        self.generation_prompt_var = tk.StringVar(value="")
        self.include_memory_training_var = tk.BooleanVar(value=True)
        self.memory_recent_weight_var = tk.DoubleVar(value=0.7)
        self.dream_fps_var = tk.IntVar(value=16)
        self.evolution_count_var = tk.IntVar(value=6)
        self.evolution_selection_var = tk.StringVar(value="")
        self.is_training = False
        self.model_cycle_paths: list[Path] = []
        self.model_cycle_queue: list[Path] = []
        self.model_cycle_active = False
        self.model_cycle_delay_ms = 2000

        # Dream Cycle state
        self.current_latent = None
        self.target_latent = None
        self.interpolation_step = 0
        self.total_interpolation_steps = 20
        self.last_generated_latents: list[torch.Tensor] = []
        self.recent_memory_records = []
        self.evolution_candidates: list[dict] = []
        self.latent_map_points: list[dict] = []
        self.latent_map_window: tk.Toplevel | None = None
        self.latent_map_canvas: tk.Canvas | None = None
        self.model_map_window: tk.Toplevel | None = None
        self.model_map_tree: ttk.Treeview | None = None
        self.model_map_details_var = tk.StringVar(value="Select a .pt model to inspect its training details.")
        self.model_map_item_paths: dict[str, Path] = {}
        self.last_training_metadata: dict = {}
        self.memory_bank = MemoryBank(Path("Memory"))

        self.output_window: tk.Toplevel | None = None
        self.output_canvas: tk.Canvas | None = None
        self._output_photo = None

        self._build_ui()
        self._refresh_memory_list()
        self._create_output_window()

    def _create_output_window(self):
        """Second window: image-only view for OBS (window capture) without cropping the main UI."""
        if self.output_window is not None:
            try:
                if self.output_window.winfo_exists():
                    self.output_window.deiconify()
                    self.output_window.lift()
                    return
            except tk.TclError:
                pass
            self.output_window = None
            self.output_canvas = None

        w = tk.Toplevel(self.root)
        w.title("APVD Output Display")
        inner = 512
        pad = 8
        w.minsize(inner + pad * 2, inner + pad * 2 + 24)
        w.geometry(f"{inner + pad * 2}x{inner + pad * 2 + 24}")

        self.root.update_idletasks()
        try:
            x = self.root.winfo_x() + self.root.winfo_width() + 12
            y = self.root.winfo_y()
            w.geometry(f"{inner + pad * 2}x{inner + pad * 2 + 24}+{x}+{y}")
        except tk.TclError:
            pass

        outer = ttk.Frame(w, padding=pad)
        outer.pack(fill=tk.BOTH, expand=True)
        cv = tk.Canvas(
            outer,
            width=inner,
            height=inner,
            bg="#0d0d12",
            highlightthickness=0,
        )
        cv.pack()

        self.output_window = w
        self.output_canvas = cv

        def _on_close():
            if self.output_window is not None:
                try:
                    self.output_window.destroy()
                except tk.TclError:
                    pass
            self.output_window = None
            self.output_canvas = None
            self._output_photo = None

        w.protocol("WM_DELETE_WINDOW", _on_close)

    def _build_ui(self):
        # Top frame: Setup buttons
        btn_frame = ttk.Frame(self.root, padding=10)
        btn_frame.pack(fill=tk.X)

        ttk.Button(btn_frame, text="Select Images", command=self._select_images).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Select Folder", command=self._select_folder).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Batch Folder", command=self._select_batch_folder).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Select Video(s)", command=self._select_videos).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Select archive(s)", command=self._select_archives).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Clear sources", command=self._clear_training_sources).pack(side=tk.LEFT, padx=5)
        
        self.train_btn = ttk.Button(btn_frame, text="Train APVD", command=self._train)
        self.train_btn.pack(side=tk.LEFT, padx=5)
        
        self.stop_btn = ttk.Button(btn_frame, text="Stop Training", command=self._stop_training, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=5)

        ttk.Button(btn_frame, text="Save Model", command=self._save_model).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Load Model", command=self._load_model).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="PyTorch Map", command=self._open_model_map).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Merge Models", command=self._merge_models).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Prompt Generation", command=self._auto_load_model).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Compose Scene", command=self._compose_scene_prompt).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Output display", command=self._create_output_window).pack(
            side=tk.LEFT, padx=5
        )

        # Training settings
        train_frame = ttk.Frame(self.root, padding=10)
        train_frame.pack(fill=tk.X)
        ttk.Label(train_frame, text="Epochs:").pack(side=tk.LEFT, padx=(0, 10))
        self.epoch_spin = tk.Spinbox(
            train_frame, from_=1, to=50000, increment=1, width=8, textvariable=self.epochs_var
        )
        self.epoch_spin.pack(side=tk.LEFT)

        ttk.Label(train_frame, text="Resolution:").pack(side=tk.LEFT, padx=(20, 10))
        self.resolution_spin = tk.Spinbox(
            train_frame, from_=32, to=1024, increment=16, width=8, textvariable=self.resolution_var
        )
        self.resolution_spin.pack(side=tk.LEFT)

        ttk.Label(train_frame, text="Video stride:").pack(side=tk.LEFT, padx=(20, 8))
        ttk.Spinbox(
            train_frame, from_=1, to=10000, increment=1, width=6, textvariable=self.video_stride_var
        ).pack(side=tk.LEFT)
        ttk.Label(train_frame, text="Max frames (0=all):").pack(side=tk.LEFT, padx=(12, 8))
        ttk.Spinbox(
            train_frame, from_=0, to=1_000_000, increment=100, width=8, textvariable=self.video_max_frames_var
        ).pack(side=tk.LEFT)

        # Generation Tools
        gen_tools_frame = ttk.LabelFrame(self.root, text="Generation Tools", padding=10)
        gen_tools_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Button(gen_tools_frame, text="Generate Unique", command=self._generate).pack(side=tk.LEFT, padx=5)
        ttk.Button(gen_tools_frame, text="Reconstruction Video", command=self._export_reconstruction_video).pack(side=tk.LEFT, padx=5)
        ttk.Button(gen_tools_frame, text="Chaos Mode", command=self._toggle_chaos).pack(side=tk.LEFT, padx=5)
        ttk.Button(gen_tools_frame, text="Model Shuffle", command=self._toggle_model_cycle).pack(side=tk.LEFT, padx=5)
        ttk.Checkbutton(gen_tools_frame, text="Auto-Cycle", variable=self.auto_cycle_var, command=self._toggle_auto_cycle).pack(side=tk.LEFT, padx=10)
        ttk.Checkbutton(gen_tools_frame, text="Dream Cycle (Morph)", variable=self.dream_cycle_var, command=self._toggle_dream_cycle).pack(side=tk.LEFT, padx=10)
        ttk.Checkbutton(gen_tools_frame, text="Blend Trained Images", variable=self.blend_mode_var).pack(side=tk.LEFT, padx=10)

        # Sliders Frame
        sliders_container = ttk.Frame(self.root, padding=10)
        sliders_container.pack(fill=tk.X)

        # Variation Intensity slider
        ttk.Label(sliders_container, text="Variation Intensity:").grid(row=0, column=0, sticky=tk.W)
        self.var_scale = tk.Scale(
            sliders_container, from_=0.0, to=20.0, resolution=0.5, orient=tk.HORIZONTAL, length=300
        )
        self.var_scale.set(1.0)
        self.var_scale.grid(row=0, column=1, sticky=tk.EW, padx=10)

        # Morph Speed slider
        ttk.Label(sliders_container, text="Morph Smoothness:").grid(row=1, column=0, sticky=tk.W)
        self.speed_scale = tk.Scale(
            sliders_container, from_=5, to=150, resolution=1, orient=tk.HORIZONTAL, length=300
        )
        self.speed_scale.set(40)
        self.speed_scale.grid(row=1, column=1, sticky=tk.EW, padx=10, pady=5)

        # Blend count control
        ttk.Label(sliders_container, text="Blend Image Count:").grid(row=2, column=0, sticky=tk.W)
        self.blend_spin = tk.Spinbox(
            sliders_container, from_=2, to=16, increment=1, width=6, textvariable=self.blend_count_var
        )
        self.blend_spin.grid(row=2, column=1, sticky=tk.W, padx=10, pady=5)

        # Output count control
        ttk.Label(sliders_container, text="Output Image Count:").grid(row=3, column=0, sticky=tk.W)
        self.output_spin = tk.Spinbox(
            sliders_container, from_=1, to=8, increment=1, width=6, textvariable=self.output_count_var
        )
        self.output_spin.grid(row=3, column=1, sticky=tk.W, padx=10, pady=5)

        # Cleanup iteration controls
        iter_frame = ttk.Frame(self.root, padding=10)
        iter_frame.pack(fill=tk.X)
        ttk.Label(iter_frame, text="Cleanup Iterations:").pack(side=tk.LEFT, padx=(0, 10))
        self.iter_spin = tk.Spinbox(
            iter_frame, from_=0, to=25, increment=1, width=6, textvariable=self.iterations_var
        )
        self.iter_spin.pack(side=tk.LEFT, padx=(0, 10))
        ttk.Checkbutton(
            iter_frame, text="Show each iteration", variable=self.show_iterations_var
        ).pack(side=tk.LEFT)

        diffusion_frame = ttk.Frame(self.root, padding=10)
        diffusion_frame.pack(fill=tk.X)
        ttk.Checkbutton(
            diffusion_frame,
            text="Use Mini Diffusion",
            variable=self.use_mini_diffusion_var,
        ).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Label(diffusion_frame, text="Diffusion Steps:").pack(side=tk.LEFT, padx=(0, 10))
        ttk.Spinbox(
            diffusion_frame, from_=1, to=50, increment=1, width=6, textvariable=self.diffusion_steps_var
        ).pack(side=tk.LEFT)
        ttk.Label(diffusion_frame, text="Denoise Strength:").pack(side=tk.LEFT, padx=(20, 10))
        tk.Scale(
            diffusion_frame,
            from_=0.1,
            to=1.5,
            resolution=0.05,
            orient=tk.HORIZONTAL,
            length=180,
            variable=self.diffusion_strength_var,
        ).pack(side=tk.LEFT)

        prompt_frame = ttk.LabelFrame(self.root, text="Prompt / Personality", padding=10)
        prompt_frame.pack(fill=tk.X, padx=10, pady=5)
        ttk.Label(prompt_frame, text="Prompt tag:").grid(row=0, column=0, sticky=tk.W)
        ttk.Entry(prompt_frame, textvariable=self.generation_prompt_var, width=42).grid(
            row=0, column=1, sticky=tk.EW, padx=(8, 12)
        )
        ttk.Label(prompt_frame, text="Personality:").grid(row=0, column=2, sticky=tk.W)
        ttk.Combobox(
            prompt_frame,
            textvariable=self.personality_var,
            values=list(PERSONALITY_PRESETS.keys()),
            state="readonly",
            width=14,
        ).grid(row=0, column=3, sticky=tk.W, padx=(8, 12))
        ttk.Button(prompt_frame, text="Apply Preset", command=self._apply_personality_preset).grid(
            row=0, column=4, sticky=tk.W
        )
        ttk.Button(prompt_frame, text="Save Seed", command=self._save_current_seed).grid(
            row=1, column=0, sticky=tk.W, pady=(10, 0)
        )
        ttk.Button(prompt_frame, text="Load Seed", command=self._load_seed).grid(
            row=1, column=1, sticky=tk.W, pady=(10, 0)
        )
        ttk.Button(prompt_frame, text="Latent Map", command=self._open_latent_map).grid(
            row=1, column=2, sticky=tk.W, pady=(10, 0)
        )
        ttk.Button(prompt_frame, text="Memory Retrain", command=self._evolve_from_memory).grid(
            row=1, column=3, sticky=tk.W, pady=(10, 0)
        )
        ttk.Checkbutton(
            prompt_frame,
            text="Blend memory into training",
            variable=self.include_memory_training_var,
        ).grid(row=1, column=4, sticky=tk.W, pady=(10, 0))
        prompt_frame.columnconfigure(1, weight=1)

        evolution_frame = ttk.LabelFrame(self.root, text="Interactive Evolution", padding=10)
        evolution_frame.pack(fill=tk.X, padx=10, pady=5)
        ttk.Label(evolution_frame, text="Candidates:").grid(row=0, column=0, sticky=tk.W)
        ttk.Spinbox(
            evolution_frame,
            from_=4,
            to=8,
            increment=1,
            width=6,
            textvariable=self.evolution_count_var,
        ).grid(row=0, column=1, sticky=tk.W, padx=(8, 12))
        ttk.Label(evolution_frame, text="Favorites:").grid(row=0, column=2, sticky=tk.W)
        ttk.Entry(evolution_frame, textvariable=self.evolution_selection_var, width=20).grid(
            row=0, column=3, sticky=tk.W, padx=(8, 12)
        )
        ttk.Label(evolution_frame, text="Use 1,3,4 format").grid(row=0, column=4, sticky=tk.W)
        ttk.Button(evolution_frame, text="Evolution Round", command=self._generate_evolution_round).grid(
            row=1, column=0, sticky=tk.W, pady=(10, 0)
        )
        ttk.Button(evolution_frame, text="Breed Favorites", command=self._breed_evolution_favorites).grid(
            row=1, column=1, sticky=tk.W, pady=(10, 0), padx=(8, 0)
        )
        ttk.Label(evolution_frame, text="Dream FPS:").grid(row=1, column=2, sticky=tk.W, pady=(10, 0))
        ttk.Spinbox(
            evolution_frame,
            from_=1,
            to=60,
            increment=1,
            width=6,
            textvariable=self.dream_fps_var,
        ).grid(row=1, column=3, sticky=tk.W, padx=(8, 12), pady=(10, 0))

        memory_frame = ttk.LabelFrame(self.root, text="Memory Stream", padding=10)
        memory_frame.pack(fill=tk.X, padx=10, pady=5)
        self.memory_listbox = tk.Listbox(memory_frame, height=6, exportselection=False)
        self.memory_listbox.grid(row=0, column=0, columnspan=4, sticky=tk.EW)
        ttk.Button(memory_frame, text="Recall Memory", command=self._recall_selected_memory).grid(
            row=1, column=0, sticky=tk.W, pady=(8, 0)
        )
        ttk.Button(memory_frame, text="Use As Prompt Tag", command=self._use_memory_prompt).grid(
            row=1, column=1, sticky=tk.W, pady=(8, 0)
        )
        ttk.Label(memory_frame, text="Recent bias:").grid(row=1, column=2, sticky=tk.E, pady=(8, 0))
        tk.Scale(
            memory_frame,
            from_=0.1,
            to=1.0,
            resolution=0.05,
            orient=tk.HORIZONTAL,
            length=160,
            variable=self.memory_recent_weight_var,
        ).grid(row=1, column=3, sticky=tk.EW, pady=(8, 0))
        memory_frame.columnconfigure(0, weight=1)

        # Status label
        self.status_var = tk.StringVar(value="Ready.")
        ttk.Label(self.root, textvariable=self.status_var, font=("", 10, "bold")).pack(pady=5)

        # Dataset loading progress
        self.load_progress_var = tk.DoubleVar(value=0.0)
        self.load_progress = ttk.Progressbar(
            self.root,
            maximum=100.0,
            variable=self.load_progress_var,
            mode="determinate",
        )
        self.load_progress.pack(fill=tk.X, padx=10, pady=(0, 8))

        # Display area
        self.canvas_frame = ttk.Frame(self.root, padding=10)
        self.canvas_frame.pack(fill=tk.BOTH, expand=True)
        self.canvas = tk.Canvas(
            self.canvas_frame, width=512, height=512, bg="#1a1a2e", highlightthickness=1, highlightbackground="#4a4a6a"
        )
        self.canvas.pack()

        # Device info
        ttk.Label(self.root, text=f"Device: {self.device}", font=("", 9), foreground="gray").pack(pady=5)

    FILE_TYPES = [("Image files", "*.png *.jpg *.jpeg *.bmp *.gif *.webp"), ("All files", "*.*")]
    VIDEO_FILE_TYPES = [
        ("Video files", "*.mp4 *.avi *.mov *.mkv *.webm *.m4v *.wmv"),
        ("All files", "*.*"),
    ]
    AUDIO_FILE_TYPES = [
        ("Audio files", "*.mp3 *.wav *.m4a *.aac *.ogg *.flac"),
        ("All files", "*.*"),
    ]
    MODEL_FILE_TYPES = [("PyTorch model", "*.pt"), ("PyTorch checkpoint", "*.pth"), ("All files", "*.*")]
    ARCHIVE_FILE_TYPES = [
        ("Archives", "*.zip *.tar *.tar.gz *.tgz"),
        ("ZIP", "*.zip"),
        ("TAR / compressed", "*.tar *.tar.gz *.tgz"),
        ("All files", "*.*"),
    ]

    @staticmethod
    def _format_duration(seconds: float) -> str:
        if seconds < 60: return f"{seconds:.1f}s"
        minutes = seconds / 60
        if minutes < 60: return f"{minutes:.1f}m"
        hours = minutes / 60
        return f"{hours:.1f}h"

    def _current_mode_label(self) -> str:
        if self.dream_cycle_var.get():
            return "dream_cycle"
        if self.auto_cycle_var.get():
            return "auto_cycle"
        if self.model_cycle_active:
            return "model_shuffle"
        return "generate"

    def _refresh_memory_list(self):
        self.recent_memory_records = self.memory_bank.load_memories(limit=24)
        self.memory_listbox.delete(0, tk.END)
        for record in self.recent_memory_records:
            self.memory_listbox.insert(tk.END, summarize_memory(record))

    def _apply_personality_preset(self):
        preset = PERSONALITY_PRESETS.get(self.personality_var.get(), PERSONALITY_PRESETS["Manual"])
        self.var_scale.set(preset["intensity"])
        self.blend_mode_var.set(bool(preset["blend"]))
        self.blend_count_var.set(int(preset["blend_count"]))
        self.iterations_var.set(int(preset["iterations"]))
        self.use_mini_diffusion_var.set(bool(preset["use_diffusion"]))
        self.diffusion_steps_var.set(int(preset["diffusion_steps"]))
        self.diffusion_strength_var.set(float(preset["diffusion_strength"]))
        self.status_var.set(f"Applied personality preset: {self.personality_var.get()}")

    def _save_current_seed(self):
        if not self.last_generated_latents:
            messagebox.showerror("Save Seed", "Generate at least one image first.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".pt",
            filetypes=[("Latent seed", "*.pt"), ("All files", "*.*")],
            parent=self.root,
        )
        if not path:
            return
        payload = {
            "latent": self.last_generated_latents[0].detach().cpu(),
            "prompt": self.generation_prompt_var.get().strip(),
            "personality": self.personality_var.get(),
            "saved_at": datetime.now().isoformat(timespec="seconds"),
        }
        torch.save(payload, path)
        self.status_var.set(f"Saved latent seed: {Path(path).name}")

    def _load_seed(self):
        if self.model is None:
            messagebox.showerror("Load Seed", "Load or train a model first.")
            return
        path = filedialog.askopenfilename(
            filetypes=[("Latent seed", "*.pt"), ("All files", "*.*")],
            parent=self.root,
        )
        if not path:
            return
        payload = torch.load(path, map_location="cpu")
        latent = payload["latent"] if isinstance(payload, dict) and "latent" in payload else payload
        self._render_latent_gallery(
            [latent.detach().float()],
            mode="seed_recall",
            status_label=f"Loaded seed: {Path(path).name}",
            save_memory=False,
        )
        if isinstance(payload, dict):
            prompt = str(payload.get("prompt", "")).strip()
            if prompt:
                self.generation_prompt_var.set(prompt)

    def _remember_generation(self, image: Image.Image, latent: torch.Tensor, *, mode: str, extra: dict | None = None):
        model_name = "unloaded"
        if self.loaded_model_path is not None:
            model_name = self.loaded_model_path.name
        elif self.model is not None:
            model_name = f"latent-{self.model.latent_dim}"
        record = self.memory_bank.save_memory(
            image,
            latent,
            prompt=self.generation_prompt_var.get().strip(),
            mode=mode,
            personality=self.personality_var.get(),
            model_name=model_name,
            metadata=extra or {},
        )
        self._refresh_memory_list()
        return record

    def _get_selected_memory_record(self):
        selection = self.memory_listbox.curselection()
        if not selection:
            return None
        idx = int(selection[0])
        if idx < 0 or idx >= len(self.recent_memory_records):
            return None
        return self.recent_memory_records[idx]

    def _recall_selected_memory(self):
        if self.model is None:
            messagebox.showerror("Recall Memory", "Load or train a model first.")
            return
        record = self._get_selected_memory_record()
        if record is None:
            messagebox.showerror("Recall Memory", "Select a memory first.")
            return
        latent = self.memory_bank.load_latent(record).detach().float()
        self._render_latent_gallery(
            [latent],
            mode="memory_recall",
            status_label=f"Recalled memory: {record.memory_id}",
            save_memory=False,
        )

    def _use_memory_prompt(self):
        record = self._get_selected_memory_record()
        if record is None:
            messagebox.showerror("Memory Stream", "Select a memory first.")
            return
        self.generation_prompt_var.set(record.prompt)
        self.personality_var.set(record.personality)
        self.status_var.set(f"Loaded prompt tag from {record.memory_id}")

    def _open_latent_map(self):
        self.latent_map_points = self.memory_bank.build_latent_map(limit=64)
        if not self.latent_map_points:
            messagebox.showerror("Latent Map", "At least two saved memories are required.")
            return

        if self.latent_map_window is None or not self.latent_map_window.winfo_exists():
            self.latent_map_window = tk.Toplevel(self.root)
            self.latent_map_window.title("Latent Space Map")
            self.latent_map_canvas = tk.Canvas(self.latent_map_window, width=560, height=360, bg="#10131a")
            self.latent_map_canvas.pack(fill=tk.BOTH, expand=True)
            self.latent_map_canvas.bind("<Button-1>", self._on_latent_map_click)

        self._draw_latent_map()

    def _draw_latent_map(self):
        if self.latent_map_canvas is None:
            return
        canvas = self.latent_map_canvas
        canvas.delete("all")
        canvas.create_text(12, 12, anchor=tk.NW, fill="#d9e2ff", text="Click a node to recall that memory")
        for idx, point in enumerate(self.latent_map_points, start=1):
            record = point["record"]
            radius = 6
            x = point["x"]
            y = point["y"]
            fill = "#ffb347" if record.personality == "Chaotic" else "#7fd6ff"
            if record.personality in {"Dreamy", "Nostalgic"}:
                fill = "#b0e57c"
            canvas.create_oval(x - radius, y - radius, x + radius, y + radius, fill=fill, outline="")
            canvas.create_text(x + 10, y - 10, anchor=tk.W, fill="#eef3ff", text=str(idx))

    def _on_latent_map_click(self, event):
        if not self.latent_map_points:
            return
        best_point = min(
            self.latent_map_points,
            key=lambda point: (point["x"] - event.x) ** 2 + (point["y"] - event.y) ** 2,
        )
        record = best_point["record"]
        try:
            latent = self.memory_bank.load_latent(record).detach().float()
        except Exception as exc:
            messagebox.showerror("Latent Map", str(exc))
            return
        self._render_latent_gallery(
            [latent],
            mode="latent_map_recall",
            status_label=f"Latent map jump: {record.memory_id}",
            save_memory=False,
        )

    @staticmethod
    def _coerce_int(value) -> int | None:
        if value is None or isinstance(value, bool):
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _format_metadata_value(value) -> str:
        if value is None or value == "":
            return "Unknown"
        if isinstance(value, (list, tuple)):
            return ", ".join(str(item) for item in value) if value else "Unknown"
        return str(value)

    def _extract_model_training_metadata(self, checkpoint: object, model_path: Path) -> dict:
        metadata: dict = {}
        if isinstance(checkpoint, dict):
            raw_metadata = checkpoint.get("training_metadata")
            if isinstance(raw_metadata, dict):
                metadata.update(raw_metadata)
            legacy_metadata = checkpoint.get("metadata")
            if isinstance(legacy_metadata, dict):
                for key, value in legacy_metadata.items():
                    metadata.setdefault(key, value)
            metadata.setdefault("latent_dim", checkpoint.get("latent_dim"))
            metadata.setdefault("output_size", checkpoint.get("output_size"))
            metadata.setdefault("version", checkpoint.get("version"))

        epochs = self._coerce_int(metadata.get("epochs"))
        total_epochs = self._coerce_int(metadata.get("total_epochs"))
        if total_epochs is None:
            total_epochs = epochs
        image_count = self._coerce_int(metadata.get("dataset_image_count"))
        if image_count is None:
            image_count = self._coerce_int(metadata.get("image_count"))

        return {
            "dataset_label": metadata.get("dataset_label"),
            "dataset_image_count": image_count,
            "epochs": epochs,
            "total_epochs": total_epochs,
            "chunk_count": self._coerce_int(metadata.get("chunk_count")),
            "source_count": self._coerce_int(metadata.get("source_count")),
            "resolution": metadata.get("resolution") or metadata.get("output_size"),
            "saved_at": metadata.get("saved_at"),
            "version": metadata.get("version"),
            "path": model_path,
        }

    def _format_model_details(self, details: dict) -> str:
        relative_path = details["path"]
        try:
            relative_path = relative_path.relative_to(Path("Models"))
        except ValueError:
            relative_path = details["path"]

        lines = [
            f"Model: {details['path'].name}",
            f"Folder: {relative_path.parent if relative_path.parent != Path('.') else 'Models'}",
            f"Dataset label: {self._format_metadata_value(details.get('dataset_label'))}",
            f"Dataset image count: {self._format_metadata_value(details.get('dataset_image_count'))}",
            f"Epochs per chunk: {self._format_metadata_value(details.get('epochs'))}",
            f"Total epochs trained: {self._format_metadata_value(details.get('total_epochs'))}",
            f"Training chunks: {self._format_metadata_value(details.get('chunk_count'))}",
            f"Source files used: {self._format_metadata_value(details.get('source_count'))}",
            f"Resolution: {self._format_metadata_value(details.get('resolution'))}",
            f"Saved at: {self._format_metadata_value(details.get('saved_at'))}",
            f"Checkpoint version: {self._format_metadata_value(details.get('version'))}",
        ]
        if details.get("dataset_image_count") is None or details.get("total_epochs") is None:
            lines.append("")
            lines.append("Legacy checkpoint: this model was saved before training metadata was tracked.")
        return "\n".join(lines)

    def _open_model_map(self):
        models_folder = Path("Models")
        if not models_folder.exists():
            messagebox.showerror("PyTorch Map", f"Model folder not found:\n{models_folder.resolve()}")
            return

        if self.model_map_window is None or not self.model_map_window.winfo_exists():
            self.model_map_window = tk.Toplevel(self.root)
            self.model_map_window.title("PyTorch Model Map")
            self.model_map_window.geometry("900x560")
            self.model_map_window.minsize(760, 440)

            container = ttk.Frame(self.model_map_window, padding=10)
            container.pack(fill=tk.BOTH, expand=True)
            container.columnconfigure(0, weight=3)
            container.columnconfigure(1, weight=2)
            container.rowconfigure(1, weight=1)

            ttk.Label(
                container,
                text="Browse trained checkpoints in Models and click a .pt file to inspect its training metadata.",
            ).grid(row=0, column=0, columnspan=2, sticky=tk.W, pady=(0, 8))

            tree_frame = ttk.Frame(container)
            tree_frame.grid(row=1, column=0, sticky=tk.NSEW, padx=(0, 10))
            tree_frame.columnconfigure(0, weight=1)
            tree_frame.rowconfigure(0, weight=1)

            self.model_map_tree = ttk.Treeview(tree_frame, show="tree")
            self.model_map_tree.grid(row=0, column=0, sticky=tk.NSEW)
            self.model_map_tree.bind("<<TreeviewSelect>>", self._on_model_map_select)

            tree_scroll = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL, command=self.model_map_tree.yview)
            tree_scroll.grid(row=0, column=1, sticky=tk.NS)
            self.model_map_tree.configure(yscrollcommand=tree_scroll.set)

            detail_frame = ttk.LabelFrame(container, text="Model Details", padding=10)
            detail_frame.grid(row=1, column=1, sticky=tk.NSEW)
            detail_frame.columnconfigure(0, weight=1)
            detail_frame.rowconfigure(0, weight=1)

            ttk.Label(
                detail_frame,
                textvariable=self.model_map_details_var,
                justify=tk.LEFT,
                anchor=tk.NW,
            ).grid(row=0, column=0, sticky=tk.NSEW)

        self._refresh_model_map()
        self.model_map_window.deiconify()
        self.model_map_window.lift()

    def _refresh_model_map(self):
        if self.model_map_tree is None:
            return

        models_folder = Path("Models")
        self.model_map_tree.delete(*self.model_map_tree.get_children())
        self.model_map_item_paths = {}
        self.model_map_details_var.set("Select a .pt model to inspect its training details.")

        root_id = self.model_map_tree.insert("", tk.END, text="Models", open=True)

        def add_folder(parent_id: str, folder: Path) -> None:
            for child in sorted(folder.iterdir(), key=lambda path: (not path.is_dir(), path.name.lower())):
                if child.is_dir():
                    child_id = self.model_map_tree.insert(parent_id, tk.END, text=child.name, open=False)
                    add_folder(child_id, child)
                elif child.suffix.lower() == ".pt":
                    item_id = self.model_map_tree.insert(parent_id, tk.END, text=child.name, open=False)
                    self.model_map_item_paths[item_id] = child

        add_folder(root_id, models_folder)

    def _on_model_map_select(self, _event=None):
        if self.model_map_tree is None:
            return
        selection = self.model_map_tree.selection()
        if not selection:
            return

        model_path = self.model_map_item_paths.get(selection[0])
        if model_path is None:
            self.model_map_details_var.set("Select a .pt model to inspect its training details.")
            return

        try:
            checkpoint = torch.load(model_path, map_location="cpu")
            details = self._extract_model_training_metadata(checkpoint, model_path)
            self.model_map_details_var.set(self._format_model_details(details))
        except Exception as exc:
            self.model_map_details_var.set(f"Could not read {model_path.name}:\n{exc}")

    def _evolve_from_memory(self):
        if self.model is None:
            messagebox.showerror("Memory Retrain", "Load or train a model first.")
            return
        memory_paths = self.memory_bank.get_weighted_image_paths(
            limit=32,
            recent_bias=float(self.memory_recent_weight_var.get()),
        )
        if not memory_paths:
            messagebox.showerror("Memory Retrain", "No saved memories were found.")
            return

        self.is_training = True
        self.train_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)

        def _run():
            try:
                completed = self._train_dataset(
                    n_epochs=max(1, min(30, int(self.epochs_var.get() // 4) or 1)),
                    resolution=int(self.resolution_var.get()),
                    dataset_label="Memory evolution",
                    training_paths=memory_paths,
                    reset_model=False,
                )
                if completed:
                    self.root.after(0, lambda: self.status_var.set("Memory evolution training finished."))
            except Exception as exc:
                self.root.after(0, lambda: messagebox.showerror("Memory Retrain", str(exc)))
            finally:
                self._finish_training()

        threading.Thread(target=_run, daemon=True).start()

    def _generate_evolution_round(self):
        if self.model is None:
            messagebox.showerror("Evolution", "Load or train a model first.")
            return
        count = max(4, min(8, int(self.evolution_count_var.get())))
        self.evolution_selection_var.set("")
        self.evolution_candidates = [{"latent": self._get_random_latent()} for _ in range(count)]
        self._render_latent_gallery(
            [candidate["latent"] for candidate in self.evolution_candidates],
            mode="evolution_round",
            numbered=True,
            status_label=f"Evolution round ready. Pick favorites from 1-{count}.",
        )

    def _breed_evolution_favorites(self):
        if self.model is None:
            messagebox.showerror("Evolution", "Load or train a model first.")
            return
        if not self.evolution_candidates:
            messagebox.showerror("Evolution", "Run an evolution round first.")
            return
        selected = parse_selection_indices(
            self.evolution_selection_var.get(),
            upper_bound=len(self.evolution_candidates),
        )
        if not selected:
            messagebox.showerror("Evolution", "Enter favorite candidate numbers like 1,3,4.")
            return
        parents = [self.evolution_candidates[idx]["latent"] for idx in selected]
        child_latents = breed_latents(
            parents,
            noise_scale=max(0.05, self.var_scale.get() / 12.0),
            child_count=len(self.evolution_candidates),
        )
        self.evolution_candidates = [{"latent": latent} for latent in child_latents]
        self._render_latent_gallery(
            child_latents,
            mode="evolution_breed",
            numbered=True,
            status_label=f"Bred {len(child_latents)} children from favorites {self.evolution_selection_var.get()}",
        )

    def _select_images(self):
        paths = filedialog.askopenfilenames(title="Select training images", filetypes=self.FILE_TYPES)
        if not paths: return
        self.training_paths = [Path(p) for p in paths]
        self.training_folder = None
        self.batch_training_root = None
        self.batch_training_folders = []
        self.status_var.set(f"Selected {len(self.training_paths)} images.")

    def _select_videos(self):
        paths = filedialog.askopenfilenames(
            title="Select training video(s)",
            filetypes=self.VIDEO_FILE_TYPES,
        )
        if not paths:
            return
        self.video_paths = [Path(p) for p in paths]
        n = len(self.video_paths)
        extra = ""
        extra_parts = []
        if self.training_paths:
            extra_parts.append(f"{len(self.training_paths)} image(s)")
        if self.archive_entries:
            extra_parts.append(f"{len(self.archive_entries)} from archive(s)")
        extra = ""
        if extra_parts:
            extra = " Will merge with " + " and ".join(extra_parts) + " on train."
        self.status_var.set(f"Selected {n} video file(s).{extra}")

    def _select_batch_folder(self):
        folder = filedialog.askdirectory(title="Select parent folder with dataset subfolders")
        if not folder:
            return
        root = Path(folder)
        dataset_folders = self._find_batch_dataset_folders(root)
        if not dataset_folders:
            messagebox.showerror(
                "Error",
                "No dataset subfolders with images were found.\n"
                "Put each dataset in its own folder inside the selected parent folder.",
            )
            return

        self.batch_training_root = root
        self.batch_training_folders = dataset_folders
        self.training_folder = None
        self.training_paths = None
        self.archive_entries = []
        self.video_paths = []
        self.status_var.set(
            f"Queued {len(dataset_folders)} dataset folder(s) for batch training."
        )

    def _select_archives(self):
        paths = filedialog.askopenfilenames(
            title="Select training archive(s)",
            filetypes=self.ARCHIVE_FILE_TYPES,
        )
        if not paths:
            return
        new_entries: list[tuple[Path, str]] = []
        n_ok = 0
        for p in paths:
            ap = Path(p)
            members = list_image_members(ap)
            if not members:
                messagebox.showerror(
                    "Error",
                    f"No supported images found in archive (or unsupported format):\n{ap.name}",
                )
                continue
            n_ok += 1
            new_entries.extend((ap, m) for m in members)
        if not new_entries:
            return
        self.archive_entries.extend(new_entries)
        self.status_var.set(
            f"Added {len(new_entries)} image(s) from {n_ok} archive(s). "
            f"Total from archives: {len(self.archive_entries)}."
        )

    def _clear_training_sources(self):
        self.training_paths = None
        self.training_folder = None
        self.batch_training_root = None
        self.batch_training_folders = []
        self.archive_entries = []
        self.video_paths = []
        self.status_var.set("Cleared image/video/archive sources.")

    def _select_folder(self):
        folder = filedialog.askdirectory(title="Select folder with training images")
        if not folder: return
        self.training_folder = Path(folder)
        paths = get_image_paths(self.training_folder)
        if not paths:
            messagebox.showerror("Error", "No images found.")
            return
        self.batch_training_root = None
        self.batch_training_folders = []
        self.training_paths = paths
        self.status_var.set(f"Found {len(paths)} images in folder.")

    def _export_reconstruction_video(self):
        frame_paths = filedialog.askopenfilenames(
            title="Select ordered latent traversal frames",
            filetypes=self.FILE_TYPES,
            parent=self.root,
        )
        if not frame_paths:
            return
        if len(frame_paths) < 2:
            messagebox.showerror("Reconstruction Video", "Select at least 2 frame images.")
            return

        original_path = None
        if messagebox.askyesno(
            "Reconstruction Video",
            "Include an original input image for the initialization and final comparison phases?",
            parent=self.root,
        ):
            picked = filedialog.askopenfilename(
                title="Select original input image",
                filetypes=self.FILE_TYPES,
                parent=self.root,
            )
            if picked:
                original_path = picked

        audio_path = None
        if messagebox.askyesno(
            "Reconstruction Video",
            "Add an ambient audio track if you have one?",
            parent=self.root,
        ):
            picked = filedialog.askopenfilename(
                title="Select optional audio track",
                filetypes=self.AUDIO_FILE_TYPES,
                parent=self.root,
            )
            if picked:
                audio_path = picked

        resolution_text = simpledialog.askstring(
            "Reconstruction Video",
            "Output resolution (`1920x1080` or `1280x720`):",
            initialvalue="1920x1080",
            parent=self.root,
        )
        if not resolution_text:
            return
        resolution_text = resolution_text.strip().lower().replace(" ", "")
        resolution_map = {
            "1920x1080": (1920, 1080),
            "1280x720": (1280, 720),
        }
        resolution = resolution_map.get(resolution_text)
        if resolution is None:
            messagebox.showerror("Reconstruction Video", "Resolution must be `1920x1080` or `1280x720`.")
            return

        fps = simpledialog.askinteger(
            "Reconstruction Video",
            "FPS (`30` or `60`):",
            initialvalue=30,
            minvalue=30,
            maxvalue=60,
            parent=self.root,
        )
        if fps is None:
            return
        if fps not in {30, 60}:
            messagebox.showerror("Reconstruction Video", "FPS must be `30` or `60`.")
            return

        suggested_name = f"reconstruction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
        output_path = filedialog.asksaveasfilename(
            title="Save reconstruction video",
            defaultextension=".mp4",
            initialdir=str(Path("Outputs").resolve()),
            initialfile=suggested_name,
            filetypes=[("MP4 video", "*.mp4"), ("All files", "*.*")],
            parent=self.root,
        )
        if not output_path:
            return

        ordered_paths = [Path(path) for path in sorted(frame_paths, key=lambda value: [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", Path(value).name.lower())])]
        self.status_var.set("Rendering Reconstruction Video Mode...")

        def progress_update(done: int, total: int):
            self.root.after(
                0,
                lambda d=done, t=total: self.status_var.set(
                    f"Rendering Reconstruction Video Mode... {int((d / max(1, t)) * 100)}%"
                ),
            )

        def render_job():
            try:
                result = render_reconstruction_video(
                    ordered_paths,
                    Path(output_path),
                    original_image_path=Path(original_path) if original_path else None,
                    audio_path=Path(audio_path) if audio_path else None,
                    resolution=resolution,
                    fps=fps,
                    progress_callback=progress_update,
                )
            except Exception as exc:
                self.root.after(
                    0,
                    lambda: (
                        self.status_var.set("Reconstruction video export failed."),
                        messagebox.showerror("Reconstruction Video", str(exc), parent=self.root),
                    ),
                )
                return

            codec_note = "H.264" if result.used_h264 else "MP4 fallback codec"
            audio_note = " with audio" if result.audio_included else ""
            self.root.after(
                0,
                lambda: (
                    self.status_var.set(
                        f"Saved reconstruction video: {result.output_path.name} ({result.duration_seconds:.1f}s, {codec_note}{audio_note})."
                    ),
                    messagebox.showinfo(
                        "Reconstruction Video",
                        f"Saved video to:\n{result.output_path}\n\nDuration: {result.duration_seconds:.1f}s\nCodec: {codec_note}",
                        parent=self.root,
                    ),
                ),
            )

        threading.Thread(target=render_job, daemon=True).start()

    def _train(self):
        has_batch_folders = bool(self.batch_training_folders)
        has_images = bool(self.training_paths) or bool(self.archive_entries)
        has_videos = bool(self.video_paths)
        if not has_batch_folders and not has_images and not has_videos:
            messagebox.showerror(
                "Error",
                "Select images, a folder, a batch folder, archive(s), and/or video(s) first.",
            )
            return
        self.is_training = True
        self.train_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        training_thread = threading.Thread(target=self._training_loop, daemon=True)
        training_thread.start()

    def _stop_training(self):
        self.is_training = False
        self.status_var.set("Stopping training...")

    @staticmethod
    def _sanitize_model_stem(name: str) -> str:
        sanitized = re.sub(r'[<>:"/\\|?*\x00-\x1f]+', " ", name).strip()
        sanitized = re.sub(r"\s+", " ", sanitized)
        return sanitized or "model"

    def _find_batch_dataset_folders(self, root: Path) -> list[Path]:
        dataset_folders: list[Path] = []
        for child in sorted((p for p in root.iterdir() if p.is_dir()), key=lambda p: p.name.lower()):
            if get_image_paths(child):
                dataset_folders.append(child)
        return dataset_folders

    def _build_model_checkpoint(self) -> dict:
        if self.model is None:
            raise ValueError("No model is loaded.")
        return {
            "model_state_dict": self.model.state_dict(),
            "latent_dim": self.model.latent_dim,
            "output_size": self.model.output_size,
            "version": "2.3-mini-diffusion",
            "training_metadata": dict(self.last_training_metadata),
        }

    def _next_available_model_path(self, target_dir: Path, model_name: str) -> Path:
        stem = self._sanitize_model_stem(model_name)
        candidate = target_dir / f"{stem}.pt"
        if not candidate.exists():
            return candidate

        index = 2
        while True:
            candidate = target_dir / f"{stem} ({index}).pt"
            if not candidate.exists():
                return candidate
            index += 1

    def _save_model_to_path(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self._build_model_checkpoint(), path)
        if self.model_map_window is not None and self.model_map_window.winfo_exists():
            self._refresh_model_map()

    @staticmethod
    def _sample_training_preview(tensors: torch.Tensor, max_items: int) -> torch.Tensor:
        if tensors.size(0) <= max_items:
            return tensors.detach().cpu()
        idx = torch.randperm(tensors.size(0))[:max_items]
        return tensors[idx].detach().cpu()

    def _train_dataset(
        self,
        n_epochs: int,
        resolution: int,
        dataset_label: str,
        training_paths: list[Path] | None = None,
        archive_entries: list[tuple[Path, str]] | None = None,
        video_paths: list[Path] | None = None,
        reset_model: bool = False,
        batch_index: int | None = None,
        batch_total: int | None = None,
    ) -> bool:
        if reset_model:
            self.model = None
            self.loaded_model_path = None

        image_paths = [Path(path) for path in (training_paths or [])]
        archive_entries = list(archive_entries or [])
        video_paths = [Path(path) for path in (video_paths or [])]
        if not image_paths and not archive_entries and not video_paths:
            raise ValueError("No training data sources were provided.")

        target_size = (resolution, resolution)
        if self.model is None or getattr(self.model, "output_size", target_size) != target_size:
            self.model = VAE(latent_dim=512, output_size=target_size).to(self.device)
            self.loaded_model_path = None

        amp_enabled = getattr(self.device, "type", "") == "cuda"
        if amp_enabled:
            torch.backends.cudnn.benchmark = True

        dataset = APVDDataset(
            image_paths,
            resolution,
            archive_entries=archive_entries,
            video_paths=video_paths,
            video_stride=self.video_stride_var.get(),
            video_max_frames=self.video_max_frames_var.get(),
        )
        batch_size = 16
        data_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=2,
        )

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=2e-4)
        scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
        self.model.train()
        train_start = time.perf_counter()
        preview_parts: list[torch.Tensor] = []
        total_batches = max(1, len(data_loader))
        total_steps = max(1, n_epochs * total_batches)
        completed_steps = 0
        total_training_images = len(dataset)

        self.root.after(0, lambda: self.load_progress_var.set(0.0))

        prefix_parts: list[str] = []
        if batch_index is not None and batch_total is not None:
            prefix_parts.append(f"[{batch_index}/{batch_total}]")
        prefix = " ".join(prefix_parts)
        if prefix:
            prefix += " "

        for epoch in range(n_epochs):
            if not self.is_training:
                return False

            epoch_loss = 0.0
            for batch_idx, batch in enumerate(data_loader, start=1):
                if not self.is_training:
                    return False

                batch = batch.to(self.device, non_blocking=amp_enabled)

                if len(preview_parts) < MAX_TRAINING_PREVIEW_IMAGES:
                    remaining = MAX_TRAINING_PREVIEW_IMAGES - len(preview_parts)
                    preview_parts.extend(batch[:remaining].detach().cpu())

                optimizer.zero_grad(set_to_none=True)

                with torch.amp.autocast("cuda", enabled=amp_enabled):
                    recon, mu, logvar = self.model(batch)
                with torch.amp.autocast("cuda", enabled=False):
                    recon_loss = vae_loss(recon.float(), batch.float(), mu.float(), logvar.float())
                    denoise_loss = latent_denoiser_loss(self.model, mu.detach().float())
                    loss = recon_loss + (0.25 * denoise_loss)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                epoch_loss += loss.detach().float().item()

                completed_steps += 1
                progress = min(100.0, (completed_steps / total_steps) * 100.0)
                elapsed = time.perf_counter() - train_start
                eta = (total_steps - completed_steps) * (elapsed / max(1, completed_steps))
                running_avg = epoch_loss / max(1, batch_idx)
                self.root.after(0, lambda p=progress: self.load_progress_var.set(p))
                if batch_idx == 1 or batch_idx % 50 == 0 or batch_idx == total_batches:
                    self.root.after(
                        0,
                        lambda e=epoch, b=batch_idx, a=running_avg, t=eta, p=prefix, label=dataset_label: self.status_var.set(
                            f"{p}{label} | Epoch {e+1}/{n_epochs} | Batch {b}/{total_batches} | Loss: {a:.0f} | ETA: {self._format_duration(t)}"
                        ),
                    )

            avg = epoch_loss / total_batches
            elapsed = time.perf_counter() - train_start
            eta = (n_epochs - (epoch + 1)) * (elapsed / max(1, epoch + 1))
            self.root.after(
                0,
                lambda e=epoch, a=avg, t=eta, p=prefix, label=dataset_label: self.status_var.set(
                    f"{p}{label} | Epoch {e+1}/{n_epochs} | Loss: {a:.0f} | ETA: {self._format_duration(t)}"
                ),
            )

        if amp_enabled:
            torch.cuda.empty_cache()

        self.model.eval()
        self.last_training_metadata = {
            "dataset_label": dataset_label,
            "dataset_image_count": total_training_images,
            "epochs": n_epochs,
            "total_epochs": n_epochs,
            "chunk_count": 1,
            "source_count": len(image_paths),
            "resolution": [resolution, resolution],
            "saved_at": datetime.now().isoformat(timespec="seconds"),
            "version": "2.3-mini-diffusion",
        }
        if preview_parts:
            self.training_tensors = torch.stack(preview_parts, dim=0)[:MAX_TRAINING_PREVIEW_IMAGES]
        return True

    def _training_loop(self):
        n_epochs = max(1, int(self.epochs_var.get()))
        resolution = int(self.resolution_var.get())
        resolution = max(32, min(1024, resolution))
        self.root.after(0, lambda: self.resolution_var.set(resolution))

        try:
            if self.batch_training_folders:
                models_folder = Path("Models")
                models_folder.mkdir(parents=True, exist_ok=True)
                total = len(self.batch_training_folders)
                saved_paths: list[Path] = []

                for index, dataset_folder in enumerate(self.batch_training_folders, start=1):
                    if not self.is_training:
                        break
                    dataset_paths = get_image_paths(dataset_folder)
                    if not dataset_paths:
                        continue

                    self.root.after(
                        0,
                        lambda i=index, t=total, name=dataset_folder.name: self.status_var.set(
                            f"[{i}/{t}] Preparing dataset: {name}"
                        ),
                    )
                    completed = self._train_dataset(
                        n_epochs=n_epochs,
                        resolution=resolution,
                        dataset_label=dataset_folder.name,
                        training_paths=dataset_paths,
                        reset_model=True,
                        batch_index=index,
                        batch_total=total,
                    )
                    if not completed:
                        break

                    model_path = self._next_available_model_path(models_folder, dataset_folder.name)
                    self._save_model_to_path(model_path)
                    saved_paths.append(model_path)
                    self.root.after(
                        0,
                        lambda i=index, t=total, p=model_path: self.status_var.set(
                            f"[{i}/{t}] Saved {p.name} to Models."
                        ),
                    )

                if saved_paths and self.is_training:
                    self.root.after(
                        0,
                        lambda count=len(saved_paths), last=saved_paths[-1].name: self.status_var.set(
                            f"Batch training finished. Saved {count} model(s); last: {last}"
                        ),
                    )
            else:
                memory_paths: list[Path] = []
                if self.include_memory_training_var.get():
                    memory_paths = self.memory_bank.get_weighted_image_paths(
                        limit=32,
                        recent_bias=float(self.memory_recent_weight_var.get()),
                    )
                merged_training_paths = list(self.training_paths or [])
                merged_training_paths.extend(memory_paths)
                self._train_dataset(
                    n_epochs=n_epochs,
                    resolution=resolution,
                    dataset_label="Current dataset",
                    training_paths=merged_training_paths,
                    archive_entries=self.archive_entries,
                    video_paths=self.video_paths,
                    reset_model=False,
                )
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Error", str(e)))
        finally:
            if not self.is_training:
                self.root.after(0, lambda: self.status_var.set("Training stopped."))
        self._finish_training()

    def _finish_training(self):
        def _finish():
            self.is_training = False
            self.train_btn.config(state=tk.NORMAL)
            self.stop_btn.config(state=tk.DISABLED)

        if threading.current_thread() is threading.main_thread():
            _finish()
        else:
            self.root.after(0, _finish)

    def _save_model(self):
        if self.model is None: return
        path = filedialog.asksaveasfilename(defaultextension=".pt", filetypes=self.MODEL_FILE_TYPES)
        if path:
            self._save_model_to_path(Path(path))

    def _load_model(self):
        path = filedialog.askopenfilename(filetypes=self.MODEL_FILE_TYPES)
        if not path: return
        self._load_model_file(Path(path))

    def _load_model_file(self, path: Path):
        checkpoint = torch.load(path, map_location=self.device)
        output_size = tuple(checkpoint.get("output_size", (256, 256)))
        self.model = VAE(latent_dim=checkpoint.get("latent_dim", 256), output_size=output_size).to(self.device)
        self.loaded_model_path = Path(path)
        self.last_training_metadata = self._extract_model_training_metadata(checkpoint, self.loaded_model_path)
        load_result = self.model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        self.model.eval()
        if len(output_size) == 2 and all(isinstance(v, int) for v in output_size):
            self.resolution_var.set(int(output_size[0]))
        missing = list(getattr(load_result, "missing_keys", []))
        has_denoiser = not any(key.startswith("latent_denoiser.") for key in missing)
        if not has_denoiser:
            self.use_mini_diffusion_var.set(False)
        note = " Mini diffusion ready." if has_denoiser else " Legacy checkpoint loaded; mini diffusion disabled until retrained."
        self.status_var.set(f"Model loaded: {path.name}.{note}")

    def _auto_load_model(self):
        models_folder = Path("Models")
        if not models_folder.exists():
            messagebox.showerror("Error", f"Model folder not found:\n{models_folder.resolve()}")
            return

        prompt = simpledialog.askstring(
            "Auto Load Model",
            "Enter a prompt, folder name, or nested category path (for example Games or Food):",
            parent=self.root,
        )
        if prompt is None:
            return

        try:
            best_model_path, selection_reason = select_model_path_for_prompt(models_folder, prompt)
            print(f"Selected model: {best_model_path.name} ({selection_reason})")
            self.generation_prompt_var.set(prompt.strip())
            self._load_model_file(best_model_path)
            self.status_var.set(
                f"Selected model: {best_model_path.name} ({selection_reason})"
            )
            self._generate_image(self.model)
        except Exception as exc:
            messagebox.showerror("Auto Load Model", str(exc))

    def _compose_scene_prompt(self):
        models_folder = Path("Models")
        if not models_folder.exists():
            messagebox.showerror("Error", f"Model folder not found:\n{models_folder.resolve()}")
            return

        prompt = simpledialog.askstring(
            "Compose Scene",
            "Enter a scene prompt (for example: a cat next to a dog, a burger above a car):",
            parent=self.root,
        )
        if prompt is None:
            return

        try:
            self.generation_prompt_var.set(prompt.strip())
            output_image, output_path, scene = generate_scene_from_prompt(
                prompt=prompt,
                models_folder=models_folder,
                device=self.device,
                output_dir=Path("Outputs"),
                target_size=(self.resolution_var.get(), self.resolution_var.get()),
            )
            self._display_image(output_image)
            description = self._describe_scene(scene)
            relation = "layered_depth" if isinstance(scene, LayeredScene) else scene.relation
            self.status_var.set(
                f"Composed scene: {description} ({relation}) -> {output_path.name}"
            )
        except Exception as exc:
            messagebox.showerror("Compose Scene", str(exc))

    def _merge_models(self):
        models_folder = Path("Models")
        if not models_folder.exists():
            messagebox.showerror("Merge Models", f"Model folder not found:\n{models_folder.resolve()}")
            return

        selected = filedialog.askopenfilenames(
            title="Select model checkpoints to merge",
            initialdir=str(models_folder.resolve()),
            filetypes=self.MODEL_FILE_TYPES,
        )
        if not selected:
            return

        checkpoint_paths = [Path(path) for path in selected]
        if len(checkpoint_paths) < 2:
            messagebox.showerror("Merge Models", "Select at least two checkpoints.")
            return

        strategy = simpledialog.askstring(
            "Merge Models",
            "Enter merge strategy: mean, weighted, or slerp",
            initialvalue="mean",
            parent=self.root,
        )
        if strategy is None:
            return
        strategy = strategy.strip().lower()
        if strategy not in {"mean", "weighted", "slerp"}:
            messagebox.showerror("Merge Models", "Strategy must be mean, weighted, or slerp.")
            return

        weights: list[float] | None = None
        if strategy == "weighted":
            raw_weights = simpledialog.askstring(
                "Merge Models",
                "Enter one comma-separated weight per selected checkpoint.",
                parent=self.root,
            )
            if raw_weights is None:
                return
            try:
                weights = [float(value.strip()) for value in raw_weights.split(",") if value.strip()]
            except ValueError:
                messagebox.showerror("Merge Models", "Weights must be numeric values.")
                return
            if len(weights) != len(checkpoint_paths):
                messagebox.showerror(
                    "Merge Models",
                    f"Expected {len(checkpoint_paths)} weights, received {len(weights)}.",
                )
                return

        output_path = models_folder / f"merged_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"
        try:
            merged_path = merge_checkpoints(
                checkpoint_paths=checkpoint_paths,
                output_path=output_path,
                strategy=strategy,
                weights=weights,
            )
            if self.model_map_window is not None and self.model_map_window.winfo_exists():
                self._refresh_model_map()
            self.status_var.set(f"Merged {len(checkpoint_paths)} models -> {merged_path.name}")
            messagebox.showinfo("Merge Models", f"Merged checkpoint saved to:\n{merged_path.resolve()}")
        except Exception as exc:
            messagebox.showerror("Merge Models", str(exc))

    def _describe_scene(self, scene: ParsedScene | LayeredScene) -> str:
        if isinstance(scene, LayeredScene):
            labels = []
            for layer_name, scene_object in (
                ("background", scene.background),
                ("midground", scene.midground),
                ("frontground", scene.frontground),
            ):
                if scene_object is not None:
                    labels.append(f"{layer_name}:{scene_object.noun}")
            return ", ".join(labels) if labels else scene.prompt
        return " + ".join(obj.noun for obj in scene.objects)

    def _toggle_chaos(self):
        self.personality_var.set("Chaotic")
        self._apply_personality_preset()
        self.var_scale.set(random.uniform(12.0, 20.0))
        self.iterations_var.set(random.randint(0, 5))
        self.blend_count_var.set(random.randint(2, 6))
        self._generate()

    def _generate_image(self, model):
        """Placeholder handoff for prompt-selected models."""
        if model is None:
            raise ValueError("No model is loaded.")
        self._generate()

    def _stop_model_cycle(self):
        self.model_cycle_active = False

    def _shuffle_model_cycle_queue(self):
        self.model_cycle_queue = list(self.model_cycle_paths)
        random.shuffle(self.model_cycle_queue)

    def _toggle_model_cycle(self):
        if self.model_cycle_active:
            self._stop_model_cycle()
            self.status_var.set("Model shuffle stopped.")
            return

        use_folder = messagebox.askyesnocancel(
            "Model Shuffle",
            "Choose models from a folder?\n\nYes = select a folder\nNo = pick specific model files",
            parent=self.root,
        )
        if use_folder is None:
            return

        selected_paths: list[Path] = []
        if use_folder:
            folder = filedialog.askdirectory(
                title="Select folder with model files",
                parent=self.root,
            )
            if not folder:
                return
            selected_paths = list_model_paths(Path(folder))
        else:
            paths = filedialog.askopenfilenames(
                title="Select model files",
                filetypes=self.MODEL_FILE_TYPES,
                parent=self.root,
            )
            if not paths:
                return
            selected_paths = [Path(p) for p in paths]

        if not selected_paths:
            messagebox.showerror("Model Shuffle", "No model files were found or selected.")
            return

        self.model_cycle_paths = sorted(selected_paths, key=lambda path: path.name.lower())
        self._shuffle_model_cycle_queue()
        self.model_cycle_active = True
        self.auto_cycle_var.set(False)
        self.dream_cycle_var.set(False)
        self.status_var.set(
            f"Model shuffle started with {len(self.model_cycle_paths)} model(s)."
        )
        self._model_cycle_loop()

    def _model_cycle_loop(self):
        if not self.model_cycle_active:
            return
        if not self.model_cycle_queue:
            if not self.model_cycle_paths:
                self._stop_model_cycle()
                self.status_var.set("Model shuffle stopped: no model files available.")
                return
            self._shuffle_model_cycle_queue()

        model_path = self.model_cycle_queue.pop(0)
        try:
            self._load_model_file(model_path)
            self._generate()
            remaining = len(self.model_cycle_queue)
            self.status_var.set(
                f"Model shuffle: {model_path.name} | Remaining this round: {remaining}"
            )
        except Exception as exc:
            self.status_var.set(f"Model shuffle skipped {model_path.name}: {exc}")

        if self.model_cycle_active:
            self.root.after(self.model_cycle_delay_ms, self._model_cycle_loop)

    def _toggle_auto_cycle(self):
        if self.auto_cycle_var.get():
            self._stop_model_cycle()
            self.dream_cycle_var.set(False)
            self._auto_generate_loop()

    def _toggle_dream_cycle(self):
        if self.dream_cycle_var.get():
            self._stop_model_cycle()
            self.auto_cycle_var.set(False)
            self.current_latent = None
            self.target_latent = None
            self.interpolation_step = 0
            self._dream_cycle_loop()

    def _auto_generate_loop(self):
        if not self.auto_cycle_var.get() or self.model is None: return
        self._generate()
        self.root.after(2000, self._auto_generate_loop)

    def _get_blended_anchor(self, blend_count: int):
        if self.training_tensors is None or self.training_tensors.size(0) == 0:
            return None

        total = self.training_tensors.size(0)
        count = max(2, min(int(blend_count), 64))
        if total >= count:
            # indices must live on the same device as the tensor we index
            idx = torch.randperm(total, device=self.training_tensors.device)[:count]
        else:
            idx = torch.randint(0, total, (count,), device=self.training_tensors.device)
        anchors = self.training_tensors[idx].to(
            self.device,
            non_blocking=(getattr(self.device, "type", "") == "cuda"),
        )
        mu, _ = self.model.encode(anchors)
        weights = torch.rand(count, device=mu.device)
        weights = weights / weights.sum()
        return (mu * weights.unsqueeze(1)).sum(dim=0, keepdim=True)

    def _get_memory_anchor(self):
        memories = self.memory_bank.load_memories(limit=12)
        prompt_tag = self.generation_prompt_var.get().strip().lower()
        if prompt_tag:
            tagged = [record for record in memories if prompt_tag and prompt_tag in record.prompt.lower()]
            if tagged:
                memories = tagged
        if not memories:
            return None
        record = random.choice(memories[: max(1, min(6, len(memories)))])
        try:
            return self.memory_bank.load_latent(record).to(self.device)
        except Exception:
            return None

    def _get_random_latent(self):
        intensity = self.var_scale.get() / 10.0
        personality = self.personality_var.get()
        blend_enabled = bool(self.blend_mode_var.get())
        blend_count = max(2, int(self.blend_count_var.get()))
        with torch.no_grad():
            if personality in {"Dreamy", "Hybrid"}:
                memory_anchor = self._get_memory_anchor()
                if memory_anchor is not None:
                    noise = torch.randn_like(memory_anchor, device=memory_anchor.device)
                    return memory_anchor + (noise * max(0.05, intensity * 0.65))

            if blend_enabled:
                blended = self._get_blended_anchor(blend_count)
                if blended is not None:
                    noise = torch.randn_like(blended, device=blended.device)
                    return blended + (intensity * noise)

            if self.training_tensors is not None and self.training_tensors.size(0) > 0:
                total_items = self.training_tensors.size(0)
                if personality == "Nostalgic" and total_items > 1:
                    nostalgic_span = max(1, total_items // 3)
                    idx = torch.randint(0, nostalgic_span, (1,), device=self.training_tensors.device)
                else:
                    idx = torch.randint(
                        0,
                        total_items,
                        (1,),
                        device=self.training_tensors.device,
                    )
                img_batch = self.training_tensors[idx].to(
                    self.device,
                    non_blocking=(getattr(self.device, "type", "") == "cuda"),
                )
                mu, _ = self.model.encode(img_batch)
                noise = torch.randn_like(mu, device=mu.device)
                if personality == "Chaotic":
                    noise = noise * 1.6
                elif personality == "Dreamy":
                    noise = noise * 0.6
                return mu + (intensity * noise)
            else:
                if personality == "Corruption":
                    intensity *= 1.4
                return torch.randn(1, self.model.latent_dim, device=self.device) * intensity

    def _compose_labeled_grid(self, images, columns: int = 3):
        if not images:
            raise ValueError("No images to compose.")
        width, height = images[0].size
        columns = max(1, columns)
        rows = math.ceil(len(images) / columns)
        pad = 12
        label_h = 28
        canvas = Image.new(
            "RGB",
            (columns * width + (columns + 1) * pad, rows * (height + label_h) + (rows + 1) * pad),
            color=(8, 10, 18),
        )
        draw = ImageDraw.Draw(canvas)
        for idx, image in enumerate(images, start=1):
            row = (idx - 1) // columns
            col = (idx - 1) % columns
            x = pad + col * (width + pad)
            y = pad + row * (height + label_h + pad)
            img = image.resize((width, height), Image.Resampling.LANCZOS) if image.size != (width, height) else image
            canvas.paste(img, (x, y))
            draw.text((x, y + height + 6), f"#{idx}", fill=(235, 240, 255))
        return canvas

    def _render_latent_gallery(
        self,
        latents,
        *,
        mode: str,
        numbered: bool = False,
        status_label: str = "",
        save_memory: bool = True,
    ):
        images = []
        self.last_generated_latents = []

        with torch.no_grad():
            for latent in latents:
                latent_device = latent.to(self.device)
                recon = self._decode_latent(
                    latent_device,
                    show_steps=(len(latents) == 1 and mode == "generate" and self.show_iterations_var.get()),
                )
                image = tensor_to_pil(recon)
                images.append(image)
                stored_latent = latent_device.detach().cpu()
                self.last_generated_latents.append(stored_latent)
                if save_memory:
                    self._remember_generation(
                        image,
                        stored_latent,
                        mode=mode,
                        extra={
                            "diffusion_steps": int(self.diffusion_steps_var.get()),
                            "diffusion_strength": float(self.diffusion_strength_var.get()),
                            "iterations": int(self.iterations_var.get()),
                        },
                    )

        if len(images) == 1:
            final_image = images[0]
        elif numbered:
            final_image = self._compose_labeled_grid(images)
        else:
            final_image = self._compose_side_by_side(images)

        self._display_image(final_image)
        if status_label:
            self.status_var.set(status_label)
        return images

    def _dream_cycle_loop(self):
        if not self.dream_cycle_var.get() or self.model is None: return

        # Initialize or Pick New Target
        if self.current_latent is None:
            self.current_latent = self._get_random_latent()
            self.target_latent = self._get_random_latent()
            self.interpolation_step = 0
        
        # When we finish one segment, target becomes the new start
        if self.interpolation_step >= self.total_interpolation_steps:
            self.current_latent = self.target_latent
            self.target_latent = self._get_random_latent()
            self.interpolation_step = 0

        # Update smoothness/speed from slider live
        self.total_interpolation_steps = int(self.speed_scale.get())
        
        # Calculate alpha
        alpha = self.interpolation_step / self.total_interpolation_steps
        
        with torch.no_grad():
            # Linear Slerp transition
            interp_latent = slerp(alpha, self.current_latent, self.target_latent)
            recon = self._decode_latent(interp_latent, show_steps=False)
            image = tensor_to_pil(recon)
            self._display_image(image)
            if self.interpolation_step == 0:
                self.last_generated_latents = [interp_latent.detach().cpu()]
                self._remember_generation(image, interp_latent.detach().cpu(), mode="dream_journal")

        self.interpolation_step += 1
        self.status_var.set("Dreaming Cycle active...")
        
        fps = max(1, int(self.dream_fps_var.get()))
        delay_ms = max(16, int(1000 / fps))
        self.root.after(delay_ms, self._dream_cycle_loop)

    def _generate(self):
        if self.model is None:
            messagebox.showerror("Error", "Load/Train model first.")
            return

        intensity = self.var_scale.get()
        iterations = max(0, int(self.iterations_var.get()))
        show_steps = self.show_iterations_var.get()
        output_count = max(1, min(8, int(self.output_count_var.get())))
        blend_enabled = bool(self.blend_mode_var.get())
        blend_count = max(2, int(self.blend_count_var.get()))
        use_diffusion = bool(self.use_mini_diffusion_var.get())
        diffusion_steps = max(1, int(self.diffusion_steps_var.get()))
        latents = []
        
        with torch.no_grad():
            for _ in range(output_count):
                latents.append(self._get_random_latent())

        self._render_latent_gallery(
            latents,
            mode=self._current_mode_label(),
            numbered=False,
            status_label="",
            save_memory=not self.auto_cycle_var.get(),
        )
        if not self.auto_cycle_var.get() and not self.dream_cycle_var.get():
            mode = f"Blend x{blend_count}" if blend_enabled else "Single anchor"
            self.status_var.set(
                f"Generated ({self.personality_var.get()} | Intensity: {intensity:.1f} | {mode} | Diffusion: {'on' if use_diffusion else 'off'} x{diffusion_steps} | Outputs: {output_count})"
            )

    def _decode_latent(self, z, show_steps: bool = False):
        current = z
        if self.use_mini_diffusion_var.get():
            current = self._mini_diffusion_refine(current, show_steps=show_steps)

        recon = self.model.decode(current)
        iterations = max(0, int(self.iterations_var.get()))
        for _step in range(iterations):
            mu_step, _ = self.model.encode(recon)
            recon = self.model.decode(mu_step)
            if show_steps:
                self._display_image(tensor_to_pil(recon))
                self.root.update()
                self.root.after(50)
        return recon

    def _mini_diffusion_refine(self, z, show_steps: bool = False):
        steps = max(1, int(self.diffusion_steps_var.get()))
        strength = max(0.05, float(self.diffusion_strength_var.get()))
        intensity_scale = max(0.2, min(2.0, self.var_scale.get() / 10.0))
        current = z.clone()

        for step_idx in range(steps):
            if steps == 1:
                t_value = 1.0
            else:
                t_value = 1.0 - (step_idx / (steps - 1))
            t = torch.full((current.size(0), 1), t_value, device=current.device)
            predicted_noise = self.model.predict_latent_noise(current, t)
            step_scale = strength * intensity_scale * (0.2 + 0.8 * t_value)
            current = current - (predicted_noise * step_scale)

            if step_idx < steps - 1:
                residual_scale = 0.03 * intensity_scale * t_value
                current = current + (torch.randn_like(current) * residual_scale)

            if show_steps:
                preview = self.model.decode(current)
                self._display_image(tensor_to_pil(preview))
                self.root.update()
                self.root.after(50)

        return current

    def _compose_side_by_side(self, images):
        if not images:
            raise ValueError("No images to compose.")
        width, height = images[0].size
        canvas = Image.new("RGB", (width * len(images), height), color=(8, 8, 16))
        x_offset = 0
        for image in images:
            if image.size != (width, height):
                image = image.resize((width, height), Image.Resampling.LANCZOS)
            canvas.paste(image, (x_offset, 0))
            x_offset += width
        return canvas

    def _display_image(self, pil_img):
        from PIL import ImageTk, Image
        pil_img = pil_img.resize((512, 512), Image.Resampling.LANCZOS)
        self.photo = ImageTk.PhotoImage(pil_img)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)

        ow = self.output_canvas
        if ow is not None:
            try:
                win = self.output_window
                if win is not None and win.winfo_exists():
                    self._output_photo = ImageTk.PhotoImage(pil_img)
                    ow.delete("all")
                    ow.create_image(0, 0, anchor=tk.NW, image=self._output_photo)
            except tk.TclError:
                self.output_window = None
                self.output_canvas = None
                self._output_photo = None

def main():
    app = APVDApp()
    app.root.mainloop()

if __name__ == "__main__":
    main()

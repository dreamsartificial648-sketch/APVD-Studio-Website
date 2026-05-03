from __future__ import annotations

import base64
import json
import os
import random
import tarfile
import threading
import time
import uuid
import zipfile
from dataclasses import dataclass, field
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import torch
from flask import Flask, jsonify, render_template, request, send_from_directory
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, Dataset
from werkzeug.datastructures import FileStorage
from werkzeug.utils import secure_filename

from model import VAE, get_device, latent_denoiser_loss, vae_loss
from utils import IMAGE_EXTENSIONS, MODEL_EXTENSIONS, get_image_paths, list_model_paths, tensor_to_pil

ImageFile.LOAD_TRUNCATED_IMAGES = True

BASE_DIR = Path(__file__).resolve().parent
STORAGE_DIR = BASE_DIR / "web_storage"
DATASET_DIR = STORAGE_DIR / "datasets"
MODEL_DIR = STORAGE_DIR / "models"
GENERATED_DIR = STORAGE_DIR / "generated"
SHARED_DIR = STORAGE_DIR / "shared"
SHARES_FILE = STORAGE_DIR / "shares.json"
DESKTOP_MODEL_DIR = BASE_DIR / "Models"

for folder in (DATASET_DIR, MODEL_DIR, GENERATED_DIR, SHARED_DIR):
    folder.mkdir(parents=True, exist_ok=True)

app = Flask(__name__, template_folder="templates", static_folder="static")
app.config["MAX_CONTENT_LENGTH"] = 1024 * 1024 * 1024


PERSONALITY_PRESETS: dict[str, dict[str, Any]] = {
    "Manual": {
        "intensity": 1.0,
        "iterations": 3,
        "use_diffusion": True,
        "diffusion_steps": 8,
        "diffusion_strength": 0.85,
    },
    "Dreamy": {
        "intensity": 0.8,
        "iterations": 2,
        "use_diffusion": True,
        "diffusion_steps": 10,
        "diffusion_strength": 0.55,
    },
    "Chaotic": {
        "intensity": 8.5,
        "iterations": 0,
        "use_diffusion": True,
        "diffusion_steps": 14,
        "diffusion_strength": 1.25,
    },
    "Nostalgic": {
        "intensity": 0.9,
        "iterations": 4,
        "use_diffusion": True,
        "diffusion_steps": 7,
        "diffusion_strength": 0.7,
    },
    "Hybrid": {
        "intensity": 2.2,
        "iterations": 3,
        "use_diffusion": True,
        "diffusion_steps": 12,
        "diffusion_strength": 0.95,
    },
    "Corruption": {
        "intensity": 6.0,
        "iterations": 10,
        "use_diffusion": True,
        "diffusion_steps": 18,
        "diffusion_strength": 1.35,
    },
}


@dataclass
class TrainingJob:
    id: str
    created_at: str
    status: str = "queued"
    progress: float = 0.0
    message: str = "Queued"
    error: str | None = None
    model_id: str | None = None
    model_name: str | None = None
    dataset_count: int = 0
    settings: dict[str, Any] = field(default_factory=dict)
    loss: float | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "created_at": self.created_at,
            "status": self.status,
            "progress": round(self.progress, 2),
            "message": self.message,
            "error": self.error,
            "model_id": self.model_id,
            "model_name": self.model_name,
            "dataset_count": self.dataset_count,
            "settings": self.settings,
            "loss": self.loss,
        }


TRAINING_JOBS: dict[str, TrainingJob] = {}
TRAINING_LOCK = threading.Lock()
MODEL_CACHE: dict[str, tuple[float, VAE, torch.device]] = {}
MODEL_CACHE_LOCK = threading.Lock()
SHARE_LOCK = threading.Lock()


class ImagePathDataset(Dataset):
    def __init__(self, image_paths: list[Path], resolution: int):
        self.image_paths = [Path(path) for path in image_paths]
        self.target_size = (int(resolution), int(resolution))

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> torch.Tensor:
        path = self.image_paths[index]
        try:
            with Image.open(path) as img:
                return pil_to_tensor(img, self.target_size)
        except Exception:
            return torch.zeros(3, self.target_size[1], self.target_size[0], dtype=torch.float32)


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def pil_to_tensor(image: Image.Image, target_size: tuple[int, int]) -> torch.Tensor:
    rgb = image.convert("RGB").resize(target_size, Image.Resampling.LANCZOS)
    arr = np.asarray(rgb, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


def encode_model_id(path: Path) -> str:
    raw = str(path.resolve()).encode("utf-8")
    return base64.urlsafe_b64encode(raw).decode("ascii")


def decode_model_id(model_id: str) -> Path:
    try:
        raw = base64.urlsafe_b64decode(model_id.encode("ascii")).decode("utf-8")
    except Exception as exc:
        raise ValueError("Invalid model id.") from exc
    path = Path(raw).resolve()
    allowed_roots = [MODEL_DIR.resolve(), DESKTOP_MODEL_DIR.resolve()]
    if not any(path == root or root in path.parents for root in allowed_roots):
        raise ValueError("Model path is outside allowed model folders.")
    if path.suffix.lower() not in MODEL_EXTENSIONS or not path.is_file():
        raise ValueError("Model file was not found.")
    return path


def is_supported_image(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTENSIONS


def is_supported_archive(path: Path) -> bool:
    name = path.name.lower()
    return name.endswith(".zip") or name.endswith(".tar") or name.endswith(".tar.gz") or name.endswith(".tgz")


def is_safe_member(name: str) -> bool:
    normalized = name.replace("\\", "/").strip("/")
    if not normalized:
        return False
    parts = normalized.split("/")
    if ".." in parts:
        return False
    if parts[0].endswith(":"):
        return False
    return True


def unique_path(folder: Path, filename: str) -> Path:
    safe = secure_filename(filename) or f"upload-{uuid.uuid4().hex}"
    path = folder / safe
    if not path.exists():
        return path
    stem = path.stem
    suffix = path.suffix
    return folder / f"{stem}-{uuid.uuid4().hex[:8]}{suffix}"


def save_uploads(files: list[FileStorage], dataset_folder: Path) -> list[Path]:
    upload_folder = dataset_folder / "uploads"
    images_folder = dataset_folder / "images"
    upload_folder.mkdir(parents=True, exist_ok=True)
    images_folder.mkdir(parents=True, exist_ok=True)
    saved_images: list[Path] = []

    for upload in files:
        if not upload or not upload.filename:
            continue
        target = unique_path(upload_folder, upload.filename)
        upload.save(target)
        if is_supported_image(target):
            image_target = unique_path(images_folder, target.name)
            target.replace(image_target)
            saved_images.append(image_target)
        elif is_supported_archive(target):
            saved_images.extend(extract_archive_images(target, images_folder))

    return sorted(saved_images)


def extract_archive_images(archive_path: Path, output_folder: Path) -> list[Path]:
    output_folder.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    lower = archive_path.name.lower()

    if lower.endswith(".zip"):
        with zipfile.ZipFile(archive_path, "r") as zf:
            for info in zf.infolist():
                if info.is_dir() or not is_safe_member(info.filename):
                    continue
                source_name = Path(info.filename).name
                if not is_supported_image(Path(source_name)):
                    continue
                with zf.open(info, "r") as src:
                    image_bytes = src.read()
                saved.append(write_image_bytes(image_bytes, output_folder, source_name))
    else:
        with tarfile.open(archive_path, "r:*") as tf:
            for member in tf.getmembers():
                if not member.isfile() or not is_safe_member(member.name):
                    continue
                source_name = Path(member.name).name
                if not is_supported_image(Path(source_name)):
                    continue
                src = tf.extractfile(member)
                if src is None:
                    continue
                saved.append(write_image_bytes(src.read(), output_folder, source_name))

    return saved


def write_image_bytes(image_bytes: bytes, output_folder: Path, filename: str) -> Path:
    with Image.open(BytesIO(image_bytes)) as img:
        img.verify()
    target = unique_path(output_folder, filename)
    target.write_bytes(image_bytes)
    return target


def coerce_int(value: Any, default: int, min_value: int, max_value: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    return max(min_value, min(max_value, parsed))


def coerce_float(value: Any, default: float, min_value: float, max_value: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = default
    return max(min_value, min(max_value, parsed))


def bool_value(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).lower() in {"1", "true", "yes", "on"}


def update_job(job_id: str, **changes: Any) -> None:
    with TRAINING_LOCK:
        job = TRAINING_JOBS[job_id]
        for key, value in changes.items():
            setattr(job, key, value)


def run_training_job(job_id: str, image_paths: list[Path], settings: dict[str, Any]) -> None:
    try:
        device = get_device()
        resolution = int(settings["resolution"])
        latent_dim = int(settings["latent_dim"])
        epochs = int(settings["epochs"])
        batch_size = int(settings["batch_size"])
        learning_rate = float(settings["learning_rate"])

        update_job(
            job_id,
            status="running",
            progress=0.0,
            message=f"Training on {len(image_paths)} image(s) with {device.type.upper()}",
            dataset_count=len(image_paths),
        )

        dataset = ImagePathDataset(image_paths, resolution)
        if len(dataset) == 0:
            raise ValueError("No valid training images were uploaded.")

        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        model = VAE(latent_dim=latent_dim, output_size=(resolution, resolution)).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        amp_enabled = device.type == "cuda"
        scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
        total_batches = max(1, len(loader))
        total_steps = max(1, epochs * total_batches)
        completed_steps = 0
        started = time.perf_counter()
        model.train()
        last_loss = 0.0

        for epoch in range(epochs):
            epoch_loss = 0.0
            for batch_index, batch in enumerate(loader, start=1):
                batch = batch.to(device, non_blocking=amp_enabled)
                optimizer.zero_grad(set_to_none=True)

                with torch.amp.autocast("cuda", enabled=amp_enabled):
                    recon, mu, logvar = model(batch)
                with torch.amp.autocast("cuda", enabled=False):
                    recon_loss = vae_loss(recon.float(), batch.float(), mu.float(), logvar.float())
                    denoise_loss = latent_denoiser_loss(model, mu.detach().float())
                    loss = (recon_loss / max(1, batch.size(0))) + (0.25 * denoise_loss)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                loss_value = float(loss.detach().float().item())
                epoch_loss += loss_value
                completed_steps += 1
                progress = (completed_steps / total_steps) * 100.0
                elapsed = max(0.001, time.perf_counter() - started)
                eta_seconds = ((total_steps - completed_steps) / completed_steps) * elapsed
                last_loss = epoch_loss / max(1, batch_index)

                update_job(
                    job_id,
                    progress=progress,
                    loss=last_loss,
                    message=(
                        f"Epoch {epoch + 1}/{epochs} | Batch {batch_index}/{total_batches} | "
                        f"Loss {last_loss:.0f} | ETA {format_duration(eta_seconds)}"
                    ),
                )

        model.eval()
        model_name = f"apvd-web-{job_id}.pt"
        model_path = MODEL_DIR / model_name
        metadata = {
            "dataset_label": f"web-{job_id}",
            "dataset_image_count": len(dataset),
            "epochs": epochs,
            "total_epochs": epochs,
            "resolution": [resolution, resolution],
            "latent_dim": latent_dim,
            "source": "APVD web trainer",
            "saved_at": now_iso(),
            "version": "web-mvp-mini-diffusion",
        }
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "latent_dim": latent_dim,
                "output_size": (resolution, resolution),
                "training_metadata": metadata,
                "version": metadata["version"],
            },
            model_path,
        )
        if device.type == "cuda":
            torch.cuda.empty_cache()

        update_job(
            job_id,
            status="complete",
            progress=100.0,
            message="Training complete",
            model_id=encode_model_id(model_path),
            model_name=model_name,
            loss=last_loss,
        )
    except Exception as exc:
        update_job(job_id, status="failed", error=str(exc), message="Training failed")


def format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    minutes, sec = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes}m"
    if minutes:
        return f"{minutes}m {sec}s"
    return f"{sec}s"


def list_available_models() -> list[dict[str, Any]]:
    paths: list[Path] = []
    paths.extend(list_model_paths(MODEL_DIR))
    paths.extend(list_model_paths(DESKTOP_MODEL_DIR))
    models = []
    for path in sorted(set(paths), key=lambda p: p.stat().st_mtime if p.exists() else 0, reverse=True):
        models.append(
            {
                "id": encode_model_id(path),
                "name": path.name,
                "source": "web" if MODEL_DIR.resolve() in path.resolve().parents else "desktop",
                "modified_at": datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="seconds"),
                "size_mb": round(path.stat().st_size / (1024 * 1024), 2),
            }
        )
    return models


def load_model(model_id: str) -> tuple[VAE, torch.device, Path]:
    path = decode_model_id(model_id)
    device = get_device()
    cache_key = f"{path.resolve()}::{device.type}"
    mtime = path.stat().st_mtime

    with MODEL_CACHE_LOCK:
        cached = MODEL_CACHE.get(cache_key)
        if cached and cached[0] == mtime:
            return cached[1], cached[2], path

    checkpoint = torch.load(path, map_location=device)
    if not isinstance(checkpoint, dict):
        raise ValueError("Unsupported checkpoint format.")
    output_size = tuple(checkpoint.get("output_size", (256, 256)))
    latent_dim = int(checkpoint.get("latent_dim", 256))
    model = VAE(latent_dim=latent_dim, output_size=output_size).to(device)
    state_dict = checkpoint.get("model_state_dict")
    if state_dict is None:
        raise ValueError("Checkpoint does not contain model_state_dict.")
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    with MODEL_CACHE_LOCK:
        MODEL_CACHE[cache_key] = (mtime, model, device)

    return model, device, path


def refine_latent(
    model: VAE,
    z: torch.Tensor,
    *,
    steps: int,
    strength: float,
    intensity: float,
) -> torch.Tensor:
    current = z.clone()
    intensity_scale = max(0.2, min(2.0, intensity / 10.0))
    for step_idx in range(max(1, steps)):
        t_value = 1.0 if steps <= 1 else 1.0 - (step_idx / (steps - 1))
        t = torch.full((current.size(0), 1), t_value, device=current.device)
        predicted_noise = model.predict_latent_noise(current, t)
        step_scale = strength * intensity_scale * (0.2 + 0.8 * t_value)
        current = current - (predicted_noise * step_scale)
        if step_idx < steps - 1:
            residual_scale = 0.03 * intensity_scale * t_value
            current = current + (torch.randn_like(current) * residual_scale)
    return current


def decode_generation(
    model: VAE,
    z: torch.Tensor,
    *,
    use_diffusion: bool,
    diffusion_steps: int,
    diffusion_strength: float,
    intensity: float,
    iterations: int,
) -> torch.Tensor:
    current = z
    if use_diffusion:
        current = refine_latent(
            model,
            current,
            steps=diffusion_steps,
            strength=diffusion_strength,
            intensity=intensity,
        )
    recon = model.decode(current)
    for _ in range(max(0, iterations)):
        mu_step, _ = model.encode(recon)
        recon = model.decode(mu_step)
    return recon


def build_latent(
    model: VAE,
    device: torch.device,
    *,
    anchor_image: FileStorage | None,
    intensity: float,
    personality: str,
) -> torch.Tensor:
    noise_scale = intensity / 10.0
    if personality == "Chaotic":
        noise_scale *= 1.6
    elif personality == "Dreamy":
        noise_scale *= 0.6
    elif personality == "Corruption":
        noise_scale *= 1.4

    with torch.no_grad():
        if anchor_image and anchor_image.filename:
            try:
                anchor_image.stream.seek(0)
            except Exception:
                pass
            with Image.open(anchor_image.stream) as img:
                anchor = pil_to_tensor(img, model.output_size).unsqueeze(0).to(device)
            mu, _ = model.encode(anchor)
            return mu + (torch.randn_like(mu) * max(0.01, noise_scale))

        return torch.randn(1, model.latent_dim, device=device) * max(0.01, noise_scale)


def save_generated_image(image: Image.Image) -> dict[str, str]:
    image_id = uuid.uuid4().hex
    filename = f"{image_id}.png"
    image.save(GENERATED_DIR / filename)
    return {"id": image_id, "url": f"/generated/{filename}", "filename": filename}


def load_shares() -> list[dict[str, Any]]:
    if not SHARES_FILE.exists():
        return []
    try:
        with SHARES_FILE.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
    except Exception:
        return []
    return data if isinstance(data, list) else []


def save_shares(records: list[dict[str, Any]]) -> None:
    STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    with SHARES_FILE.open("w", encoding="utf-8") as fh:
        json.dump(records, fh, indent=2)


@app.get("/")
def index():
    return render_template("index.html", presets=PERSONALITY_PRESETS)


@app.get("/api/models")
def api_models():
    return jsonify({"models": list_available_models()})


@app.post("/api/train")
def api_train():
    files = request.files.getlist("dataset")
    if not files:
        return jsonify({"error": "Upload at least one image or archive."}), 400

    job_id = uuid.uuid4().hex[:12]
    dataset_folder = DATASET_DIR / job_id
    dataset_folder.mkdir(parents=True, exist_ok=True)
    image_paths = save_uploads(files, dataset_folder)
    if not image_paths:
        return jsonify({"error": "No supported images were found in the upload."}), 400

    settings = {
        "epochs": coerce_int(request.form.get("epochs"), 5, 1, 5000),
        "resolution": coerce_int(request.form.get("resolution"), 256, 32, 1024),
        "latent_dim": coerce_int(request.form.get("latent_dim"), 512, 32, 2048),
        "batch_size": coerce_int(request.form.get("batch_size"), 8, 1, 128),
        "learning_rate": coerce_float(request.form.get("learning_rate"), 2e-4, 1e-6, 1e-2),
    }
    job = TrainingJob(
        id=job_id,
        created_at=now_iso(),
        status="queued",
        progress=0.0,
        message=f"Queued {len(image_paths)} image(s)",
        dataset_count=len(image_paths),
        settings=settings,
    )
    with TRAINING_LOCK:
        TRAINING_JOBS[job_id] = job

    thread = threading.Thread(target=run_training_job, args=(job_id, image_paths, settings), daemon=True)
    thread.start()
    return jsonify({"job": job.as_dict()}), 202


@app.get("/api/train/<job_id>")
def api_train_status(job_id: str):
    with TRAINING_LOCK:
        job = TRAINING_JOBS.get(job_id)
        if job is None:
            return jsonify({"error": "Unknown training job."}), 404
        return jsonify({"job": job.as_dict()})


@app.post("/api/generate")
def api_generate():
    model_id = request.form.get("model_id", "")
    if not model_id:
        return jsonify({"error": "Select or train a model first."}), 400

    model, device, model_path = load_model(model_id)
    personality = request.form.get("personality", "Manual")
    preset = PERSONALITY_PRESETS.get(personality, PERSONALITY_PRESETS["Manual"])
    intensity = coerce_float(request.form.get("intensity"), float(preset["intensity"]), 0.1, 20.0)
    iterations = coerce_int(request.form.get("iterations"), int(preset["iterations"]), 0, 20)
    output_count = coerce_int(request.form.get("output_count"), 1, 1, 8)
    use_diffusion = bool_value(request.form.get("use_diffusion"), bool(preset["use_diffusion"]))
    diffusion_steps = coerce_int(request.form.get("diffusion_steps"), int(preset["diffusion_steps"]), 1, 40)
    diffusion_strength = coerce_float(
        request.form.get("diffusion_strength"),
        float(preset["diffusion_strength"]),
        0.05,
        3.0,
    )
    seed_raw = request.form.get("seed", "").strip()
    seed = int(seed_raw) if seed_raw.isdigit() else random.randint(1, 2_147_483_647)
    torch.manual_seed(seed)
    random.seed(seed)

    anchor_image = request.files.get("anchor")
    prompt = request.form.get("prompt", "").strip()
    outputs: list[dict[str, str]] = []

    with torch.no_grad():
        for _ in range(output_count):
            latent = build_latent(
                model,
                device,
                anchor_image=anchor_image,
                intensity=intensity,
                personality=personality,
            )
            recon = decode_generation(
                model,
                latent,
                use_diffusion=use_diffusion,
                diffusion_steps=diffusion_steps,
                diffusion_strength=diffusion_strength,
                intensity=intensity,
                iterations=iterations,
            )
            image = tensor_to_pil(recon)
            outputs.append(save_generated_image(image))

    return jsonify(
        {
            "images": outputs,
            "settings": {
                "model_id": model_id,
                "model_name": model_path.name,
                "prompt": prompt,
                "personality": personality,
                "intensity": intensity,
                "iterations": iterations,
                "output_count": output_count,
                "use_diffusion": use_diffusion,
                "diffusion_steps": diffusion_steps,
                "diffusion_strength": diffusion_strength,
                "seed": seed,
            },
        }
    )


@app.post("/api/share")
def api_share():
    payload = request.get_json(silent=True) or {}
    image_id = str(payload.get("image_id", "")).strip()
    filename = f"{image_id}.png"
    source = GENERATED_DIR / filename
    if not image_id or not source.is_file():
        return jsonify({"error": "Generated image was not found."}), 404

    record = {
        "id": uuid.uuid4().hex,
        "image_id": image_id,
        "image_url": f"/generated/{filename}",
        "title": str(payload.get("title") or "Untitled structure")[:80],
        "prompt": str(payload.get("prompt") or "")[:240],
        "model_name": str(payload.get("model_name") or "")[:120],
        "settings": payload.get("settings") if isinstance(payload.get("settings"), dict) else {},
        "created_at": now_iso(),
    }
    with SHARE_LOCK:
        records = load_shares()
        records.insert(0, record)
        save_shares(records[:200])
    return jsonify({"share": record}), 201


@app.get("/api/gallery")
def api_gallery():
    return jsonify({"shares": load_shares()[:80]})


@app.get("/generated/<path:filename>")
def generated_file(filename: str):
    return send_from_directory(GENERATED_DIR, filename)


@app.get("/health")
def health():
    return jsonify({"ok": True, "device": get_device().type})


if __name__ == "__main__":
    port = int(os.environ.get("APVD_WEB_PORT", "7860"))
    app.run(host="127.0.0.1", port=port, debug=False, threaded=True)

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
import torch


@dataclass
class MemoryRecord:
    memory_id: str
    timestamp: str
    prompt: str
    mode: str
    personality: str
    model_name: str
    image_path: str
    latent_path: str
    metadata: dict[str, Any]


class MemoryBank:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.images_dir = self.root / "images"
        self.latents_dir = self.root / "latents"
        self.index_path = self.root / "memories.jsonl"
        self.root.mkdir(parents=True, exist_ok=True)
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.latents_dir.mkdir(parents=True, exist_ok=True)

    def save_memory(
        self,
        image: Image.Image,
        latent: torch.Tensor,
        *,
        prompt: str,
        mode: str,
        personality: str,
        model_name: str,
        metadata: dict[str, Any] | None = None,
    ) -> MemoryRecord:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        memory_id = f"mem_{timestamp}"
        image_path = self.images_dir / f"{memory_id}.png"
        latent_path = self.latents_dir / f"{memory_id}.pt"

        image.save(image_path)
        torch.save(latent.detach().cpu(), latent_path)

        record = MemoryRecord(
            memory_id=memory_id,
            timestamp=datetime.now().isoformat(timespec="seconds"),
            prompt=prompt.strip(),
            mode=mode,
            personality=personality,
            model_name=model_name,
            image_path=str(image_path),
            latent_path=str(latent_path),
            metadata=metadata or {},
        )
        with self.index_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(asdict(record), ensure_ascii=True) + "\n")
        return record

    def load_memories(self, limit: int | None = None) -> list[MemoryRecord]:
        if not self.index_path.exists():
            return []

        records: list[MemoryRecord] = []
        for line in self.index_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
                records.append(MemoryRecord(**payload))
            except Exception:
                continue

        records.sort(key=lambda rec: rec.timestamp, reverse=True)
        if limit is not None:
            return records[: max(0, int(limit))]
        return records

    def get_record(self, memory_id: str) -> MemoryRecord | None:
        for record in self.load_memories():
            if record.memory_id == memory_id:
                return record
        return None

    def get_weighted_image_paths(
        self,
        *,
        limit: int = 32,
        recent_bias: float = 0.65,
        min_copies: int = 1,
        max_copies: int = 5,
    ) -> list[Path]:
        recent = list(reversed(self.load_memories(limit=max(1, limit))))
        if not recent:
            return []

        bias = float(max(0.0, min(1.0, recent_bias)))
        out: list[Path] = []
        total = len(recent)
        for idx, record in enumerate(recent):
            path = Path(record.image_path)
            if not path.exists():
                continue
            rank = idx / max(1, total - 1)
            copies = min_copies + int(round((bias + rank * (1.0 - bias)) * (max_copies - min_copies)))
            out.extend([path] * max(min_copies, copies))
        return out

    def load_latent(self, record: MemoryRecord) -> torch.Tensor:
        return torch.load(record.latent_path, map_location="cpu")

    def build_latent_map(
        self,
        *,
        limit: int = 64,
        canvas_size: tuple[int, int] = (560, 360),
    ) -> list[dict[str, Any]]:
        records = self.load_memories(limit=max(2, limit))
        vectors: list[np.ndarray] = []
        valid_records: list[MemoryRecord] = []

        for record in records:
            latent_path = Path(record.latent_path)
            if not latent_path.exists():
                continue
            try:
                latent = torch.load(latent_path, map_location="cpu")
                vector = latent.reshape(-1).detach().float().cpu().numpy()
            except Exception:
                continue
            vectors.append(vector)
            valid_records.append(record)

        if len(vectors) < 2:
            return []

        matrix = np.stack(vectors, axis=0)
        centered = matrix - matrix.mean(axis=0, keepdims=True)
        try:
            _, _, vt = np.linalg.svd(centered, full_matrices=False)
            projected = centered @ vt[:2].T
        except np.linalg.LinAlgError:
            projected = centered[:, :2]

        xs = projected[:, 0]
        ys = projected[:, 1]
        width, height = canvas_size
        x_min, x_max = float(xs.min()), float(xs.max())
        y_min, y_max = float(ys.min()), float(ys.max())
        x_span = max(1e-6, x_max - x_min)
        y_span = max(1e-6, y_max - y_min)

        points: list[dict[str, Any]] = []
        for idx, record in enumerate(valid_records):
            x_norm = (xs[idx] - x_min) / x_span
            y_norm = (ys[idx] - y_min) / y_span
            points.append(
                {
                    "record": record,
                    "x": 24 + x_norm * (width - 48),
                    "y": 24 + (1.0 - y_norm) * (height - 48),
                }
            )
        return points


def breed_latents(
    latents: list[torch.Tensor],
    *,
    noise_scale: float = 0.35,
    child_count: int = 4,
) -> list[torch.Tensor]:
    if not latents:
        return []

    flat = [latent.detach().clone() for latent in latents]
    out: list[torch.Tensor] = []
    for _ in range(max(1, child_count)):
        if len(flat) == 1:
            base = flat[0].clone()
        else:
            idx = np.random.choice(len(flat), size=min(2, len(flat)), replace=False)
            parent_a = flat[int(idx[0])]
            parent_b = flat[int(idx[-1])]
            mix = float(np.random.uniform(0.25, 0.75))
            base = (parent_a * mix) + (parent_b * (1.0 - mix))
        noise = torch.randn_like(base) * float(max(0.0, noise_scale))
        out.append(base + noise)
    return out


def summarize_memory(record: MemoryRecord) -> str:
    prompt = record.prompt.strip() or "untitled"
    prompt = prompt[:40] + ("..." if len(prompt) > 40 else "")
    return f"{record.memory_id} | {record.personality} | {prompt}"


def parse_selection_indices(raw_value: str, upper_bound: int) -> list[int]:
    tokens = [piece.strip() for piece in raw_value.replace(";", ",").split(",")]
    selected: list[int] = []
    for token in tokens:
        if not token:
            continue
        try:
            value = int(token)
        except ValueError:
            continue
        if 1 <= value <= upper_bound:
            idx = value - 1
            if idx not in selected:
                selected.append(idx)
    return selected

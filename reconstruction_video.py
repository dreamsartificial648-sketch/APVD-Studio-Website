from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math
import random
import re
import shutil
import subprocess
from typing import Callable, Iterable

import cv2
import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageEnhance, ImageFilter, ImageOps


ProgressCallback = Callable[[int, int], None]


@dataclass
class ReconstructionVideoResult:
    output_path: Path
    total_frames: int
    fps: int
    resolution: tuple[int, int]
    duration_seconds: float
    used_h264: bool
    audio_included: bool


def _natural_key(path: Path) -> list[object]:
    parts = re.split(r"(\d+)", path.name.lower())
    return [int(part) if part.isdigit() else part for part in parts]


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _lerp(a: float, b: float, t: float) -> float:
    return a + ((b - a) * t)


def _ease_in_out(t: float) -> float:
    t = _clamp(t, 0.0, 1.0)
    return t * t * (3.0 - (2.0 * t))


def _load_rgb_image(path: Path) -> Image.Image:
    with Image.open(path) as img:
        return ImageOps.exif_transpose(img).convert("RGB")


def _fit_cover(img: Image.Image, size: tuple[int, int]) -> Image.Image:
    return ImageOps.fit(img, size, method=Image.Resampling.LANCZOS, centering=(0.5, 0.5))


def _fit_contain(img: Image.Image, size: tuple[int, int]) -> Image.Image:
    out = img.copy()
    out.thumbnail(size, Image.Resampling.LANCZOS)
    return out


def _apply_zoom_pan(
    img: Image.Image,
    size: tuple[int, int],
    *,
    zoom: float,
    pan_x: float,
    pan_y: float,
) -> Image.Image:
    base = _fit_cover(img, size)
    if zoom <= 1.001:
        return base
    w, h = base.size
    scaled_w = max(w + 2, int(round(w * zoom)))
    scaled_h = max(h + 2, int(round(h * zoom)))
    scaled = base.resize((scaled_w, scaled_h), Image.Resampling.LANCZOS)
    max_x = max(0, scaled_w - w)
    max_y = max(0, scaled_h - h)
    left = int(round((max_x / 2.0) + (pan_x * max_x / 2.0)))
    top = int(round((max_y / 2.0) + (pan_y * max_y / 2.0)))
    left = max(0, min(left, max_x))
    top = max(0, min(top, max_y))
    return scaled.crop((left, top, left + w, top + h))


def _alpha_blend(a: Image.Image, b: Image.Image, alpha: float) -> Image.Image:
    return Image.blend(a, b, _clamp(alpha, 0.0, 1.0))


def _text(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, fill: tuple[int, int, int]):
    draw.text(xy, text, fill=fill)


def _apply_film_grain(img: Image.Image, strength: float, rng: random.Random) -> Image.Image:
    if strength <= 0:
        return img
    arr = np.asarray(img).astype(np.int16)
    noise = rng.normalvariate(0.0, 1.0)
    sigma = max(2.0, strength * 42.0) + abs(noise) * 2.0
    grain = np.random.normal(0.0, sigma, arr.shape).astype(np.int16)
    arr = np.clip(arr + grain, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, "RGB")


def _apply_scanlines(img: Image.Image, opacity: float = 0.1) -> Image.Image:
    arr = np.asarray(img).astype(np.float32)
    arr[::2, :, :] *= (1.0 - opacity)
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGB")


def _apply_glitch(img: Image.Image, rng: random.Random, intensity: float = 0.75) -> Image.Image:
    arr = np.asarray(img).copy()
    h, w, _ = arr.shape
    channel_shift = max(2, int(w * 0.008 * intensity))
    arr[:, :, 0] = np.roll(arr[:, :, 0], channel_shift, axis=1)
    arr[:, :, 2] = np.roll(arr[:, :, 2], -channel_shift, axis=1)
    band_count = rng.randint(4, 10)
    for _ in range(band_count):
        y = rng.randint(0, max(0, h - 8))
        band_h = rng.randint(3, max(4, h // 18))
        shift = rng.randint(-max(2, w // 25), max(2, w // 25))
        arr[y : y + band_h] = np.roll(arr[y : y + band_h], shift, axis=1)
    out = Image.fromarray(arr, "RGB")
    out = _apply_scanlines(out, opacity=0.08 + (0.08 * intensity))
    if rng.random() < 0.5:
        out = ImageEnhance.Contrast(out).enhance(1.15 + 0.25 * intensity)
    return out


def _apply_failure_frame(img: Image.Image, rng: random.Random) -> Image.Image:
    glitched = _apply_glitch(img, rng, intensity=1.2)
    if rng.random() < 0.5:
        glitched = ImageOps.posterize(glitched, bits=4)
    if rng.random() < 0.4:
        glitched = ImageOps.solarize(glitched, threshold=100)
    arr = np.asarray(glitched).astype(np.int16)
    noise = np.random.normal(0.0, 28.0, arr.shape).astype(np.int16)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, "RGB")


def _apply_motion_blur(img: Image.Image, amount: float) -> Image.Image:
    if amount <= 0.0:
        return img
    radius = max(0.0, amount * 2.4)
    return img.filter(ImageFilter.GaussianBlur(radius=radius))


def _add_vignette(img: Image.Image, strength: float) -> Image.Image:
    if strength <= 0:
        return img
    w, h = img.size
    y, x = np.ogrid[-1.0:1.0:h * 1j, -1.0:1.0:w * 1j]
    dist = np.sqrt((x * x) + (y * y))
    mask = np.clip((dist - 0.3) / 0.9, 0.0, 1.0)
    darken = 1.0 - (mask * strength)
    arr = np.asarray(img).astype(np.float32)
    arr *= darken[..., None]
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGB")


def _add_glow(img: Image.Image, strength: float) -> Image.Image:
    if strength <= 0:
        return img
    bloom = img.filter(ImageFilter.GaussianBlur(radius=14.0)).convert("RGB")
    return Image.blend(img, bloom, _clamp(strength, 0.0, 0.4))


def _compose_depth_scene(
    main_image: Image.Image,
    resolution: tuple[int, int],
    *,
    zoom: float,
    pan_x: float,
    pan_y: float,
    polaroid: bool = False,
    comparison_image: Image.Image | None = None,
    vignette_strength: float = 0.0,
    glow_strength: float = 0.0,
) -> Image.Image:
    w, h = resolution
    bg = _apply_zoom_pan(main_image, resolution, zoom=max(1.04, zoom * 1.03), pan_x=pan_x * 0.35, pan_y=pan_y * 0.35)
    bg = bg.filter(ImageFilter.GaussianBlur(radius=16))
    bg = ImageEnhance.Brightness(bg).enhance(0.82)
    canvas = bg.convert("RGBA")

    if comparison_image is not None:
        left_box = (int(w * 0.06), int(h * 0.18), int(w * 0.46), int(h * 0.84))
        right_box = (int(w * 0.54), int(h * 0.18), int(w * 0.94), int(h * 0.84))
        for box, src in ((left_box, comparison_image), (right_box, main_image)):
            pane = _fit_contain(src, (box[2] - box[0], box[3] - box[1]))
            frame = Image.new("RGBA", (pane.width + 18, pane.height + 18), (12, 14, 20, 242))
            frame.paste(pane.convert("RGBA"), (9, 9))
            shadow = Image.new("RGBA", frame.size, (0, 0, 0, 0))
            shadow_draw = ImageDraw.Draw(shadow)
            shadow_draw.rounded_rectangle((0, 0, frame.width - 1, frame.height - 1), radius=18, fill=(0, 0, 0, 70))
            shadow = shadow.filter(ImageFilter.GaussianBlur(radius=10))
            x = box[0] + ((box[2] - box[0] - frame.width) // 2)
            y = box[1] + ((box[3] - box[1] - frame.height) // 2)
            canvas.alpha_composite(shadow, (x + 8, y + 10))
            canvas.alpha_composite(frame, (x, y))
    else:
        fg = _fit_contain(main_image, (int(w * 0.82), int(h * 0.78)))
        fg = _apply_zoom_pan(fg, fg.size, zoom=zoom, pan_x=pan_x, pan_y=pan_y)
        if polaroid:
            matte = Image.new("RGBA", (fg.width + 44, fg.height + 82), (245, 243, 238, 255))
            matte.paste(fg.convert("RGBA"), (22, 22))
            fg_rgba = matte
        else:
            fg_rgba = fg.convert("RGBA")
        shadow = Image.new("RGBA", fg_rgba.size, (0, 0, 0, 0))
        shadow_draw = ImageDraw.Draw(shadow)
        shadow_draw.rounded_rectangle((0, 0, fg_rgba.width - 1, fg_rgba.height - 1), radius=18, fill=(0, 0, 0, 78))
        shadow = shadow.filter(ImageFilter.GaussianBlur(radius=14))
        x = (w - fg_rgba.width) // 2
        y = (h - fg_rgba.height) // 2
        canvas.alpha_composite(shadow, (x + 10, y + 14))
        canvas.alpha_composite(fg_rgba, (x, y))

    out = canvas.convert("RGB")
    out = _add_glow(out, glow_strength)
    out = _add_vignette(out, vignette_strength)
    return out


def _draw_overlay(
    img: Image.Image,
    *,
    progress: float,
    phase_title: str,
    top_text: str,
    bottom_text: str,
) -> Image.Image:
    out = img.convert("RGBA")
    draw = ImageDraw.Draw(out)
    w, h = out.size
    draw.rectangle((0, 0, w, 72), fill=(4, 6, 12, 150))
    draw.rectangle((0, h - 96, w, h), fill=(4, 6, 12, 150))
    _text(draw, (24, 18), top_text, (235, 240, 255))
    _text(draw, (24, h - 66), bottom_text, (220, 224, 236))
    _text(draw, (w - 220, 18), phase_title.upper(), (255, 195, 120))

    bar_margin = 24
    bar_h = 16
    bar_y = h - 30
    draw.rounded_rectangle((bar_margin, bar_y, w - bar_margin, bar_y + bar_h), radius=8, fill=(40, 44, 60, 200))
    fill_w = int(round((w - (bar_margin * 2)) * _clamp(progress, 0.0, 1.0)))
    if fill_w > 0:
        draw.rounded_rectangle((bar_margin, bar_y, bar_margin + fill_w, bar_y + bar_h), radius=8, fill=(255, 166, 77, 235))
    pct_text = f"{int(round(_clamp(progress, 0.0, 1.0) * 100.0))}%"
    _text(draw, (w - 72, bar_y - 20), pct_text, (255, 230, 200))
    return out.convert("RGB")


def _image_difference_score(a: Image.Image, b: Image.Image) -> float:
    a_small = a.resize((160, 160), Image.Resampling.BILINEAR).convert("L")
    b_small = b.resize((160, 160), Image.Resampling.BILINEAR).convert("L")
    diff = ImageChops.difference(a_small, b_small)
    arr = np.asarray(diff, dtype=np.float32)
    return float(arr.mean() + arr.std())


def _sharpness_score(img: Image.Image) -> float:
    arr = np.asarray(img.resize((256, 256), Image.Resampling.BILINEAR).convert("L"))
    return float(cv2.Laplacian(arr, cv2.CV_64F).var())


def _choose_hook_indices(frames: list[Image.Image], count: int) -> list[int]:
    if len(frames) <= count:
        return list(range(len(frames)))
    candidates: list[tuple[float, int]] = []
    for idx, frame in enumerate(frames):
        prev_frame = frames[max(0, idx - 1)]
        next_frame = frames[min(len(frames) - 1, idx + 1)]
        score = _image_difference_score(frame, prev_frame) + _image_difference_score(frame, next_frame)
        candidates.append((score, idx))
    selected = sorted(candidates, reverse=True)[:count]
    ordered = sorted(idx for _, idx in selected)
    if len(ordered) < count:
        stride = max(1, len(frames) // count)
        ordered = sorted(set(ordered + list(range(0, len(frames), stride))))[:count]
    return ordered


def _choose_best_frame_index(frames: list[Image.Image]) -> int:
    scores = [(_sharpness_score(frame), idx) for idx, frame in enumerate(frames)]
    return max(scores)[1]


def _pil_to_bgr(img: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.asarray(img.convert("RGB")), cv2.COLOR_RGB2BGR)


def _phase_counts(frame_count: int, fps: int) -> tuple[int, int, int, int]:
    hook = max(int(2.6 * fps), int(3.0 * fps))
    init = int(5.0 * fps)
    progression_seconds = _clamp(frame_count * 0.45, 12.0, 32.0)
    progression = int(round(progression_seconds * fps))
    final = int(round((_clamp(5.5 + (frame_count / 80.0), 5.0, 8.5)) * fps))
    return hook, init, progression, final


def _emit_frame(
    writer: cv2.VideoWriter,
    frame_image: Image.Image,
    frame_index: int,
    total_frames: int,
    progress_callback: ProgressCallback | None,
):
    writer.write(_pil_to_bgr(frame_image))
    if progress_callback is not None and (frame_index == 1 or frame_index == total_frames or frame_index % 12 == 0):
        progress_callback(frame_index, total_frames)


def _ffmpeg_command(ffmpeg_executable: str, raw_path: Path, out_path: Path, audio_path: Path | None) -> list[str]:
    cmd = [
        ffmpeg_executable,
        "-y",
        "-i",
        str(raw_path),
    ]
    if audio_path is not None:
        cmd.extend(["-i", str(audio_path), "-map", "0:v:0", "-map", "1:a:0", "-c:a", "aac", "-shortest"])
    else:
        cmd.append("-an")
    cmd.extend(["-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out_path)])
    return cmd


def render_reconstruction_video(
    frame_paths: Iterable[Path],
    output_path: Path,
    *,
    original_image_path: Path | None = None,
    audio_path: Path | None = None,
    resolution: tuple[int, int] = (1920, 1080),
    fps: int = 30,
    seed: int = 1337,
    progress_callback: ProgressCallback | None = None,
) -> ReconstructionVideoResult:
    ordered_paths = sorted((Path(p) for p in frame_paths), key=_natural_key)
    if len(ordered_paths) < 2:
        raise ValueError("Reconstruction Video Mode needs at least 2 ordered frame images.")

    width, height = resolution
    if (width, height) not in {(1920, 1080), (1280, 720)}:
        raise ValueError("Resolution must be 1920x1080 or 1280x720.")
    if fps not in {30, 60}:
        raise ValueError("FPS must be 30 or 60.")

    rng = random.Random(seed)
    np.random.seed(seed)

    base_frames = [_load_rgb_image(path) for path in ordered_paths]
    original_image = _load_rgb_image(original_image_path) if original_image_path else None

    hook_count, init_count, progression_count, final_count = _phase_counts(len(base_frames), fps)
    total_frames = hook_count + init_count + progression_count + final_count
    frame_cursor = 0

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    raw_output_path = output_path.with_name(f"{output_path.stem}_raw.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(raw_output_path), fourcc, float(fps), (width, height))
    if not writer.isOpened():
        raise RuntimeError("Could not open video writer for MP4 export.")

    try:
        hook_indices = _choose_hook_indices(base_frames, count=max(5, min(10, len(base_frames))))
        if len(hook_indices) < 5:
            hook_indices = list(range(min(len(base_frames), 5)))

        hook_durations = [max(int(0.12 * fps), min(int(0.28 * fps), hook_count // max(1, len(hook_indices))))] * len(hook_indices)
        remaining = hook_count - sum(hook_durations)
        idx = 0
        while remaining > 0:
            hook_durations[idx % len(hook_durations)] += 1
            remaining -= 1
            idx += 1

        hook_sequence: list[int] = []
        for hook_idx, duration in zip(hook_indices, hook_durations):
            hook_sequence.extend([hook_idx] * duration)
        hook_sequence = hook_sequence[:hook_count]

        for local_idx, source_idx in enumerate(hook_sequence):
            t = local_idx / max(1, hook_count - 1)
            zoom = _lerp(1.02, 1.14, t)
            pan_x = math.sin(t * math.pi * 2.3) * 0.12
            pan_y = math.cos(t * math.pi * 1.7) * 0.08
            frame = _compose_depth_scene(base_frames[source_idx], resolution, zoom=zoom, pan_x=pan_x, pan_y=pan_y)
            if local_idx == 0 or hook_sequence[local_idx - 1] != source_idx:
                frame = _apply_glitch(frame, rng, intensity=0.95)
            if local_idx % max(2, fps // 15) == 0:
                frame = _apply_motion_blur(frame, 0.25)
            frame = _draw_overlay(
                frame,
                progress=frame_cursor / max(1, total_frames - 1),
                phase_title="Hook",
                top_text="RECONSTRUCTING...",
                bottom_text="This is what AI remembers",
            )
            frame = _apply_film_grain(frame, 0.16, rng)
            frame_cursor += 1
            _emit_frame(writer, frame, frame_cursor, total_frames, progress_callback)

        init_source = original_image if original_image is not None else base_frames[0]
        init_start = _compose_depth_scene(init_source, resolution, zoom=1.0, pan_x=0.0, pan_y=0.0, polaroid=True)
        init_end = _compose_depth_scene(base_frames[0], resolution, zoom=1.08, pan_x=0.02, pan_y=-0.02)
        for i in range(init_count):
            t = _ease_in_out(i / max(1, init_count - 1))
            if t < 0.72:
                local_t = t / 0.72
                frame = _compose_depth_scene(init_source, resolution, zoom=_lerp(1.0, 1.1, local_t), pan_x=0.0, pan_y=-0.03, polaroid=True)
            else:
                blend_t = (t - 0.72) / 0.28
                frame = _alpha_blend(init_start, init_end, blend_t)
            frame = _draw_overlay(
                frame,
                progress=frame_cursor / max(1, total_frames - 1),
                phase_title="Initialization",
                top_text="RECONSTRUCTING...",
                bottom_text="This is what AI remembers",
            )
            frame = _apply_film_grain(frame, 0.1, rng)
            frame_cursor += 1
            _emit_frame(writer, frame, frame_cursor, total_frames, progress_callback)

        best_idx = _choose_best_frame_index(base_frames)
        pair_count = max(1, len(base_frames) - 1)
        failure_interval = max(fps * 4, progression_count // 4)
        compare_interval = max(fps * 6, progression_count // 3)
        for progress_idx in range(progression_count):
            timeline_t = progress_idx / max(1, progression_count - 1)
            position = timeline_t * pair_count
            pair_idx = min(pair_count - 1, int(math.floor(position)))
            local_t = _ease_in_out(position - pair_idx)
            current = base_frames[pair_idx]
            nxt = base_frames[min(pair_idx + 1, len(base_frames) - 1)]
            segment_seed = seed + (pair_idx * 97)
            segment_rng = random.Random(segment_seed)
            pan_start_x = segment_rng.uniform(-0.16, 0.16)
            pan_start_y = segment_rng.uniform(-0.12, 0.12)
            pan_end_x = segment_rng.uniform(-0.16, 0.16)
            pan_end_y = segment_rng.uniform(-0.12, 0.12)
            zoom_start = segment_rng.uniform(1.01, 1.06)
            zoom_end = min(1.16, zoom_start + segment_rng.uniform(0.03, 0.08))
            blended = _alpha_blend(current, nxt, local_t)
            comparison = None
            if original_image is not None and frame_cursor > 0 and frame_cursor % compare_interval < 8:
                comparison = original_image
            frame = _compose_depth_scene(
                blended,
                resolution,
                zoom=_lerp(zoom_start, zoom_end, local_t),
                pan_x=_lerp(pan_start_x, pan_end_x, local_t),
                pan_y=_lerp(pan_start_y, pan_end_y, local_t),
                comparison_image=comparison,
            )
            if frame_cursor > 0 and frame_cursor % failure_interval < 3:
                frame = _apply_failure_frame(frame, segment_rng)
            elif progress_idx == 0 or progress_idx == progression_count - 1:
                frame = _apply_motion_blur(frame, 0.18)
            frame = _draw_overlay(
                frame,
                progress=frame_cursor / max(1, total_frames - 1),
                phase_title="Progression",
                top_text="RECONSTRUCTING...",
                bottom_text="This is what AI remembers",
            )
            frame = _apply_film_grain(frame, 0.08, segment_rng)
            frame_cursor += 1
            _emit_frame(writer, frame, frame_cursor, total_frames, progress_callback)

        final_frame = base_frames[best_idx]
        comparison_target = None
        if original_image is not None:
            comparison_target = _compose_depth_scene(
                final_frame,
                resolution,
                zoom=1.08,
                pan_x=0.0,
                pan_y=0.0,
                comparison_image=original_image,
                vignette_strength=0.18,
                glow_strength=0.12,
            )
        for i in range(final_count):
            t = _ease_in_out(i / max(1, final_count - 1))
            frame = _compose_depth_scene(
                final_frame,
                resolution,
                zoom=_lerp(1.05, 1.17, t),
                pan_x=_lerp(-0.02, 0.03, t),
                pan_y=_lerp(0.01, -0.03, t),
                vignette_strength=_lerp(0.12, 0.28, t),
                glow_strength=_lerp(0.04, 0.18, t),
            )
            if comparison_target is not None and t > 0.58:
                blend_t = (t - 0.58) / 0.42
                frame = _alpha_blend(frame, comparison_target, blend_t * 0.72)
            frame = _draw_overlay(
                frame,
                progress=frame_cursor / max(1, total_frames - 1),
                phase_title="Final Reveal",
                top_text="RECONSTRUCTING...",
                bottom_text="This is what AI remembers",
            )
            frame = _apply_film_grain(frame, 0.06, rng)
            frame_cursor += 1
            _emit_frame(writer, frame, frame_cursor, total_frames, progress_callback)
    finally:
        writer.release()

    used_h264 = False
    audio_included = False
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path is not None:
        cmd = _ffmpeg_command(ffmpeg_path, raw_output_path, output_path, audio_path)
        completed = subprocess.run(cmd, capture_output=True, text=True)
        if completed.returncode == 0 and output_path.exists():
            used_h264 = True
            audio_included = audio_path is not None
            try:
                raw_output_path.unlink(missing_ok=True)
            except Exception:
                pass
        else:
            if output_path.exists():
                try:
                    output_path.unlink()
                except Exception:
                    pass
            shutil.move(str(raw_output_path), str(output_path))
    else:
        shutil.move(str(raw_output_path), str(output_path))

    return ReconstructionVideoResult(
        output_path=output_path,
        total_frames=total_frames,
        fps=fps,
        resolution=resolution,
        duration_seconds=total_frames / float(fps),
        used_h264=used_h264,
        audio_included=audio_included,
    )

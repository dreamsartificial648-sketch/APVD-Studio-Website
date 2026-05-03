from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import re

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps, ImageStat
import torch

from model import VAE
from utils import select_model_path_for_prompt, tensor_to_pil


RELATION_ALIASES = {
    "next to": "next_to",
    "beside": "next_to",
    "above": "above",
    "below": "below",
    "behind": "behind",
    "in front of": "frontground",
    "in the background": "background_layer",
    "in the midground": "midground_layer",
}
RELATION_ORDER = sorted(RELATION_ALIASES, key=len, reverse=True)
DEPTH_RELATION_PHRASES = ("in front of", "in the background", "in the midground")
FILLER_WORDS = {
    "a",
    "an",
    "the",
    "with",
    "and",
    "of",
    "in",
    "on",
    "at",
    "to",
    "for",
    "from",
    "by",
    "very",
    "really",
    "small",
    "big",
    "large",
    "giant",
    "tiny",
}


@dataclass
class SceneObject:
    phrase: str
    noun: str
    model_path: Path
    generated_image: Image.Image | None = None
    processed_image: Image.Image | None = None


@dataclass
class ParsedScene:
    prompt: str
    relation: str
    objects: list[SceneObject]


@dataclass
class LayeredScene:
    """Scene prompt resolved into background, midground, and frontground objects."""

    background: SceneObject | None
    midground: SceneObject | None
    frontground: SceneObject | None
    prompt: str


SceneParseResult = ParsedScene | LayeredScene


def parse_scene_prompt(prompt: str, models_folder: Path) -> ParsedScene:
    """Parse a standard two-object spatial prompt into a flat scene structure."""
    cleaned_prompt = " ".join(prompt.strip().split())
    if not cleaned_prompt:
        raise ValueError("Prompt cannot be empty.")

    prompt_lower = cleaned_prompt.lower()
    relation_phrase = None
    relation_index = -1
    for phrase in RELATION_ORDER:
        idx = prompt_lower.find(phrase)
        if idx >= 0:
            relation_phrase = phrase
            relation_index = idx
            break

    if relation_phrase is None:
        phrase = _normalize_object_phrase(cleaned_prompt)
        model_path, _ = select_model_path_for_prompt(models_folder, phrase)
        return ParsedScene(
            prompt=cleaned_prompt,
            relation="single",
            objects=[SceneObject(phrase=phrase, noun=_head_noun(phrase), model_path=model_path)],
        )

    left = cleaned_prompt[:relation_index].strip(" ,.")
    right = cleaned_prompt[relation_index + len(relation_phrase) :].strip(" ,.")
    if not left or not right:
        raise ValueError(f"Prompt must contain an object on both sides of '{relation_phrase}'.")

    object_phrases = [_normalize_object_phrase(left), _normalize_object_phrase(right)]
    objects: list[SceneObject] = []
    for phrase in object_phrases:
        model_path, _ = select_model_path_for_prompt(models_folder, phrase)
        objects.append(SceneObject(phrase=phrase, noun=_head_noun(phrase), model_path=model_path))

    return ParsedScene(
        prompt=cleaned_prompt,
        relation=RELATION_ALIASES[relation_phrase],
        objects=objects,
    )


def parse_layered_scene_prompt(prompt: str, models_folder: Path) -> SceneParseResult:
    """Parse depth-layer prompts, falling back to the legacy flat scene parser."""
    cleaned_prompt = " ".join(prompt.strip().split())
    if not cleaned_prompt:
        raise ValueError("Prompt cannot be empty.")

    prompt_lower = cleaned_prompt.lower()
    if not any(phrase in prompt_lower for phrase in DEPTH_RELATION_PHRASES):
        return parse_scene_prompt(cleaned_prompt, models_folder=models_folder)

    segments = _split_prompt_by_relations(cleaned_prompt)
    if not segments:
        return parse_scene_prompt(cleaned_prompt, models_folder=models_folder)

    object_phrases = [_normalize_object_phrase(phrase) for phrase, _ in segments]
    relation_names = [RELATION_ALIASES[relation] for _, relation in segments[:-1] if relation]

    layer_phrases = _resolve_layer_phrases(object_phrases, relation_names)
    scene_objects = {
        layer: _build_scene_object(phrase, models_folder)
        for layer, phrase in layer_phrases.items()
        if phrase is not None
    }
    if not scene_objects:
        return parse_scene_prompt(cleaned_prompt, models_folder=models_folder)

    return LayeredScene(
        background=scene_objects.get("background"),
        midground=scene_objects.get("midground"),
        frontground=scene_objects.get("frontground"),
        prompt=cleaned_prompt,
    )


def generate_scene_from_prompt(
    prompt: str,
    models_folder: Path,
    device: torch.device,
    output_dir: Path,
    target_size: tuple[int, int] = (256, 256),
    soften_background: bool = True,
    feather_radius: int = 10,
    add_noise: bool = True,
) -> tuple[Image.Image, Path, SceneParseResult]:
    """Generate one image per scene object and compose either a flat or layered scene."""
    scene = parse_layered_scene_prompt(prompt, models_folder=models_folder)

    for scene_object in _iter_scene_objects(scene):
        generated = generate_image_from_checkpoint(
            scene_object.model_path,
            device=device,
            target_size=target_size,
        )
        scene_object.generated_image = generated
        scene_object.processed_image = prepare_object_image(
            generated,
            target_size=target_size,
            soften_background=soften_background,
            feather_radius=feather_radius,
        )

    if isinstance(scene, LayeredScene):
        composed = compose_layered_scene(scene)
    else:
        composed = compose_scene(scene)
        if add_noise:
            composed = apply_global_noise(composed, amount=8.0)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"scene_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    composed.save(output_path)
    return composed, output_path, scene


def generate_image_from_checkpoint(
    checkpoint_path: Path,
    device: torch.device,
    target_size: tuple[int, int] = (256, 256),
    latent_scale: float = 1.0,
    diffusion_steps: int = 8,
    diffusion_strength: float = 0.85,
) -> Image.Image:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    output_size = tuple(checkpoint.get("output_size", target_size))
    model = VAE(
        latent_dim=checkpoint.get("latent_dim", 256),
        output_size=output_size,
    ).to(device)
    load_result = model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model.eval()

    with torch.no_grad():
        z = torch.randn(1, model.latent_dim, device=device) * latent_scale
        if _has_latent_denoiser(load_result):
            z = refine_latent_with_denoiser(
                model,
                z,
                steps=diffusion_steps,
                strength=diffusion_strength,
            )
        recon = model.decode(z)

    pil_img = tensor_to_pil(recon)
    if pil_img.size != target_size:
        pil_img = pil_img.resize(target_size, Image.Resampling.LANCZOS)
    return pil_img


def refine_latent_with_denoiser(
    model: VAE,
    latent: torch.Tensor,
    steps: int = 8,
    strength: float = 0.85,
) -> torch.Tensor:
    current = latent.clone()
    for step_idx in range(max(1, steps)):
        t_value = 1.0 if steps == 1 else 1.0 - (step_idx / (steps - 1))
        timestep = torch.full((current.size(0), 1), t_value, device=current.device)
        predicted_noise = model.predict_latent_noise(current, timestep)
        current = current - (predicted_noise * strength * (0.2 + 0.8 * t_value))
        if step_idx < steps - 1:
            current = current + (torch.randn_like(current) * (0.025 * t_value))
    return current


def prepare_object_image(
    image: Image.Image,
    target_size: tuple[int, int] = (256, 256),
    soften_background: bool = True,
    feather_radius: int = 10,
) -> Image.Image:
    rgba = image.convert("RGBA")
    if rgba.size != target_size:
        rgba = rgba.resize(target_size, Image.Resampling.LANCZOS)

    if not soften_background:
        return rgba

    alpha = estimate_foreground_alpha(rgba, feather_radius=feather_radius)
    rgba.putalpha(alpha)
    return rgba


def estimate_foreground_alpha(image: Image.Image, feather_radius: int = 10) -> Image.Image:
    rgb = np.asarray(image.convert("RGB"), dtype=np.float32)
    h, w, _ = rgb.shape

    corner_size = max(8, min(h, w) // 8)
    corner_samples = np.concatenate(
        [
            rgb[:corner_size, :corner_size].reshape(-1, 3),
            rgb[:corner_size, -corner_size:].reshape(-1, 3),
            rgb[-corner_size:, :corner_size].reshape(-1, 3),
            rgb[-corner_size:, -corner_size:].reshape(-1, 3),
        ],
        axis=0,
    )
    background_color = corner_samples.mean(axis=0, keepdims=True)
    distance = np.linalg.norm(rgb - background_color, axis=2)

    threshold = max(18.0, float(np.percentile(distance, 65)))
    normalized = np.clip((distance - threshold * 0.45) / max(threshold, 1.0), 0.0, 1.0)
    alpha = (normalized * 255.0).astype(np.uint8)
    alpha_img = Image.fromarray(alpha, mode="L")
    if feather_radius > 0:
        alpha_img = alpha_img.filter(ImageFilter.GaussianBlur(radius=feather_radius))
    alpha_img = ImageOps.autocontrast(alpha_img)
    return alpha_img


def compose_scene(scene: ParsedScene) -> Image.Image:
    """Compose a flat spatial scene using the legacy relation layout rules."""
    processed_images = [obj.processed_image for obj in scene.objects if obj.processed_image is not None]
    if not processed_images:
        raise ValueError("No generated images available for composition.")

    processed_images = match_image_tone(processed_images)

    if scene.relation == "single":
        return flatten_rgba(processed_images[0])

    base_w, base_h = processed_images[0].size
    gap = max(18, base_w // 12)
    background = (18, 20, 28, 255)

    if scene.relation == "next_to":
        canvas = Image.new("RGBA", (base_w * 2 + gap * 3, base_h + gap * 2), background)
        positions = [
            (gap, gap),
            (gap * 2 + base_w, gap),
        ]
    elif scene.relation == "above":
        canvas = Image.new("RGBA", (base_w + gap * 2, base_h * 2 + gap * 3), background)
        positions = [
            (gap, gap),
            (gap, gap * 2 + base_h),
        ]
    elif scene.relation == "below":
        canvas = Image.new("RGBA", (base_w + gap * 2, base_h * 2 + gap * 3), background)
        positions = [
            (gap, gap * 2 + base_h),
            (gap, gap),
        ]
    elif scene.relation == "behind":
        canvas = Image.new("RGBA", (base_w + gap * 2, base_h + gap * 2), background)
        positions = [
            (gap + base_w // 9, gap + base_h // 10),
            (gap - base_w // 14, gap - base_h // 14),
        ]
        processed_images[1] = processed_images[1].resize(
            (int(base_w * 0.92), int(base_h * 0.92)),
            Image.Resampling.LANCZOS,
        )
    else:
        raise ValueError(f"Unsupported scene relation: {scene.relation}")

    for image, position in zip(processed_images, positions):
        layer = feather_image_edges(image, radius=8)
        canvas.alpha_composite(layer, dest=position)

    return flatten_rgba(canvas)


def compose_layered_scene(
    layered_scene: LayeredScene,
    canvas_size: tuple[int, int] = (768, 512),
) -> Image.Image:
    """Compose a three-layer scene with scale, opacity, and depth effects."""
    canvas = Image.new("RGBA", canvas_size, (18, 20, 28, 255))
    width, height = canvas_size
    layer_specs = [
        ("background", layered_scene.background, 0.85, height // 3, 180, 3.0, False),
        ("midground", layered_scene.midground, 0.70, height // 2, 220, 0.0, False),
        ("frontground", layered_scene.frontground, 0.55, int(height * 0.82), 255, 0.0, True),
    ]

    rendered_any = False
    for _, scene_object, width_ratio, center_y, opacity, blur_radius, sharpen in layer_specs:
        if scene_object is None or scene_object.processed_image is None:
            continue

        layer = _prepare_layer_image(
            scene_object.processed_image,
            target_width=max(1, int(width * width_ratio)),
            opacity=opacity,
            blur_radius=blur_radius,
            sharpen=sharpen,
        )
        x = max(0, (width - layer.width) // 2)
        y = max(0, min(height - layer.height, center_y - layer.height // 2))
        canvas.alpha_composite(layer, dest=(x, y))
        rendered_any = True

    if not rendered_any:
        raise ValueError("No generated images available for layered composition.")

    composed = apply_global_noise(canvas, amount=8.0)
    return flatten_rgba(composed)


def match_image_tone(images: list[Image.Image]) -> list[Image.Image]:
    if len(images) < 2:
        return images

    reference = images[0].convert("RGB")
    ref_mean = np.array(ImageStat.Stat(reference).mean, dtype=np.float32)
    ref_std = np.maximum(np.array(ImageStat.Stat(reference).stddev, dtype=np.float32), 1.0)

    matched = [images[0]]
    for image in images[1:]:
        rgb = image.convert("RGB")
        arr = np.asarray(rgb, dtype=np.float32)
        src_mean = arr.reshape(-1, 3).mean(axis=0)
        src_std = np.maximum(arr.reshape(-1, 3).std(axis=0), 1.0)
        adjusted = ((arr - src_mean) / src_std) * ref_std + ref_mean
        adjusted = np.clip(adjusted, 0, 255).astype(np.uint8)
        adjusted_img = Image.fromarray(adjusted, mode="RGB")
        adjusted_img = ImageEnhance.Contrast(adjusted_img).enhance(0.97)
        adjusted_rgba = adjusted_img.convert("RGBA")
        adjusted_rgba.putalpha(image.getchannel("A"))
        matched.append(adjusted_rgba)
    return matched


def feather_image_edges(image: Image.Image, radius: int = 8) -> Image.Image:
    rgba = image.convert("RGBA")
    alpha = rgba.getchannel("A").filter(ImageFilter.GaussianBlur(radius=radius))
    feathered = rgba.copy()
    feathered.putalpha(alpha)
    return feathered


def flatten_rgba(image: Image.Image, background: tuple[int, int, int] = (18, 20, 28)) -> Image.Image:
    """Flatten an RGBA image onto a solid RGB background."""
    base = Image.new("RGB", image.size, background)
    base.paste(image.convert("RGBA"), mask=image.getchannel("A") if image.mode == "RGBA" else None)
    return base


def apply_global_noise(image: Image.Image, amount: float = 8.0) -> Image.Image:
    """Add light global Gaussian noise to an image."""
    rgb = np.asarray(image.convert("RGB"), dtype=np.float32)
    noise = np.random.normal(loc=0.0, scale=amount, size=rgb.shape).astype(np.float32)
    blended = np.clip(rgb + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(blended, mode="RGB")


def _normalize_object_phrase(value: str) -> str:
    """Reduce an object phrase to meaningful lowercase tokens."""
    tokens = re.findall(r"[a-z0-9]+", value.lower())
    filtered = [token for token in tokens if token not in FILLER_WORDS]
    if not filtered:
        raise ValueError(f"Could not detect an object in: {value!r}")
    return " ".join(filtered)


def _head_noun(value: str) -> str:
    """Return the last token from a phrase as a simple noun label."""
    tokens = re.findall(r"[a-z0-9]+", value.lower())
    if not tokens:
        return value.strip().lower()
    return tokens[-1]


def _has_latent_denoiser(load_result: object) -> bool:
    """Detect whether a checkpoint fully provided latent denoiser weights."""
    missing = list(getattr(load_result, "missing_keys", []))
    return not any(key.startswith("latent_denoiser.") for key in missing)


def _build_scene_object(phrase: str, models_folder: Path) -> SceneObject:
    """Resolve one object phrase to its best matching checkpoint."""
    model_path, _ = select_model_path_for_prompt(models_folder, phrase)
    return SceneObject(phrase=phrase, noun=_head_noun(phrase), model_path=model_path)


def _iter_scene_objects(scene: SceneParseResult) -> list[SceneObject]:
    """Return every populated scene object regardless of scene layout type."""
    if isinstance(scene, LayeredScene):
        return [obj for obj in (scene.background, scene.midground, scene.frontground) if obj is not None]
    return list(scene.objects)


def _split_prompt_by_relations(prompt: str) -> list[tuple[str, str | None]]:
    """Split a prompt into ordered object phrases and the relations between them."""
    parts: list[tuple[str, str | None]] = []
    remaining = prompt.strip()
    while remaining:
        match = _find_first_relation(remaining)
        if match is None:
            parts.append((remaining.strip(" ,."), None))
            break
        start, end, relation_phrase = match
        left = remaining[:start].strip(" ,.")
        if not left:
            raise ValueError(f"Prompt must contain an object before '{relation_phrase}'.")
        parts.append((left, relation_phrase))
        remaining = remaining[end:].strip()
    return parts


def _find_first_relation(text: str) -> tuple[int, int, str] | None:
    """Return the first relation phrase match in left-to-right order."""
    matches: list[tuple[int, int, str]] = []
    text_lower = text.lower()
    for phrase in RELATION_ORDER:
        index = text_lower.find(phrase)
        if index >= 0:
            matches.append((index, index + len(phrase), phrase))
    if not matches:
        return None
    return min(matches, key=lambda item: (item[0], -len(item[2])))


def _resolve_layer_phrases(
    object_phrases: list[str],
    relation_names: list[str],
) -> dict[str, str | None]:
    """Infer layer assignments from sequential depth relations."""
    if not object_phrases:
        return {"background": None, "midground": None, "frontground": None}

    depth_values: list[int | None] = [None] * len(object_phrases)
    depth_values[0] = 1
    changed = True
    while changed:
        changed = False
        for index, relation_name in enumerate(relation_names):
            left_depth = depth_values[index]
            right_depth = depth_values[index + 1]
            delta = _relation_depth_delta(relation_name)
            if delta is None:
                continue
            if left_depth is not None and right_depth is None:
                depth_values[index + 1] = left_depth + delta
                changed = True
            elif right_depth is not None and left_depth is None:
                depth_values[index] = right_depth - delta
                changed = True

    known_depths = [value for value in depth_values if value is not None]
    if not known_depths:
        depth_values = list(range(len(object_phrases)))
    else:
        minimum = min(known_depths)
        maximum = max(known_depths)
        span = max(maximum - minimum, 1)
        for index, value in enumerate(depth_values):
            if value is None:
                if len(object_phrases) == 1:
                    depth_values[index] = 1
                else:
                    relative = index / (len(object_phrases) - 1)
                    depth_values[index] = int(round(minimum + relative * span))

    buckets = {"background": None, "midground": None, "frontground": None}
    minimum = min(depth_values)
    maximum = max(depth_values)
    for value, phrase in zip(depth_values, object_phrases):
        if maximum == minimum:
            layer = "midground"
        elif value == minimum:
            layer = "frontground"
        elif value == maximum:
            layer = "background"
        else:
            layer = "midground"
        buckets[layer] = phrase
    return buckets


def _relation_depth_delta(relation_name: str) -> int | None:
    """Convert a depth relation into a signed relative depth offset."""
    if relation_name == "frontground":
        return 1
    if relation_name == "behind":
        return -1
    if relation_name == "background_layer":
        return -2
    if relation_name == "midground_layer":
        return -1
    return None


def _prepare_layer_image(
    image: Image.Image,
    target_width: int,
    opacity: int,
    blur_radius: float,
    sharpen: bool,
) -> Image.Image:
    """Resize and post-process one depth layer image before compositing."""
    rgba = feather_image_edges(image, radius=8).convert("RGBA")
    target_width = max(1, target_width)
    target_height = max(1, int(round(rgba.height * (target_width / max(1, rgba.width)))))
    layer = rgba.resize((target_width, target_height), Image.Resampling.LANCZOS)
    if blur_radius > 0:
        layer = layer.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    if sharpen:
        layer = layer.filter(ImageFilter.UnsharpMask(radius=1, percent=125, threshold=2))
    alpha = layer.getchannel("A").point(lambda value: int(value * (opacity / 255)))
    layer.putalpha(alpha)
    return layer

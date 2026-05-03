from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import torch

from model import VAE


def merge_checkpoints(
    checkpoint_paths: list[Path],
    output_path: Path,
    strategy: str = "mean",
    weights: list[float] | None = None,
) -> Path:
    """Merge multiple VAE checkpoints into one averaged checkpoint on CPU."""
    if len(checkpoint_paths) < 2:
        raise ValueError("Provide at least two checkpoints to merge.")
    if strategy not in {"mean", "weighted", "slerp"}:
        raise ValueError("strategy must be one of: mean, weighted, slerp")

    checkpoints = [torch.load(path, map_location="cpu") for path in checkpoint_paths]
    latent_dim = checkpoints[0].get("latent_dim", 256)
    output_size = tuple(checkpoints[0].get("output_size", (256, 256)))

    for path, checkpoint in zip(checkpoint_paths, checkpoints):
        current_latent_dim = checkpoint.get("latent_dim", 256)
        current_output_size = tuple(checkpoint.get("output_size", (256, 256)))
        if current_latent_dim != latent_dim:
            raise ValueError(
                f"latent_dim mismatch for {path}: expected {latent_dim}, found {current_latent_dim}"
            )
        if current_output_size != output_size:
            raise ValueError(
                f"output_size mismatch for {path}: expected {output_size}, found {current_output_size}"
            )
        if "model_state_dict" not in checkpoint:
            raise ValueError(f"Checkpoint missing model_state_dict: {path}")

    normalized_weights = _normalize_weights(strategy, len(checkpoint_paths), weights)
    state_dicts = [checkpoint["model_state_dict"] for checkpoint in checkpoints]

    with torch.no_grad():
        merged_state = _merge_state_dicts(
            state_dicts=state_dicts,
            checkpoint_paths=checkpoint_paths,
            strategy=strategy,
            normalized_weights=normalized_weights,
        )
        base_model = VAE(latent_dim=latent_dim, output_size=output_size)
        base_state = base_model.state_dict()
        base_state.update(merged_state)
        merged_state = base_state

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": merged_state,
            "latent_dim": latent_dim,
            "output_size": output_size,
            "merged_from": [str(path) for path in checkpoint_paths],
            "merge_strategy": strategy,
            "merge_weights": normalized_weights,
        },
        output_path,
    )
    return output_path


def _normalize_weights(
    strategy: str,
    checkpoint_count: int,
    weights: list[float] | None,
) -> list[float]:
    """Return normalized merge weights for the requested strategy."""
    if strategy == "mean":
        return [1.0 / checkpoint_count] * checkpoint_count
    if strategy == "slerp":
        if checkpoint_count != 2:
            raise ValueError("slerp requires exactly 2 checkpoints.")
        return [0.5, 0.5]

    if weights is None or len(weights) != checkpoint_count:
        raise ValueError("weighted merge requires one weight per checkpoint.")
    total = sum(float(weight) for weight in weights)
    if total <= 0:
        raise ValueError("weights must sum to a positive value.")
    return [float(weight) / total for weight in weights]


def _merge_state_dicts(
    state_dicts: list[dict[str, torch.Tensor]],
    checkpoint_paths: list[Path],
    strategy: str,
    normalized_weights: list[float],
) -> dict[str, torch.Tensor]:
    """Merge compatible keys across multiple state dicts and warn on skipped keys."""
    common_keys = set(state_dicts[0].keys())
    for state_dict in state_dicts[1:]:
        common_keys &= set(state_dict.keys())

    merged: dict[str, torch.Tensor] = {}
    for key in sorted(common_keys):
        tensors = [state_dict[key] for state_dict in state_dicts]
        reference_shape = tensors[0].shape
        reference_dtype = tensors[0].dtype
        if any(tensor.shape != reference_shape for tensor in tensors[1:]):
            warnings.warn(f"Skipping key '{key}' because tensor shapes do not match.")
            continue

        try:
            if strategy == "slerp":
                merged_tensor = _slerp_tensor_pair(tensors[0], tensors[1], t=0.5)
            else:
                merged_tensor = _weighted_merge_tensors(tensors, normalized_weights)
        except Exception as exc:
            warnings.warn(f"Skipping key '{key}' because it could not be merged: {exc}")
            continue

        merged[key] = _cast_like_reference(merged_tensor, reference_dtype)

    for index, state_dict in enumerate(state_dicts):
        missing_keys = sorted(set(state_dict.keys()) - common_keys)
        for key in missing_keys:
            warnings.warn(
                f"Skipping key '{key}' from {checkpoint_paths[index].name} because it is not present in every checkpoint."
            )
    return merged


def _weighted_merge_tensors(
    tensors: list[torch.Tensor],
    normalized_weights: list[float],
) -> torch.Tensor:
    """Average tensors element-wise using normalized weights."""
    merged = torch.zeros_like(tensors[0], dtype=torch.float32)
    for tensor, weight in zip(tensors, normalized_weights):
        merged = merged + (tensor.detach().to(dtype=torch.float32) * weight)
    return merged


def _slerp_tensor_pair(a: torch.Tensor, b: torch.Tensor, t: float) -> torch.Tensor:
    """Spherically interpolate between two tensors, with linear fallback when needed."""
    a_float = a.detach().to(dtype=torch.float32)
    b_float = b.detach().to(dtype=torch.float32)
    a_flat = a_float.reshape(-1)
    b_flat = b_float.reshape(-1)
    a_norm = torch.linalg.norm(a_flat)
    b_norm = torch.linalg.norm(b_flat)
    if a_norm.item() < 1e-12 or b_norm.item() < 1e-12:
        return ((1.0 - t) * a_float) + (t * b_float)

    dot = torch.clamp(torch.dot(a_flat / a_norm, b_flat / b_norm), -1.0, 1.0)
    theta = torch.acos(dot)
    sin_theta = torch.sin(theta)
    if sin_theta.abs().item() < 1e-6:
        return ((1.0 - t) * a_float) + (t * b_float)

    left = torch.sin((1.0 - t) * theta) / sin_theta
    right = torch.sin(t * theta) / sin_theta
    return (a_float * left) + (b_float * right)


def _cast_like_reference(tensor: torch.Tensor, reference_dtype: torch.dtype) -> torch.Tensor:
    """Cast merged tensors back to the original dtype when possible."""
    if reference_dtype.is_floating_point:
        return tensor.to(dtype=reference_dtype)
    if reference_dtype == torch.bool:
        return tensor >= 0.5
    return tensor.round().to(dtype=reference_dtype)


def _parse_cli_args() -> argparse.Namespace:
    """Parse CLI arguments for checkpoint merging."""
    parser = argparse.ArgumentParser(description="Merge multiple VAE checkpoints into one model.")
    parser.add_argument("--inputs", nargs="+", required=True, help="Input .pt checkpoint paths.")
    parser.add_argument("--output", required=True, help="Output checkpoint path.")
    parser.add_argument(
        "--strategy",
        default="mean",
        choices=("mean", "weighted", "slerp"),
        help="Checkpoint merge strategy.",
    )
    parser.add_argument(
        "--weights",
        nargs="*",
        type=float,
        help="Optional weights for --strategy weighted.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_cli_args()
    output = merge_checkpoints(
        checkpoint_paths=[Path(value) for value in args.inputs],
        output_path=Path(args.output),
        strategy=args.strategy,
        weights=args.weights,
    )
    print(output)

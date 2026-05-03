"""
APVD v2 - VAE model with a lightweight latent denoiser.
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class LatentDenoiser(nn.Module):
    """Small MLP that predicts latent noise from a noisy latent and a timestep."""

    def __init__(self, latent_dim: int, hidden_dim: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim + 3, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, z: Tensor, t: Tensor) -> Tensor:
        if t.ndim == 1:
            t = t.unsqueeze(1)
        t = t.clamp(0.0, 1.0)
        t_features = torch.cat(
            (
                t,
                torch.sin(t * math.pi),
                torch.cos(t * math.pi),
            ),
            dim=1,
        )
        return self.net(torch.cat((z, t_features), dim=1))


class VAE(nn.Module):
    """Convolutional VAE with configurable output resolution and latent denoiser."""

    def __init__(
        self,
        latent_dim: int = 256,
        in_channels: int = 3,
        output_size: tuple[int, int] = (256, 256),
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.in_channels = in_channels
        self.output_size = output_size

        # Downsample while preserving flexibility for different input resolutions.
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
        )

        # Normalize encoder output to a fixed latent projection shape.
        self.enc_dim = 4 * 4 * 512

        self.fc_mu = nn.Linear(self.enc_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.enc_dim, latent_dim)
        self.fc_decode = nn.Linear(latent_dim, self.enc_dim)
        self.latent_denoiser = LatentDenoiser(latent_dim)

        # Decoder base output is 128x128; final interpolate reaches requested output size.
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(32, in_channels, kernel_size=4, stride=2, padding=1),
        )

    def encode(self, x: Tensor) -> tuple[Tensor, Tensor]:
        h = self.encoder(x)
        h = F.adaptive_avg_pool2d(h, output_size=(4, 4))
        h = h.reshape(h.size(0), -1)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std, device=mu.device)
        return mu + eps * std

    def decode(self, z: Tensor) -> Tensor:
        h = self.fc_decode(z)
        h = h.reshape(h.size(0), 512, 4, 4)
        x = self.decoder(h)
        x = F.interpolate(x, size=self.output_size, mode="bilinear", align_corners=False)
        return torch.sigmoid(x)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar

    def predict_latent_noise(self, z: Tensor, t: Tensor) -> Tensor:
        return self.latent_denoiser(z, t)


def vae_loss(recon: Tensor, x: Tensor, mu: Tensor, logvar: Tensor) -> Tensor:
    """VAE loss = reconstruction (BCE) + KL divergence."""
    bce = F.binary_cross_entropy(recon, x, reduction="sum")
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return bce + kl


def latent_denoiser_loss(
    model: VAE,
    clean_latents: Tensor,
    max_noise_scale: float = 2.0,
) -> Tensor:
    """Train the latent denoiser to predict injected Gaussian noise."""
    batch = clean_latents.size(0)
    t = torch.rand(batch, 1, device=clean_latents.device)
    noise = torch.randn_like(clean_latents)
    noise_scale = 1.0 + (max_noise_scale - 1.0) * t
    noisy_latents = clean_latents + (noise * noise_scale)
    predicted_noise = model.predict_latent_noise(noisy_latents, t)
    return F.mse_loss(predicted_noise, noise)


def get_device() -> torch.device:
    """Return CUDA, MPS, or CPU device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

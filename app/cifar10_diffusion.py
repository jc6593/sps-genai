"""A small time-conditioned U-Net and diffusion process for CIFAR-10."""

import math

import torch
from torch import nn
from torch.nn import functional as F


TIME_DIM = 32


class DenoisingBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm = nn.GroupNorm(4, out_channels)
        self.time_projection = nn.Linear(TIME_DIM, out_channels)

    def forward(self, images: torch.Tensor, time_embedding: torch.Tensor) -> torch.Tensor:
        time_features = self.time_projection(time_embedding)[:, :, None, None]
        return F.silu(self.norm(self.conv(images)) + time_features)


class DiffusionUNet(nn.Module):
    def __init__(self):
        super().__init__()
        frequencies = 10000 ** (-torch.arange(TIME_DIM // 2) / (TIME_DIM // 2))
        self.register_buffer("frequencies", frequencies)
        self.time_mlp = nn.Sequential(
            nn.Linear(TIME_DIM, TIME_DIM), nn.SiLU(), nn.Linear(TIME_DIM, TIME_DIM)
        )
        self.enc32 = DenoisingBlock(3, 16)
        self.down16 = nn.Conv2d(16, 32, 4, stride=2, padding=1)
        self.enc16 = DenoisingBlock(32, 32)
        self.down8 = nn.Conv2d(32, 64, 4, stride=2, padding=1)
        self.middle = DenoisingBlock(64, 64)
        self.up16 = nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1)
        self.dec16 = DenoisingBlock(64, 32)
        self.up32 = nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1)
        self.dec32 = DenoisingBlock(32, 16)
        self.output = nn.Conv2d(16, 3, 3, padding=1)

    def forward(self, images: torch.Tensor, times: torch.Tensor) -> torch.Tensor:
        # Scale normalized time to the range used by a 1,000-step embedding.
        angles = 1000 * times[:, None] * self.frequencies[None, :]
        time_embedding = self.time_mlp(torch.cat([angles.cos(), angles.sin()], dim=1))

        skip32 = self.enc32(images, time_embedding)
        skip16 = self.enc16(self.down16(skip32), time_embedding)
        features = self.down8(skip16)
        features = features + self.middle(features, time_embedding)
        features = torch.cat([self.up16(features), skip16], dim=1)
        features = self.dec16(features, time_embedding)
        features = torch.cat([self.up32(features), skip32], dim=1)
        return self.output(self.dec32(features, time_embedding))


def diffusion_rates(times: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return signal and noise coefficients for the offset cosine schedule."""
    start_angle = math.acos(0.95)
    end_angle = math.acos(0.02)
    angles = (start_angle + times * (end_angle - start_angle)).reshape(-1, 1, 1, 1)
    return angles.cos(), angles.sin()


def diffusion_loss(model: DiffusionUNet, images: torch.Tensor) -> torch.Tensor:
    times = torch.rand(len(images), device=images.device)
    noise = torch.randn_like(images)
    signal_rate, noise_rate = diffusion_rates(times)
    noisy_images = signal_rate * images + noise_rate * noise
    return F.l1_loss(model(noisy_images, times), noise)


@torch.no_grad()
def sample_diffusion(
    model: DiffusionUNet, images: torch.Tensor, steps: int
) -> torch.Tensor:
    """Turn initial Gaussian noise into images by running the schedule backward."""
    times = torch.linspace(1, 0, steps + 1, device=images.device)
    signal_rates, noise_rates = diffusion_rates(times)
    for step in range(steps):
        predicted_noise = model(images, times[step].expand(len(images)))
        clean_images = (
            (images - noise_rates[step] * predicted_noise) / signal_rates[step]
        ).clamp(-1, 1)
        images = signal_rates[step + 1] * clean_images + noise_rates[step + 1] * predicted_noise
    return clean_images

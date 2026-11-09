"""An energy network and input-gradient sampler for CIFAR-10."""

import torch
from torch import nn


LANGEVIN_STEP_SIZE = 10.0
LANGEVIN_NOISE_STD = 0.005


class EnergyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(3, 16, 5, stride=2, padding=2),
            nn.SiLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(32, 48, 3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(48, 64, 3, stride=2, padding=1),
            nn.SiLU(),
            nn.Flatten(),
            nn.Linear(64 * 2 * 2, 64),
            nn.SiLU(),
            nn.Linear(64, 1),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.layers(images).squeeze(1)


def sample_langevin(
    model: EnergyModel,
    images: torch.Tensor,
    steps: int,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Lower the energy by differentiating with respect to pixels at each step."""
    images = images.detach()
    for _ in range(steps):
        with torch.no_grad():
            noise = torch.randn(
                images.shape, device=images.device, generator=generator
            ) * LANGEVIN_NOISE_STD
            images = (images + noise).clamp(-1, 1)
        with torch.enable_grad():
            images.requires_grad_(True)
            pixel_gradient, = torch.autograd.grad(model(images).sum(), images)
        with torch.no_grad():
            images = (images - LANGEVIN_STEP_SIZE * pixel_gradient.clamp(-0.03, 0.03))
            images = images.clamp(-1, 1)
    return images.detach()

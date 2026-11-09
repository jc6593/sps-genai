"""Load trained CIFAR-10 generators and encode their samples as PNG files."""

from functools import cache
from io import BytesIO
from pathlib import Path
from typing import Literal

import torch
from PIL import Image

from app.cifar10_diffusion import DiffusionUNet, sample_diffusion
from app.cifar10_ebm import EnergyModel, sample_langevin


GeneratorName = Literal["diffusion", "ebm"]
IMAGE_SHAPE = (3, 32, 32)
ARTIFACTS = Path(__file__).resolve().parents[1] / "artifacts"


class Cifar10ImageGenerator:
    def __init__(self, name: GeneratorName, checkpoint_path: Path):
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if checkpoint.get("model") != name or tuple(checkpoint.get("image_shape", ())) != IMAGE_SHAPE:
            raise ValueError(f"Checkpoint metadata does not match the CIFAR-10 {name} generator.")
        self.name = name
        self.model = DiffusionUNet() if name == "diffusion" else EnergyModel()
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.requires_grad_(False).eval()

    def generate(self, seed: int | None = None) -> bytes:
        generator = torch.Generator(device="cpu")
        if seed is None:
            generator.seed()
        else:
            generator.manual_seed(seed)

        if self.name == "diffusion":
            initial_images = torch.randn(1, *IMAGE_SHAPE, generator=generator)
            images = sample_diffusion(self.model, initial_images, steps=32)
        else:
            initial_images = torch.rand(1, *IMAGE_SHAPE, generator=generator) * 2 - 1
            images = sample_langevin(self.model, initial_images, steps=128, generator=generator)

        pixels = ((images[0].permute(1, 2, 0) + 1) * 127.5)
        pixels = pixels.clamp(0, 255).round().to(torch.uint8).numpy()
        output = BytesIO()
        Image.fromarray(pixels).save(output, format="PNG")
        return output.getvalue()


@cache
def load_cifar10_generator(name: GeneratorName) -> Cifar10ImageGenerator:
    return Cifar10ImageGenerator(name, ARTIFACTS / f"cifar10_{name}.pt")

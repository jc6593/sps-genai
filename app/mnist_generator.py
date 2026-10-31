"""Generate MNIST-style images with the trained GAN."""

from io import BytesIO
from pathlib import Path

import torch
from PIL import Image

from app.mnist_gan import LATENT_DIM, Generator


DEFAULT_CHECKPOINT_PATH = (
    Path(__file__).resolve().parents[1] / "artifacts" / "mnist_generator.pt"
)


class MnistGenerator:
    def __init__(self, checkpoint_path: Path = DEFAULT_CHECKPOINT_PATH):
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if checkpoint.get("latent_dim") != LATENT_DIM:
            raise ValueError("Checkpoint noise dimension does not match the generator.")
        if tuple(checkpoint.get("image_shape", ())) != (1, 28, 28):
            raise ValueError("Checkpoint image shape does not match MNIST.")
        self.model = Generator()
        self.model.load_state_dict(checkpoint["generator_state_dict"])
        self.model.eval()

    def generate(self, seed: int | None = None) -> bytes:
        random_generator = torch.Generator(device="cpu")
        if seed is None:
            random_generator.seed()
        else:
            random_generator.manual_seed(seed)
        noise = torch.randn(1, LATENT_DIM, generator=random_generator)
        with torch.inference_mode():
            image = self.model(noise)[0, 0]
        pixels = ((image + 1) * 127.5).clamp(0, 255).round().to(torch.uint8).numpy()
        output = BytesIO()
        Image.fromarray(pixels).save(output, format="PNG")
        return output.getvalue()

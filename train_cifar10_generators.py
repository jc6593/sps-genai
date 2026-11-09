"""Train either Assignment 4 image generator on CIFAR-10."""

import argparse
import tempfile
import time
from itertools import islice
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from app.cifar10_diffusion import DiffusionUNet, diffusion_loss
from app.cifar10_ebm import EnergyModel, LANGEVIN_NOISE_STD, sample_langevin
from train_cifar10 import choose_device


class ReplayBuffer:
    """Reuse recent negative samples, with a small fraction of fresh starts."""

    def __init__(self, device: torch.device):
        self.images = torch.empty(4096, 3, 32, 32, device=device)
        self.count = 0
        self.position = 0

    def draw(self, real_images: torch.Tensor) -> torch.Tensor:
        fresh = torch.rand_like(real_images) * 2 - 1
        if self.count == 0:
            return fresh
        device = real_images.device
        indices = torch.randint(self.count, (len(real_images),), device=device)
        use_fresh = torch.rand(len(real_images), 1, 1, 1, device=device) < 0.05
        return torch.where(use_fresh, fresh, self.images[indices])

    def add(self, images: torch.Tensor) -> None:
        images = images[-len(self.images):]
        indices = (
            self.position + torch.arange(len(images), device=images.device)
        ) % len(self.images)
        self.images[indices] = images
        self.position = (self.position + len(images)) % len(self.images)
        self.count = min(self.count + len(images), len(self.images))


def energy_loss(
    model: EnergyModel,
    real_images: torch.Tensor,
    replay: ReplayBuffer,
    langevin_steps: int,
) -> torch.Tensor:
    # Sampling differentiates the images. The returned negatives are detached,
    # so the parameter update treats them as fixed inputs.
    fake_images = sample_langevin(model, replay.draw(real_images), langevin_steps)
    replay.add(fake_images)
    real_images = (
        real_images + LANGEVIN_NOISE_STD * torch.randn_like(real_images)
    ).clamp(-1, 1)
    real_energy, fake_energy = model(torch.cat([real_images, fake_images])).chunk(2)
    contrastive_loss = real_energy.mean() - fake_energy.mean()
    regularization = 0.1 * (real_energy.square().mean() + fake_energy.square().mean())
    return contrastive_loss + regularization


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=["diffusion", "ebm"])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--langevin-steps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--max-batches", type=int, help="Limit each epoch for a quick test")
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.langevin_steps) < 1:
        parser.error("epochs, batch size, and Langevin steps must be positive")
    if args.max_batches is not None and args.max_batches < 1:
        parser.error("max batches must be positive")
    return args


def save_checkpoint(checkpoint: dict, path: Path) -> None:
    """Publish only a complete checkpoint, even if a write is interrupted."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        temporary_path = Path(stream.name)
    try:
        torch.save(checkpoint, temporary_path)
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = choose_device() if args.device == "auto" else torch.device(args.device)
    if device.type == "cpu":
        torch.set_num_threads(4)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    dataset = datasets.CIFAR10(args.data_dir, train=True, download=True, transform=transform)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    output_path = args.output_dir / f"cifar10_{args.model}.pt"

    if args.model == "diffusion":
        model = DiffusionUNet().to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    else:
        model = EnergyModel().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.0, 0.999))
        replay = ReplayBuffer(device)

    print(f"Training {args.model} on {device} with {len(dataset):,} CIFAR-10 images", flush=True)
    loss_history = []
    for epoch in range(1, args.epochs + 1):
        started = time.monotonic()
        model.train()
        total_loss = 0.0
        images_seen = 0
        for batch, (images, _) in enumerate(islice(loader, args.max_batches), start=1):
            images = images.to(device)
            optimizer.zero_grad(set_to_none=True)
            if args.model == "diffusion":
                loss = diffusion_loss(model, images)
            else:
                loss = energy_loss(model, images, replay, args.langevin_steps)
            if not torch.isfinite(loss):
                raise RuntimeError("Training produced a non-finite loss.")
            loss.backward()
            if args.model == "ebm":
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
            optimizer.step()

            total_loss += loss.item() * len(images)
            images_seen += len(images)
            if batch % 100 == 0:
                print(
                    f"Epoch {epoch}/{args.epochs}, batch {batch}: "
                    f"loss={total_loss / images_seen:.4f}",
                    flush=True,
                )

        loss_history.append(total_loss / images_seen)
        save_checkpoint({
            "model": args.model,
            "model_state_dict": {
                name: value.detach().cpu() for name, value in model.state_dict().items()
            },
            "image_shape": (3, 32, 32),
            "epochs": epoch,
            "images_per_epoch": images_seen,
            "batch_size": args.batch_size,
            "seed": args.seed,
            "loss_history": loss_history,
            "langevin_training_steps": args.langevin_steps if args.model == "ebm" else None,
        }, output_path)
        print(
            f"Epoch {epoch}/{args.epochs}: loss={loss_history[-1]:.4f}, "
            f"{time.monotonic() - started:.1f}s; saved {output_path}",
            flush=True,
        )


if __name__ == "__main__":
    main()

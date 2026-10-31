"""Train the Assignment 3 GAN on MNIST."""

import argparse
import csv
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image

from app.mnist_gan import LATENT_DIM, Discriminator, Generator


def initialize_weights(layer):
    if isinstance(layer, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(layer.weight, 0.0, 0.02)
        if layer.bias is not None:
            nn.init.zeros_(layer.bias)
    elif isinstance(layer, nn.BatchNorm2d):
        nn.init.normal_(layer.weight, 1.0, 0.02)
        nn.init.zeros_(layer.bias)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.0002)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.lr <= 0:
        parser.error("epochs, batch size, and learning rate must be positive")

    torch.manual_seed(args.seed)
    if args.device == "auto":
        device = torch.device(
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )
    else:
        device = torch.device(args.device)
    if device.type == "cpu":
        torch.set_num_threads(4)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])
    dataset = datasets.MNIST(args.data_dir, train=True, download=True, transform=transform)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    generator = Generator().to(device)
    discriminator = Discriminator().to(device)
    generator.apply(initialize_weights)
    discriminator.apply(initialize_weights)
    g_optimizer = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=(0.5, 0.999))
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=args.lr, betas=(0.5, 0.999))
    criterion = nn.BCELoss()
    fixed_noise = torch.randn(64, LATENT_DIM, device=device)
    history = []
    print(f"Training on {device}; {len(dataset)} images; {args.epochs} epochs", flush=True)

    for epoch in range(1, args.epochs + 1):
        started = time.monotonic()
        generator.train()
        discriminator.train()
        d_total = g_total = 0.0
        seen = 0

        for real_images, _ in loader:
            real_images = real_images.to(device)
            batch_size = real_images.size(0)
            real_targets = torch.ones(batch_size, 1, device=device)
            fake_targets = torch.zeros(batch_size, 1, device=device)
            noise = torch.randn(batch_size, LATENT_DIM, device=device)
            fake_images = generator(noise)

            # Train D with real images labeled 1 and generated images labeled 0.
            d_optimizer.zero_grad(set_to_none=True)
            d_loss = (
                criterion(discriminator(real_images), real_targets)
                + criterion(discriminator(fake_images.detach()), fake_targets)
            )
            d_loss.backward()
            d_optimizer.step()

            # Keep D fixed while G learns to make D predict 1 for generated images.
            discriminator.requires_grad_(False)
            g_optimizer.zero_grad(set_to_none=True)
            g_loss = criterion(discriminator(fake_images), real_targets)
            g_loss.backward()
            g_optimizer.step()
            discriminator.requires_grad_(True)

            if not torch.isfinite(d_loss) or not torch.isfinite(g_loss):
                raise RuntimeError("Training produced a non-finite loss.")
            d_total += d_loss.item() * batch_size
            g_total += g_loss.item() * batch_size
            seen += batch_size

        row = {
            "epoch": epoch,
            "discriminator_loss": d_total / seen,
            "generator_loss": g_total / seen,
            "seconds": round(time.monotonic() - started, 2),
        }
        history.append(row)
        with (args.output_dir / "mnist_gan_training.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(row))
            writer.writeheader()
            writer.writerows(history)

        generator.eval()
        with torch.inference_mode():
            samples = generator(fixed_noise).cpu()
        save_image(
            samples, args.output_dir / "mnist_gan_samples.png",
            nrow=8, normalize=True, value_range=(-1, 1),
        )
        torch.save({
            "generator_state_dict": {
                name: value.detach().cpu() for name, value in generator.state_dict().items()
            },
            "latent_dim": LATENT_DIM,
            "image_shape": (1, 28, 28),
            "epochs": epoch,
            "training_images": len(dataset),
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "seed": args.seed,
        }, args.output_dir / "mnist_generator.pt")
        print(
            f"Epoch {epoch:02d}/{args.epochs}: "
            f"D loss={row['discriminator_loss']:.4f}, "
            f"G loss={row['generator_loss']:.4f}, "
            f"{row['seconds']:.1f}s", flush=True,
        )

    print(f"Saved generator and sample images to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()

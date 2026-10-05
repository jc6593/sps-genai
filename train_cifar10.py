"""Train the Assignment 2 CNN on CIFAR-10 and save a checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets

from app.cifar10_model import (
    CIFAR10_CLASSES,
    IMAGE_SIZE,
    AssignmentCNN,
    build_image_transform,
)


def choose_device() -> torch.device:
    """Prefer Apple MPS, then CUDA, and otherwise use the CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def create_dataloaders(
    data_dir: Path,
    batch_size: int,
) -> tuple[DataLoader, DataLoader]:
    """Load CIFAR-10 and resize every image to the required 64 x 64."""
    transform = build_image_transform()

    train_dataset = datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=True,
        transform=transform,
    )
    test_dataset = datasets.CIFAR10(
        root=data_dir,
        train=False,
        download=True,
        transform=transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
    )
    return train_loader, test_loader


def train_one_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    max_batches: int | None = None,
) -> tuple[float, float]:
    """Run one training epoch and return average loss and accuracy."""
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for batch_number, (images, labels) in enumerate(data_loader):
        if max_batches is not None and batch_number >= max_batches:
            break

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_examples += batch_size

    if total_examples == 0:
        raise ValueError("The training data loader did not provide any examples.")

    return total_loss / total_examples, total_correct / total_examples


def evaluate(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    max_batches: int | None = None,
) -> tuple[float, float]:
    """Evaluate without updating the model and return loss and accuracy."""
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    with torch.no_grad():
        for batch_number, (images, labels) in enumerate(data_loader):
            if max_batches is not None and batch_number >= max_batches:
                break

            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_correct += (logits.argmax(dim=1) == labels).sum().item()
            total_examples += batch_size

    if total_examples == 0:
        raise ValueError("The test data loader did not provide any examples.")

    return total_loss / total_examples, total_correct / total_examples


def save_checkpoint(
    model: nn.Module,
    checkpoint_path: Path,
    epoch: int,
    test_accuracy: float,
) -> None:
    """Save learned weights together with the preprocessing metadata."""
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "epoch": epoch,
            "test_accuracy": test_accuracy,
            "image_size": IMAGE_SIZE,
            "classes": CIFAR10_CLASSES,
        },
        checkpoint_path,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.0005)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("artifacts/cifar10_cnn.pt"),
    )
    parser.add_argument("--max-train-batches", type=int)
    parser.add_argument("--max-test-batches", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(42)

    device = choose_device()
    print(f"Using device: {device}")

    train_loader, test_loader = create_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
    )
    model = AssignmentCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    for epoch in range(1, args.epochs + 1):
        train_loss, train_accuracy = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            max_batches=args.max_train_batches,
        )
        test_loss, test_accuracy = evaluate(
            model,
            test_loader,
            criterion,
            device,
            max_batches=args.max_test_batches,
        )
        print(
            f"Epoch {epoch:02d}/{args.epochs}: "
            f"train loss={train_loss:.4f}, "
            f"train accuracy={train_accuracy:.2%}, "
            f"test loss={test_loss:.4f}, "
            f"test accuracy={test_accuracy:.2%}"
        )

    save_checkpoint(model, args.checkpoint, args.epochs, test_accuracy)
    print(f"Saved checkpoint: {args.checkpoint}")


if __name__ == "__main__":
    main()

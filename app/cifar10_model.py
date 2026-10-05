"""CNN model required by Assignment 2."""

import torch
import torch.nn.functional as F
from torch import nn
from torchvision import transforms


IMAGE_SIZE = 64
NUM_CLASSES = 10

CIFAR10_CLASSES = (
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)


def build_image_transform() -> transforms.Compose:
    """Return the preprocessing shared by training and API inference."""
    return transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
        ]
    )


class AssignmentCNN(nn.Module):
    """Two-layer CNN matching the architecture in Assignment 2."""

    def __init__(self) -> None:
        super().__init__()

        # [B, 3, 64, 64] -> [B, 16, 64, 64]
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=16,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # [B, 16, 32, 32] -> [B, 32, 32, 32]
        self.conv2 = nn.Conv2d(
            in_channels=16,
            out_channels=32,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        # Two pooling operations reduce 64 x 64 to 16 x 16.
        self.fc1 = nn.Linear(32 * 16 * 16, 100)
        self.fc2 = nn.Linear(100, NUM_CLASSES)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Return one raw classification score (logit) per CIFAR-10 class."""
        features = self.pool(F.relu(self.conv1(images)))
        features = self.pool(F.relu(self.conv2(features)))
        features = torch.flatten(features, start_dim=1)
        features = F.relu(self.fc1(features))
        return self.fc2(features)

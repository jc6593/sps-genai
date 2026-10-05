"""Inference service for the trained Assignment 2 CIFAR-10 model."""

from pathlib import Path

import torch
from PIL import Image

from app.cifar10_model import (
    CIFAR10_CLASSES,
    IMAGE_SIZE,
    AssignmentCNN,
    build_image_transform,
)


DEFAULT_CHECKPOINT_PATH = (
    Path(__file__).resolve().parents[1] / "artifacts" / "cifar10_cnn.pt"
)


class Cifar10Classifier:
    """Load one trained model and reuse it for API predictions."""

    def __init__(self, checkpoint_path: Path = DEFAULT_CHECKPOINT_PATH) -> None:
        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"CIFAR-10 checkpoint was not found: {checkpoint_path}"
            )

        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )
        if checkpoint.get("image_size") != IMAGE_SIZE:
            raise ValueError("Checkpoint image size does not match the API model.")
        if tuple(checkpoint.get("classes", ())) != CIFAR10_CLASSES:
            raise ValueError("Checkpoint classes do not match CIFAR-10.")

        self.model = AssignmentCNN()
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        self.transform = build_image_transform()

    def predict(
        self,
        image: Image.Image,
    ) -> tuple[int, str, float, dict[str, float]]:
        """Classify one image and return its label and class probabilities."""
        image_tensor = self.transform(image.convert("RGB")).unsqueeze(0)

        with torch.no_grad():
            logits = self.model(image_tensor)
            probabilities_tensor = logits.softmax(dim=1)[0]

        predicted_index = int(probabilities_tensor.argmax().item())
        probabilities = {
            class_name: float(probability)
            for class_name, probability in zip(
                CIFAR10_CLASSES,
                probabilities_tensor.tolist(),
                strict=True,
            )
        }
        predicted_class = CIFAR10_CLASSES[predicted_index]
        confidence = probabilities[predicted_class]
        return predicted_index, predicted_class, confidence, probabilities

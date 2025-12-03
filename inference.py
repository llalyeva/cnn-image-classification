# Description:
# This file should be used for performing inference on a network
# Usage: inference.py <dataset_path> <model_path>

from argparse import ArgumentParser
from pathlib import Path

import albumentations as A
import torch
import csv
import numpy as np
from network import StandardCNN
from PIL import Image


# declaration for this function should not be changed
@torch.no_grad()  # do not calculate the gradients
def inference(dataset_path: Path, model_path: Path) -> None:
    """Performs inference on the given dataset using the specified model.

    Args:
        dataset_path: Path to the dataset. The function processes all PNG images in
            this directory (optionally recursively in its subdirectories).
        model_path: Path to the model file.

    Saves:
        predictions to 'output_predictions' folder. The files can be saved in a flat
            structure with the same name as the input file.
    """
    # Check for available GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Computing with {}!".format(device))
    dataset_path = Path(dataset_path)
    model_path = Path(model_path)
    # loading the model
    model = StandardCNN().to(device)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    transform = A.Compose([
        A.Resize(64, 64),
        A.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        A.ToTensorV2(),
    ])

    out_dir = Path("output_predictions")
    out_dir.mkdir(exist_ok=True)

    images = list(dataset_path.rglob("*.png"))

    with open(out_dir / "predictions.csv", "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["filename", "class_id"])

        for img_path in images:
            img = Image.open(img_path).convert("RGB")
            img_np = np.array(img)
            x = transform(image=img_np)['image'].unsqueeze(0).to(device)
            out = model(x)
            pred = out.argmax(dim=1).item()
            writer.writerow([img_path.stem, pred])


def main() -> None:
    parser = ArgumentParser(description="Inference script for a neural network.")
    parser.add_argument("dataset_path", type=Path, help="Path to the dataset")
    parser.add_argument("model_path", type=Path, help="Path to the model weights")
    args = parser.parse_args()
    inference(args.dataset_path, args.model_path)


if __name__ == "__main__":
    main()

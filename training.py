# Description:
# This file should be used for performing training of a network
# Usage: python training.py <dataset_path>

from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import albumentations as A
from tqdm import tqdm
from dataset import AugmentedDataset, DeviceDataLoader, RawCachingImageFolder
from network import StandardCNN
from torch import Tensor, nn
from torch import optim
from torch.utils.data import DataLoader, random_split
from torchview import draw_graph


def evaluate(model, loss_func, test_dl, device):
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    total = 0

    with torch.no_grad():
        for xb, yb in test_dl:
            xb, yb = xb.to(device), yb.to(device)
            outputs = model(xb)
            loss = loss_func(outputs, yb)

            running_loss += loss.item() * xb.size(0)
            preds = outputs.argmax(dim=1)
            running_corrects += (preds == yb).sum().item()
            total += xb.size(0)

    test_loss = running_loss / total
    test_acc = running_corrects / total * 100
    return test_loss, test_acc


def loss_batch(model, loss_func, xb, yb, opt=None):
    outputs = model(xb)
    loss = loss_func(outputs, yb)
    preds = torch.argmax(outputs, dim=1)
    corrects = torch.sum(preds == yb).item()

    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()

    return loss.item(), xb.size(0), corrects


# sample function for model architecture visualization
# draw_graph function saves an additional file: Graphviz DOT graph file, it's not necessary to delete it
def draw_network_architecture(net: nn.Module, input_sample: Tensor) -> None:
    # saves visualization of model architecture to the model_architecture.png
    draw_graph(
        net,
        input_sample,
        graph_dir="TB",
        save_graph=True,
        filename="model_architecture",
        expand_nested=True,
    )


# sample function for losses visualization
def plot_learning_curves(
    train_losses: list[float], validation_losses: list[float]
) -> None:
    plt.figure(figsize=(10, 5))
    plt.title("Train and Evaluation Losses During Training")
    plt.plot(train_losses, label="train_loss")
    plt.plot(validation_losses, label="validation_loss")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("learning_curves.png")


def fit(epochs, model, loss_func, opt, train_dl, valid_dl, device, scheduler =None):
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    for epoch in tqdm(range(epochs), maxinterval=epochs, desc="FIT"):
        model.train()
        running_loss = 0.0
        running_corrects = 0
        total = 0

        for xb, yb in train_dl:
            loss, batch_size, corrects = loss_batch(model, loss_func, xb, yb, opt)
            running_loss += loss * batch_size
            running_corrects += corrects
            total += batch_size

        epoch_loss = running_loss / total
        epoch_acc = (running_corrects / total) * 100
        train_losses.append(epoch_loss)
        train_accs.append(epoch_acc)

        model.eval()
        running_loss = 0.0
        running_corrects = 0
        total = 0
        with torch.no_grad():
            for xb, yb in valid_dl:
                loss, batch_size, corrects = loss_batch(model, loss_func, xb, yb)
                running_loss += loss * batch_size
                running_corrects += corrects
                total += batch_size

        val_loss = running_loss / total
        if scheduler is not None:
            scheduler.step(val_loss)

        val_acc = (running_corrects / total) * 100
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.2f}% | Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
    
    return train_losses, val_losses


# declaration for this function should not be changed
def training(dataset_path: Path) -> None:
    """Performs training on the given dataset.

    Args:
        dataset_path: Path to the dataset.

    Saves:
        - model.pt (trained model)
        - learning_curves.png (learning curves generated during training)
        - model_architecture.png (a scheme of model's architecture)
    """
    # Check for available GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Computing with {}!".format(device))

    train_aug = A.Compose([
        A.Resize(64, 64),
        A.HorizontalFlip(p=0.5),
        A.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1),
        A.GaussNoise(),
        A.Affine(rotate=(-30,30), p=0.5),
        A.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225)),
        A.ToTensorV2(),
    ])

    val_tf = A.Compose([
        A.Resize(64, 64),
        A.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225)),
        A.ToTensorV2(),
    ])

    raw_ds = RawCachingImageFolder(dataset_path, num_workers=8)

    counts = torch.tensor([414, 1569, 1064, 1664, 463, 1100], dtype=torch.float)
    weights = 1.0 / counts
    weights1 = weights / weights.sum() * len(counts)

    n = len(raw_ds)
    batch_size = 32
    n_train = int(0.8 * n)
    n_val = int(0.1 * n)
    n_test = n - n_train - n_val
    train_raw, val_raw, test_raw = random_split(raw_ds, [n_train, n_val, n_test])
    
    train_ds = AugmentedDataset(train_raw, transform=train_aug, times=3)
    val_ds = AugmentedDataset(val_raw, transform=val_tf)
    test_ds = AugmentedDataset(test_raw, transform=val_tf)

    train_dl = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle = True,
        num_workers=4,
        pin_memory=True,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    test_dl = DataLoader(
        test_ds,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    print(f"train dataset size: {len(train_ds)} ({len(train_raw)} originals x 3)")
    print(f"validation dataset size: {len(val_ds)}")
    print(f"test dataset size: {len(test_ds)}")

    net = StandardCNN().to(device)
    input_sample = torch.zeros((1, 3, 64, 64))
    draw_network_architecture(net, input_sample)
    weights1 = weights1.to(device)

    opt = optim.Adam(net.parameters(), lr=3e-4, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode='min', factor=0.3, patience=3
    )
    loss_func = nn.CrossEntropyLoss(weight=weights1.to(device))


    valid_dl = DeviceDataLoader(val_dl, device)
    train_dl = DeviceDataLoader(train_dl, device)

    # train the network
    train_losses, val_losses = fit(
        epochs=30, model=net, loss_func=loss_func, opt=opt, 
        train_dl=train_dl, valid_dl=valid_dl, device=device, scheduler=scheduler,
    )

    test_loss, test_acc = evaluate(net, loss_func, test_dl, device)

    print(f"Test loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_acc:.2f}%")

    # save the trained model and plot the losses, feel free to create your own functions
    torch.save(net.state_dict(), "model.pt")
    plot_learning_curves(train_losses, val_losses)


# #### code below should not be changed ############################################################################


def main() -> None:
    parser = ArgumentParser(description="Training script.")
    parser.add_argument("dataset_path", type=Path, help="Path to the dataset")
    args = parser.parse_args()
    training(args.dataset_path)


if __name__ == "__main__":
    main()

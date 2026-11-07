
import random
import numpy as np
import json
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from pathlib import Path
from torch.utils.data import DataLoader, random_split
from src.model import FashionCNN


def main():
    # 1. Reproducibility
    SEED = 42

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # 2. Configuration
    BATCH_SIZE = 64
    EPOCHS = 10
    LEARNING_RATE = 0.001

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # 3. Load dataset
    transform = transforms.ToTensor()

    train_dataset = torchvision.datasets.FashionMNIST(
        root="./data",
        train=True,
        download=True,
        transform=transform
    )

    # Split training data into 48,000 train and 12,000 validation
    split_generator = torch.Generator().manual_seed(SEED)

    train_subset, val_subset = random_split(
        train_dataset,
        [48000, 12000],
        generator=split_generator
    )

    train_loader = DataLoader(
        train_subset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=torch.Generator().manual_seed(SEED)
    )

    val_loader = DataLoader(
        val_subset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    # 4. Initialize model
    model = FashionCNN().to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    # 5. Training function
    def train_one_epoch(model, loader, criterion, optimizer):
        model.train()

        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(loader)
        epoch_accuracy = 100 * correct / total

        return epoch_loss, epoch_accuracy

    # 6. Validation function
    def validate_one_epoch(model, loader, criterion):
        model.eval()

        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                running_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(loader)
        epoch_accuracy = 100 * correct / total

        return epoch_loss, epoch_accuracy

    # 7. Train model and save the best checkpoint
    history = {
        "train_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_accuracy": []
    }

    best_val_accuracy = 0.0

    checkpoint_dir = Path("checkpoints")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(EPOCHS):

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer
        )

        val_loss, val_acc = validate_one_epoch(
            model, val_loader, criterion
        )

        history["train_loss"].append(train_loss)
        history["train_accuracy"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_acc)

        print(
            f"Epoch [{epoch + 1}/{EPOCHS}] "
            f"Train Loss: {train_loss:.4f} "
            f"Train Acc: {train_acc:.2f}% "
            f"Val Loss: {val_loss:.4f} "
            f"Val Acc: {val_acc:.2f}%"
        )

        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc

            torch.save(
                model.state_dict(),
                checkpoint_dir / "best_model.pth"
            )

            print("Best model checkpoint saved!")
    with open("training_history.json", "w") as f:
        json.dump(history, f, indent=4)

    print("\nTraining completed.")
    print(f"Best Validation Accuracy: {best_val_accuracy:.2f}%")
    print("Checkpoint saved at: checkpoints/best_model.pth")
    print("Training history saved at: training_history.json")


if __name__ == "__main__":
    main()

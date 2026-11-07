
import torch
import torchvision
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
from src.model import FashionCNN


# 1. Configuration
BATCH_SIZE = 64

classes = [
    "T-shirt", "Trouser", "Pullover", "Dress",
    "Coat", "Sandal", "Shirt", "Sneaker",
    "Bag", "Ankle Boot"
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device:", device)


# 2. Load test dataset
transform = transforms.ToTensor()

test_dataset = torchvision.datasets.FashionMNIST(
    root="./data",
    train=False,
    download=True,
    transform=transform
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# 3. Load trained model
model = FashionCNN().to(device)

checkpoint_path = "checkpoints/best_model.pth"

model.load_state_dict(
    torch.load(checkpoint_path, map_location=device, weights_only=True)
)

model.eval()


# 4. Evaluate model
correct = 0
total = 0
running_loss = 0.0

criterion = torch.nn.CrossEntropyLoss()

all_predictions = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        loss = criterion(outputs, labels)
        running_loss += loss.item()

        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        all_predictions.extend(predicted.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())


# 5. Calculate metrics
test_loss = running_loss / len(test_loader)
test_accuracy = 100 * correct / total

print("\nTest Results")
print("-------------------------")
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.2f}%")
print(f"Correct Predictions: {correct}/{total}")


# 6. Classification Report
print("\nClassification Report")
print(classification_report(
    all_labels,
    all_predictions,
    target_names=classes
))


# 7. Confusion Matrix
print("\nConfusion Matrix")
print(confusion_matrix(all_labels, all_predictions))

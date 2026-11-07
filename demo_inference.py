import torch
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from src.model import FashionCNN

# Class names
classes = [
    "T-shirt", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle Boot"
]

# Load test dataset
transform = transforms.ToTensor()

test_dataset = datasets.FashionMNIST(
    root="./data",
    train=False,
    download=False,
    transform=transform
)

# Load trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FashionCNN().to(device)

checkpoint = torch.load(
    "checkpoints/best_model.pth",
    map_location=device
)

# Support either a raw state_dict or a checkpoint dictionary
if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
    model.load_state_dict(checkpoint["model_state_dict"])
elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
    model.load_state_dict(checkpoint["state_dict"])
else:
    model.load_state_dict(checkpoint)

model.eval()

# Find one success and one failure.
# Prefer a Shirt incorrectly predicted as T-shirt for the failure case.
success = None
failure = None
shirt_failure = None

with torch.no_grad():
    for i in range(len(test_dataset)):
        image, true_label = test_dataset[i]
        output = model(image.unsqueeze(0).to(device))
        predicted = output.argmax(dim=1).item()

        if predicted == true_label and success is None:
            success = (image, true_label, predicted)

        if predicted != true_label:
            if failure is None:
                failure = (image, true_label, predicted)

            if true_label == 6 and predicted == 0:
                shirt_failure = (image, true_label, predicted)
                break

# Use the preferred Shirt failure if found.
if shirt_failure is not None:
    failure = shirt_failure

if success is None or failure is None:
    raise RuntimeError("Could not find both a success and failure example.")

# Display predictions
fig, axes = plt.subplots(1, 2, figsize=(9, 4))

for ax, sample, title in zip(
    axes,
    [success, failure],
    ["Success Case", "Failure Case"]
):
    image, true_label, predicted = sample

    ax.imshow(image.squeeze(), cmap="gray")
    ax.set_title(
        f"{title}\nTrue: {classes[true_label]}\n"
        f"Predicted: {classes[predicted]}"
    )
    ax.axis("off")

    print(
        f"{title}: True = {classes[true_label]}, "
        f"Predicted = {classes[predicted]}"
    )

plt.tight_layout()
plt.show()

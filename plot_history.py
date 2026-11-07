import json
import matplotlib.pyplot as plt

with open("training_history.json", "r") as f:
    history = json.load(f)

epochs = range(1, len(history["train_loss"]) + 1)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Loss Curve
axes[0].plot(epochs, history["train_loss"], label="Training Loss")
axes[0].plot(epochs, history["val_loss"], label="Validation Loss")
axes[0].set_title("Training and Validation Loss")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
axes[0].legend()
axes[0].grid(True)

# Accuracy Curve
axes[1].plot(epochs, history["train_accuracy"], label="Training Accuracy")
axes[1].plot(epochs, history["val_accuracy"], label="Validation Accuracy")
axes[1].set_title("Training and Validation Accuracy")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Accuracy (%)")
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.savefig("training_curves.png", dpi=300)
plt.show()

print("Training curves saved to training_curves.png")
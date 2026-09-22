import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

from torchvision import datasets, transforms

import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
    f1_score
)


# ============================================================
# 1. BASIC SETTINGS
# ============================================================

OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

BATCH_SIZE = 128
EPOCHS = 5
LEARNING_RATE = 0.001

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 60)
print("EXPERIMENT 10 - CNN DIGIT CLASSIFICATION")
print("=" * 60)

print("Device:", DEVICE)


# ============================================================
# 2. DATA PREPROCESSING
# ============================================================

print("\nLoading MNIST dataset...")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])


# Download MNIST dataset
train_full = datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=transform
)

test_dataset = datasets.MNIST(
    root="data",
    train=False,
    download=True,
    transform=transform
)


# Split training data into training and validation
train_size = int(0.9 * len(train_full))
validation_size = len(train_full) - train_size

train_dataset, validation_dataset = random_split(
    train_full,
    [train_size, validation_size],
    generator=torch.Generator().manual_seed(42)
)


train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

validation_loader = DataLoader(
    validation_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


print("Training samples:", len(train_dataset))
print("Validation samples:", len(validation_dataset))
print("Testing samples:", len(test_dataset))


# ============================================================
# 3. CNN MODEL
# ============================================================

class CNNModel(nn.Module):

    def __init__(self):
        super(CNNModel, self).__init__()

        self.features = nn.Sequential(

            # First convolution block
            nn.Conv2d(
                in_channels=1,
                out_channels=32,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.MaxPool2d(
                kernel_size=2
            ),


            # Second convolution block
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.MaxPool2d(
                kernel_size=2
            )
        )


        self.classifier = nn.Sequential(

            nn.Flatten(),

            nn.Linear(
                64 * 7 * 7,
                128
            ),

            nn.ReLU(),

            nn.Dropout(0.25),

            nn.Linear(
                128,
                10
            )
        )


    def forward(self, x):

        x = self.features(x)

        x = self.classifier(x)

        return x


model = CNNModel().to(DEVICE)

print("\nCNN Model:")
print(model)


# ============================================================
# 4. LOSS FUNCTION AND OPTIMIZER
# ============================================================

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# ============================================================
# 5. TRAINING
# ============================================================

print("\nStarting CNN training...")

train_losses = []
validation_losses = []

train_accuracies = []
validation_accuracies = []


for epoch in range(EPOCHS):

    # -------------------------
    # Training
    # -------------------------

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    start_time = time.time()


    for images, labels in train_loader:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)


        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()

        optimizer.step()


        running_loss += loss.item()

        _, predicted = torch.max(
            outputs.data,
            1
        )

        total += labels.size(0)

        correct += (
            predicted == labels
        ).sum().item()


    train_loss = (
        running_loss /
        len(train_loader)
    )

    train_accuracy = (
        100.0 * correct / total
    )


    # -------------------------
    # Validation
    # -------------------------

    model.eval()

    validation_loss = 0.0
    validation_correct = 0
    validation_total = 0


    with torch.no_grad():

        for images, labels in validation_loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

            validation_loss += loss.item()

            _, predicted = torch.max(
                outputs.data,
                1
            )

            validation_total += labels.size(0)

            validation_correct += (
                predicted == labels
            ).sum().item()


    validation_loss /= len(
        validation_loader
    )

    validation_accuracy = (
        100.0 *
        validation_correct /
        validation_total
    )


    train_losses.append(train_loss)
    validation_losses.append(
        validation_loss
    )

    train_accuracies.append(
        train_accuracy
    )

    validation_accuracies.append(
        validation_accuracy
    )


    elapsed = time.time() - start_time


    print(
        f"Epoch [{epoch + 1}/{EPOCHS}] | "
        f"Train Loss: {train_loss:.4f} | "
        f"Train Acc: {train_accuracy:.2f}% | "
        f"Val Loss: {validation_loss:.4f} | "
        f"Val Acc: {validation_accuracy:.2f}% | "
        f"Time: {elapsed:.1f}s"
    )


# ============================================================
# 6. TRAINING / VALIDATION CURVES
# ============================================================

epochs_range = range(
    1,
    EPOCHS + 1
)


# Accuracy curve
plt.figure(figsize=(8, 6))

plt.plot(
    epochs_range,
    train_accuracies,
    marker="o",
    label="Training Accuracy"
)

plt.plot(
    epochs_range,
    validation_accuracies,
    marker="o",
    label="Validation Accuracy"
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")

plt.title(
    "Training and Validation Accuracy"
)

plt.legend()
plt.grid(True)

plt.savefig(
    f"{OUTPUT_DIR}/accuracy_curve.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# Loss curve
plt.figure(figsize=(8, 6))

plt.plot(
    epochs_range,
    train_losses,
    marker="o",
    label="Training Loss"
)

plt.plot(
    epochs_range,
    validation_losses,
    marker="o",
    label="Validation Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.title(
    "Training and Validation Loss"
)

plt.legend()
plt.grid(True)

plt.savefig(
    f"{OUTPUT_DIR}/loss_curve.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 7. TESTING
# ============================================================

print("\nEvaluating model on test dataset...")

model.eval()

all_predictions = []
all_labels = []

test_correct = 0
test_total = 0


with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        outputs = model(images)

        _, predicted = torch.max(
            outputs.data,
            1
        )

        test_total += labels.size(0)

        test_correct += (
            predicted == labels
        ).sum().item()

        all_predictions.extend(
            predicted.cpu().numpy()
        )

        all_labels.extend(
            labels.cpu().numpy()
        )


test_accuracy = (
    100.0 *
    test_correct /
    test_total
)


print(
    f"\nTest Accuracy: "
    f"{test_accuracy:.2f}%"
)


# ============================================================
# 8. PRECISION, RECALL AND F1-SCORE
# ============================================================

precision = precision_score(
    all_labels,
    all_predictions,
    average="weighted",
    zero_division=0
)

recall = recall_score(
    all_labels,
    all_predictions,
    average="weighted",
    zero_division=0
)

f1 = f1_score(
    all_labels,
    all_predictions,
    average="weighted",
    zero_division=0
)


print("\n========================================")
print("PERFORMANCE METRICS")
print("========================================")

print(
    f"Accuracy  : {test_accuracy:.2f}%"
)

print(
    f"Precision : {precision:.4f}"
)

print(
    f"Recall    : {recall:.4f}"
)

print(
    f"F1-Score  : {f1:.4f}"
)


# Classification report
report = classification_report(
    all_labels,
    all_predictions,
    digits=4
)

print("\nClassification Report:")
print(report)


with open(
    f"{OUTPUT_DIR}/classification_report.txt",
    "w"
) as file:

    file.write(report)


# ============================================================
# 9. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    all_labels,
    all_predictions
)


plt.figure(figsize=(9, 8))

plt.imshow(cm)

plt.title(
    "MNIST CNN Confusion Matrix"
)

plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")

plt.xticks(
    range(10)
)

plt.yticks(
    range(10)
)

plt.colorbar()


for i in range(10):

    for j in range(10):

        plt.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center"
        )


plt.savefig(
    f"{OUTPUT_DIR}/confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 10. PREDICTION ON UNSEEN TEST IMAGES
# ============================================================

print("\nGenerating sample predictions...")

images, labels = next(
    iter(test_loader)
)

images_device = images.to(DEVICE)


with torch.no_grad():

    outputs = model(
        images_device
    )

    probabilities = torch.softmax(
        outputs,
        dim=1
    )

    predictions = torch.argmax(
        probabilities,
        dim=1
    )


plt.figure(
    figsize=(12, 8)
)


for i in range(12):

    plt.subplot(
        3,
        4,
        i + 1
    )

    image = images[i].squeeze().numpy()

    # Undo normalization for display
    image = (
        image * 0.3081
    ) + 0.1307

    image = np.clip(
        image,
        0,
        1
    )

    plt.imshow(
        image,
        cmap="gray"
    )

    actual = labels[i].item()

    predicted = predictions[i].item()

    confidence = (
        probabilities[i][predicted]
        .item() * 100
    )

    plt.title(
        f"Actual: {actual}\n"
        f"Predicted: {predicted}\n"
        f"Confidence: {confidence:.1f}%"
    )

    plt.axis("off")


plt.tight_layout()

plt.savefig(
    f"{OUTPUT_DIR}/sample_predictions.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 11. SAVE TRAINED MODEL
# ============================================================

torch.save(
    model.state_dict(),
    f"{OUTPUT_DIR}/mnist_cnn_model.pth"
)


# ============================================================
# 12. FINAL RESULT
# ============================================================

print("\n========================================")
print("EXPERIMENT 10 COMPLETED SUCCESSFULLY")
print("========================================")

print(
    f"Final Test Accuracy: "
    f"{test_accuracy:.2f}%"
)

print(
    f"Precision: {precision:.4f}"
)

print(
    f"Recall: {recall:.4f}"
)

print(
    f"F1-Score: {f1:.4f}"
)

print("\nGenerated files:")

for file in sorted(
    os.listdir(OUTPUT_DIR)
):

    print(
        "-",
        file
    )
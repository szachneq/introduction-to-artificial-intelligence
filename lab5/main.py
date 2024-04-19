import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import KMNIST
from sklearn.metrics import confusion_matrix

# Set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model
model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(28 * 28, 512),
    nn.ReLU(),
    nn.Linear(512, 256),
    nn.ReLU(),
    nn.Linear(256, 10),
).to(device)

# PROGRAM SETTINGS
# - to play with
# Number of hidden layers (including 0 hidden layers - a linear model)
# Width (number of neurons in hidden layers)
learning_rate = 0.01
batch_size = 64  # amount of images used in one training mini-batch
# optimizer
optimizer = optim.SGD(
    model.parameters(), lr=learning_rate
)  # Stochastic Gradient Descent (SGD)
# optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9) # SGD with momentum
# optimizer = optim.Adam(model.parameters(), lr=learning_rate) # Adam (Adaptive Moment Estimation) (Also a form of SGD)

# - to rather leave untouched
training_ratio = 0.8  # percent of dataset used as training data
# loss function
# criterion = nn.MSELoss() # Mean Squared Error
# criterion = nn.L1Loss() # Mean Absolute Error
criterion = nn.CrossEntropyLoss()  # Cross Entropy
num_epochs = 10

# Transformations
transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ]
)

# Dataset
dataset = KMNIST(
    root=".",
    transform=transform,
)
train_size = int(training_ratio * len(dataset))
validation_size = len(dataset) - train_size
train_dataset, validation_dataset = random_split(dataset, [train_size, validation_size])

# Dataloaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)


# Validation function
def validate_epoch(loader, model, criterion):
    model.eval()
    total_loss, total_correct = 0, 0

    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total_correct += predicted.eq(targets).sum().item()

    return total_loss / len(loader), total_correct / len(loader.dataset)


# Training function
def train_epoch(loader, model, criterion, optimizer):
    model.train()
    total_loss, total_correct = 0, 0

    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total_correct += predicted.eq(targets).sum().item()

    return total_loss / len(loader), total_correct / len(loader.dataset)


# Training and validation loop
for epoch in range(num_epochs):
    train_loss, train_accuracy = train_epoch(train_loader, model, criterion, optimizer)
    val_loss, val_accuracy = validate_epoch(validation_loader, model, criterion)

    print(
        f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}"
    )
    print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}")

# For evaluation, please create plots visualizing:
# • The loss value for every learning step,
# • Accuracy on the training and validation set after each epoch
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

# Transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Dataset
dataset = KMNIST(root=".", train=True, download=True, transform=transform)
train_size = int(0.8 * len(dataset))
validation_size = len(dataset) - train_size
train_dataset, validation_dataset = random_split(dataset, [train_size, validation_size])

# Dataloaders
batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)

# Model
model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(28*28, 512),
    nn.ReLU(),
    nn.Linear(512, 256),
    nn.ReLU(),
    nn.Linear(256, 10)
).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# Validation function with confusion matrix
def validate_epoch(loader, model, criterion):
    model.eval()
    total_loss, total_correct = 0, 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total_correct += predicted.eq(targets).sum().item()
            all_preds.append(predicted.cpu())
            all_targets.append(targets.cpu())
    
    # Concatenate all predictions and targets across batches
    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    
    # Compute the confusion matrix
    conf_mat = confusion_matrix(all_targets, all_preds)
    
    return total_loss / len(loader), total_correct / len(loader.dataset), conf_mat

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
num_epochs = 10
for epoch in range(num_epochs):
    train_loss, train_accuracy = train_epoch(train_loader, model, criterion, optimizer)
    val_loss, val_accuracy, val_conf_mat = validate_epoch(validation_loader, model, criterion)
    
    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}")
    print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}")
    print("Confusion Matrix:\n", val_conf_mat)

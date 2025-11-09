# Importing the requied libraries

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

# Defining the LeNet-5 architecture

class LeNet5(nn.Module):
    """
    LeNet-5 architecture implementation in PyTorch.
    """

    def __init__(self):
        super(LeNet5, self).__init__()
        self.activation = nn.Tanh()

        # C1: Convolutional layer
        # Input: 32x32x1 -> Output: 28x28x6
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=2)

        # S2: Sub-sampling (Average Pooling) layer
        # Input: 28x28x6 -> Output: 14x14x6
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)

        # C3: Convolutional layer
        # Input: 14x14x6 -> Output: 10x10x16
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1)

        # S4: Sub-sampleing (Average Pooling) layer
        # Input: 10x10x16 -> Output: 5x5x16
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)

        # C5: Convolutional layer
        # Input: 5x5x16 -> Output: 120
        self.fc1 = nn.Linear(in_features=120, out_features=84)

        # Output layer
        # Input: 84 -> Output: 10 (for 10 digits)
        self.fc2 = nn.Linear(in_features=84, out_features=10)

    def forward(self, x):
        # C1-> S2
        x = self.conv1(x)
        x = self.activation(x)
        x = self.pool1(x)

        # C3 -> S4
        x = self.conv2(x)
        x = self.activation(x)
        x = self.pool2(x)

        # C5 
        x = self.conv3(x)
        x = self.activation(x)

        # Flatten the output for the fully connected layer
        x = torch.flatten(x, 1)

        # F6
        x = self.fc1(x)
        x = self.activation(x)

        # Output layer
        x = self.fc2(x)
        return x

def print_model_summary(model):
    print("LeNet-5 Model Summary:")
    print("-"*60)
    total_params = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            num_params = param.numel()
            total_params += num_params
            print(f"{name:<40} | Params: {num_params}")
    print("-", 60)
    print(f"Total trainable parameters: {total_params}")

# Instatiate and print summary
lenet_model = LeNet5()
print(lenet_model)

# MNIST Dataset Preparation

# """
# We load the MNIST dataset using `torchvision`. We apply two transformations:
# 1. `Pad(2)`: To resize the 28x28 images to 32x32, as required by LeNet-5.
# 2. `ToTensor()`: To convert the images to PyTorch tensors.
# 3. `Normalize()`: To scale pixel values to a standard range.
# """

# Defining transformations
transform = transforms.Compose([
    transforms.Pad(2), # Resize to 32x32
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Load datasets
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# Create dataloaders
train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=1000, shuffle=False)

# Viusalizing some samples from the dataset
def visualize_samples(loader):
    """
    Visualizes a batch of images from the dataloader.
    """
    # Get a batch of training data
    images, labels = next(iter(loader))

    fig = plt.figure(figsize=(10, 4))
    for idx in np.arange(10):
        ax = fig.add_subplot(2, 5, idx + 1, xticks=[], yticks=[])
        # Un-normalize and display
        img = images[idx].squeeze()
        ax.imshow(img, cmap='gray')
        ax.set_title(f"Label: {labels[idx].item()}")
    plt.suptitle("Sample MNIST Digits (Padded to 32x32)")
    plt.show()

print("Visualizing some training samples...")
#visualize_samples(train_loader)

# Defining the training process and evaluation functions

# Training function
def train_model(model, train_loader, test_loader, epochs=10, learning_rate=0.01):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)  # Learning rate = 0.01

    history = {
        'train_loss': [], 'train_accuracy': [],
        'val_loss': [], 'val_accuracy': []
    }
    
    for epoch in epochs:
        model.train()
        running_loss, correct_train, total_train = 0, 0, 0
        
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()                    
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
        
        train_loss = running_loss / len(train_loader)
        train_accuracy = 100 * correct_train / total_train
        history['train_loss'].append(train_loss)
        history['train_accuracy'].append(train_accuracy)

# Validation phase
        
        model.eval()
        running_loss, correct_val, total_val = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in test_loader:
                outputs = model(images)
                loss = criterion(outputs, labels)
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

        val_loss = running_loss / len(test_loader)
        val_acc = 100 * correct_val / total_val
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        print(f"Epoch {epoch+1}/{epochs} | "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

    print("Training finished.")
    return history

# Train the LeNet-5 model
print("Starting training for original LeNet-5 (Tanh activation)...")
lenet_history = train_model(lenet_model, train_loader, test_loader, epochs=15)

# Plotting training curves

def plot_curves(history, title):
    """
    Plots training and validation loss and accuracy curves.
    """
    epochs = range(1, len(history['train_loss']) + 1)
    plt.figure(figsize=(12, 5))

    # Plot Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], 'b-o', label='Training Loss')
    plt.plot(epochs, history['val_loss'], 'r-o', label='Validation Loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.legend()

    # Plot Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['train_acc'], 'b-o', label='Training Accuracy')
    plt.plot(epochs, history['val_acc'], 'r-o', label='Validation Accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.legend()

    plt.suptitle(title)
    plt.show()

plot_curves(lenet_history, "LeNet-5 (Tanh) Training Curves")
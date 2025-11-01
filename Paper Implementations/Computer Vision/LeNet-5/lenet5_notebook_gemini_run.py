
# ############################################################################
# # LeNet-5 Explained: From Theory to PyTorch Implementation
# ############################################################################
#
# # This script is a self-contained guide to understanding and implementing the
# # LeNet-5 architecture, as proposed by LeCun et al. in their 1998 paper.
# # It is structured like a Jupyter notebook, with Markdown-style comments
# # explaining each section, followed by the corresponding Python code.

# ############################################################################
# # 1. Introduction and Theory
# ############################################################################

# """
# ### What is LeNet-5?
#
# LeNet-5 is one of the earliest Convolutional Neural Networks (CNNs) and a
# foundational model in the history of deep learning. It was designed by Yann
# LeCun and his colleagues for handwritten and machine-printed character
# recognition. Its success on the MNIST handwritten digit dataset helped propel
# the field of neural networks forward.
#
# ### The Problem it Solved
#
# Before LeNet-5, character recognition was a major challenge. Traditional
# machine learning and computer vision techniques struggled with the variability
# in handwriting (style, scale, rotation, etc.). LeNet-5 demonstrated that a
# neural network with a specialized architecture could learn hierarchical
# features directly from pixel data, making it robust to these variations.
#
# ### Core Concepts
#
# LeNet-5 introduced and popularized several key concepts that are still
# fundamental to modern CNNs:
#
# 1.  **Local Receptive Fields:** Each neuron in a convolutional layer is
#     connected to only a small, localized region of the input image. This
#     allows the network to learn simple features like edges and corners first.
#
# 2.  **Shared Weights:** The same set of weights (a kernel or filter) is
#     convolved across the entire image. This drastically reduces the number of
#     trainable parameters compared to a fully connected network and makes the
#     network translation-invariant.
#
# 3.  **Sub-sampling (Pooling):** After a convolution, pooling layers (average
#     pooling in LeNet-5's case) reduce the spatial resolution of the feature
#     maps. This makes the learned features more robust to small shifts and
#     distortions and further reduces the computational load.
#
# ### LeNet-5 Architecture (Layer-by-Layer)
#
# The architecture is a sequence of convolutional, pooling, and fully connected
# layers.
#
# - **Input:** 32x32 grayscale image.
# - **C1 (Convolution):** 6 filters of size 5x5. Output: 6 feature maps of 28x28.
# - **S2 (Sub-sampling/Pooling):** Average pooling with a 2x2 window. Output: 6 feature maps of 14x14.
# - **C3 (Convolution):** 16 filters of size 5x5. Output: 16 feature maps of 10x10.
# - **S4 (Sub-sampling/Pooling):** Average pooling with a 2x2 window. Output: 16 feature maps of 5x5.
# - **C5 (Convolution/Fully Connected):** 120 filters of size 5x5. Output: 120 feature maps of 1x1. This is equivalent to a fully connected layer.
# - **F6 (Fully Connected):** 84 units.
# - **Output (Softmax):** 10 units, one for each digit (0-9).
#
# ### Differences from Modern CNNs
#
# - **Activation Function:** LeNet-5 used `sigmoid` or `tanh` activations. Modern
#   CNNs almost universally use `ReLU` (Rectified Linear Unit) and its variants,
#   which help mitigate the vanishing gradient problem.
# - **Pooling:** LeNet-5 used average pooling. Modern CNNs prefer max pooling,
#   which is often more effective at capturing the most salient features.
# - **Regularization:** LeNet-5 had no explicit regularization like Dropout or
#   Batch Normalization, which are standard in modern architectures to prevent
#   overfitting and stabilize training.
# - **Optimizer:** The original paper used stochastic gradient descent (SGD), but
#   modern optimizers like Adam are now more common.
# """

# ############################################################################
# # 2. Imports and Setup
# ############################################################################

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

# ############################################################################
# # 3. Model Implementation
# ############################################################################

# """
# We define the LeNet-5 architecture as a Python class using `torch.nn.Module`.
# Note that the original MNIST dataset images are 28x28, but LeNet-5 was
# designed for 32x32 inputs. We will add padding to the input images to match
# this requirement.
# """

class LeNet5(nn.Module):
    """
    LeNet-5 architecture implementation in PyTorch.
    """
    def __init__(self):
        super(LeNet5, self).__init__()
        # The paper used `tanh` as the activation function.
        self.activation = nn.Tanh()

        # C1: Convolutional Layer
        # Input: 32x32x1 -> Output: 28x28x6
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1)

        # S2: Sub-sampling (Average Pooling)
        # Input: 28x28x6 -> Output: 14x14x6
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)

        # C3: Convolutional Layer
        # Input: 14x14x6 -> Output: 10x10x16
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1)

        # S4: Sub-sampling (Average Pooling)
        # Input: 10x10x16 -> Output: 5x5x16
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)

        # C5: Convolutional Layer (implemented as a Fully Connected layer)
        # Input: 5x5x16 -> Output: 120
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5, stride=1)

        # F6: Fully Connected Layer
        # Input: 120 -> Output: 84
        self.fc1 = nn.Linear(in_features=120, out_features=84)

        # Output Layer
        # Input: 84 -> Output: 10 (for 10 digits)
        self.fc2 = nn.Linear(in_features=84, out_features=10)

    def forward(self, x):
        # C1 -> S2
        x = self.conv1(x)
        x = self.activation(x)
        x = self.pool1(x)

        # C3 -> S4
        x = self.conv2(x)
        x = self.activation(x)
        x = self.pool2(x)

        # C5
        # This layer is a bit special. It's a 5x5 convolution that results in a
        # 1x1x120 output, which is then flattened.
        x = self.conv3(x)
        x = self.activation(x)

        # Flatten the output for the fully connected layers
        x = torch.flatten(x, 1)

        # F6
        x = self.fc1(x)
        x = self.activation(x)

        # Output
        x = self.fc2(x)
        # Note: Softmax is not applied here because CrossEntropyLoss in PyTorch
        # expects raw logits.
        return x

def print_model_summary(model):
    """
    Prints the model summary including layer names, output shapes, and params.
    """
    print("LeNet-5 Model Summary:")
    print("-" * 60)
    total_params = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            num_params = param.numel()
            total_params += num_params
            print(f"{name:<40} | Params: {num_params}")
    print("-" * 60)
    print(f"Total trainable parameters: {total_params}")

# Instantiate and print summary
lenet_model = LeNet5()
print_model_summary(lenet_model)


# ############################################################################
# # 4. MNIST Dataset Preparation
# ############################################################################

# """
# We load the MNIST dataset using `torchvision`. We apply two transformations:
# 1. `Pad(2)`: To resize the 28x28 images to 32x32, as required by LeNet-5.
# 2. `ToTensor()`: To convert the images to PyTorch tensors.
# 3. `Normalize()`: To scale pixel values to a standard range.
# """

# Define transformations
transform = transforms.Compose([
    transforms.Pad(2),  # Pad to 32x32
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)) # MNIST mean and std
])

# Load datasets
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# Create dataloaders
train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=1000, shuffle=False)

# """
# ### Visualize Sample Digits
# Let's look at a few examples from the dataset to see what we're working with.
# """

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
visualize_samples(train_loader)


# ############################################################################
# # 5. Training and Evaluation Loop
# ############################################################################

# """
# Here we define the training and evaluation functions. We'll use
# Cross-Entropy Loss as the criterion and Stochastic Gradient Descent (SGD)
# as the optimizer, which is close to the original paper's setup.
# """

def train_model(model, train_loader, test_loader, epochs=10, learning_rate=0.01):
    """
    The main training and evaluation loop.
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }

    for epoch in range(epochs):
        # --- Training ---
        model.train()
        running_loss, correct_train, total_train = 0.0, 0, 0
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
        train_acc = 100 * correct_train / total_train
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)

        # --- Validation ---
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


# ############################################################################
# # 6. Plotting Training Curves
# ############################################################################

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


# ############################################################################
# # 7. Ablation and Exploration
# ############################################################################

# """
# ### Experiment 1: Changing Activation Function to ReLU
#
# Modern CNNs use ReLU instead of Tanh or Sigmoid. Let's see what happens
# when we swap the activation function. ReLU helps with the vanishing gradient
# problem and often leads to faster convergence and better performance.
# """

class LeNet5ReLU(LeNet5):
    """
    LeNet-5 variant using ReLU activation.
    """
    def __init__(self):
        super(LeNet5ReLU, self).__init__()
        self.activation = nn.ReLU()

# Instantiate and train the ReLU variant
lenet_relu_model = LeNet5ReLU()
print("Starting training for LeNet-5 with ReLU activation...")
lenet_relu_history = train_model(lenet_relu_model, train_loader, test_loader, epochs=15)
plot_curves(lenet_relu_history, "LeNet-5 (ReLU) Training Curves")

# """
# ### Commentary on ReLU vs. Tanh
#
# You will likely observe that the ReLU model converges faster and may achieve
# slightly higher accuracy. This is because ReLU does not saturate for positive
# inputs, allowing gradients to flow more freely during backpropagation, which
# is a major advantage in deeper networks.
#
# ### Experiment 2: Adding Dropout
#
# Dropout is a regularization technique where random neurons are "dropped out"
# (ignored) during training. This prevents co-adaptation of neurons and helps
# prevent overfitting. Let's add it to our ReLU model.
# """

class LeNet5ReLUWithDropout(nn.Module):
    def __init__(self):
        super(LeNet5ReLUWithDropout, self).__init__()
        self.cnn_layers = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5), nn.ReLU(), nn.AvgPool2d(2, 2),
            nn.Conv2d(6, 16, kernel_size=5), nn.ReLU(), nn.AvgPool2d(2, 2),
            nn.Conv2d(16, 120, kernel_size=5), nn.ReLU()
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(120, 84), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(84, 10)
        )

    def forward(self, x):
        x = self.cnn_layers(x)
        x = torch.flatten(x, 1)
        x = self.fc_layers(x)
        return x

# Instantiate and train the Dropout variant
lenet_dropout_model = LeNet5ReLUWithDropout()
print("Starting training for LeNet-5 (ReLU with Dropout)...")
lenet_dropout_history = train_model(lenet_dropout_model, train_loader, test_loader, epochs=15)
plot_curves(lenet_dropout_history, "LeNet-5 (ReLU + Dropout) Training Curves")

# """
# ### Commentary on Dropout
#
# With dropout, you might see that the training accuracy is closer to the
# validation accuracy, indicating less overfitting. The final validation
# accuracy might also be higher, as the model generalizes better to unseen data.
# This demonstrates how crucial regularization is for modern deep learning.
# """

# ############################################################################
# # 8. Final Summary
# ############################################################################

# """
# Let's summarize the key results and limitations of the original LeNet-5.
# """

final_accuracy = lenet_history['val_acc'][-1]
final_accuracy = lenet_history['val_acc'][-1]

print("\n" + "="*60)
print("LeNet-5 Final Summary")
print("="*60)
print(f"- Total Trainable Parameters: {sum(p.numel() for p in lenet_model.parameters() if p.requires_grad)}")
print(f"- Final Test Accuracy (Tanh model): {final_accuracy:.2f}%")
print("\n### Limitations of LeNet-5 and Later Improvements:")
print("""
1.  **Small Scale:** LeNet-5 is a small network designed for a simple, low-resolution
    dataset. It does not scale well to complex, high-resolution color images
    like those in ImageNet.

2.  **Activation Functions:** As seen, `tanh`/`sigmoid` activations are prone to
    vanishing gradients, making it hard to train very deep networks. `ReLU` was a
    major breakthrough introduced later.

3.  **Pooling Method:** Average pooling smooths out features. Max pooling,
    popularized by later networks, is often better at preserving the most
    important structural information.

4.  **Lack of Regularization:** Without techniques like Dropout or Batch Norm,
    LeNet-5 is prone to overfitting on more complex datasets.

### The Path Forward: AlexNet, VGG, and Beyond

-   **AlexNet (2012):** The true successor to LeNet-5. It was much deeper, used
    `ReLU` activations, implemented `Dropout`, and won the ImageNet competition,
    ushering in the modern era of deep learning.

-   **VGGNet (2014):** Showed that a very simple and deep architecture using only
    3x3 convolutions could achieve state-of-the-art results.

-   **GoogLeNet / Inception (2014):** Introduced the "Inception module," which
    ran parallel convolutions of different sizes to capture features at
    multiple scales.

-   **ResNet (2015):** Introduced "residual connections" or skip connections,
    which finally made it possible to effectively train networks that were
    hundreds or even thousands of layers deep.
""")
print("="*60)

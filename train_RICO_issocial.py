# -*- coding: utf-8 -*-
"""train_RICO_issocial.py

Script for training and evaluating a ResNet model on the RICO dataset to predict 'Is_Social'.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import roc_auc_score, accuracy_score
from torch.utils.data import TensorDataset, DataLoader, random_split
from torchvision import models, transforms

# Define paths based on the bash script setup
ui_details_path = './ui_details_updated.csv'  # CSV file containing UI details and labels
images_dir = './combined/combined/'   # Directory containing the UI images

# Function to print GPU memory usage for debugging
def print_gpu_memory():
    print(f"Allocated: {torch.cuda.memory_allocated(device) / (1024 ** 3):.2f} GB")
    print(f"Cached: {torch.cuda.memory_reserved(device) / (1024 ** 3):.2f} GB")

# Load the UI details
ui_details_df = pd.read_csv(ui_details_path)
print(f"Data loaded. Number of UI samples: {len(ui_details_df)}")

# Define the image processing transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to the size expected by your model
    transforms.ToTensor(),  # Convert the image to a PyTorch tensor
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Normalization for pretrained models
])

# Load and preprocess images and labels
print("Loading and processing images...")
features = []
labels = []

for index, row in ui_details_df.iterrows():
    ui_number = row['UI Number']
    is_social = row['Is_Social']
    img_path = os.path.join(images_dir, f"{ui_number}.jpg")

    if os.path.exists(img_path):
        image = Image.open(img_path).convert("RGB")  # Convert to RGB
        image_tensor = transform(image)
        features.append(image_tensor)
        labels.append(torch.tensor(is_social, dtype=torch.float32))
    else:
        print(f"Image {ui_number}.jpg not found.")

print(f"Loaded {len(features)} images successfully.")

# Create feature and label tensors
features_tensor = torch.stack(features) if features else torch.Tensor()
labels_tensor = torch.stack(labels) if labels else torch.Tensor()

# Create a dataset and dataloaders
dataset = TensorDataset(features_tensor, labels_tensor)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_ds, val_ds = random_split(dataset, [train_size, val_size])

# Reduce batch size to lower memory usage
train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=16, shuffle=False)

# Define the model (Pretrained ResNet50)
print("Loading the ResNet50 model...")
model = models.resnet50(pretrained=True)

# Modify the classifier for binary classification
num_ftrs = model.fc.in_features
model.fc = nn.Sequential(
    nn.Linear(num_ftrs, 1),
    nn.Sigmoid()
)

# Move the model to the GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model = nn.DataParallel(model, device_ids=[0, 1, 2, 3])

# Define loss and optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Enable mixed precision training
scaler = torch.cuda.amp.GradScaler()

# Training and evaluation loop
print("Starting training...")
num_epochs = 10
best_auc = 0.0
best_model_state = None

for epoch in range(num_epochs):
    # Training phase
    model.train()
    running_loss = 0.0
    for batch in train_loader:
        data, target = batch
        data = data.to(device)
        target = target.to(device).float().view(-1)  # Ensure target is a 1D float tensor

        optimizer.zero_grad()

        # Use mixed precision for forward and backward pass
        with torch.cuda.amp.autocast():
            output = model(data)
            loss = criterion(output.view(-1), target)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()

    print(f"Epoch {epoch+1}/{num_epochs} - Training loss: {running_loss / len(train_loader):.4f}")
    print_gpu_memory()  # Print GPU memory usage after each epoch

    # Evaluation phase
    model.eval()
    val_targets = []
    val_outputs = []
    with torch.no_grad():
        for batch in val_loader:
            data, target = batch
            data = data.to(device)
            target = target.to(device).float().view(-1)
            output = model(data)
            val_outputs.extend(output.view(-1).cpu().numpy())
            val_targets.extend(target.cpu().numpy())

    # Calculate AUC for the current epoch
    auc = roc_auc_score(val_targets, val_outputs)
    print(f"Epoch {epoch+1}/{num_epochs} - Validation AUC: {auc:.4f}")

    # Save best model based on AUC
    if auc > best_auc:
        best_auc = auc
        best_model_state = model.state_dict()
        print(f"Best model updated with AUC: {best_auc:.4f}")

# Save the best model state to a file
if best_model_state is not None:
    torch.save(best_model_state, './best_resnet_finetune.pth')
    print(f"Best model saved with AUC: {best_auc:.4f}")

print("Training completed successfully.")

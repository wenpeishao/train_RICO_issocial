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
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score
from torch.utils.data import TensorDataset, DataLoader, random_split
from torchvision import models, transforms

# SSL fix for certificate issues
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# Define paths based on the bash script setup
ui_details_path = './ui_details_updated.csv'  # CSV file containing UI details and labels
images_dir = './combined/combined/'   # Directory containing the UI images

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

train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)  # Adjusted batch size to reduce memory usage
val_loader = DataLoader(val_ds, batch_size=16, shuffle=False)

# Define the model (Pretrained ResNet50 using weights argument)
print("Loading the ResNet50 model...")
from torchvision.models import ResNet50_Weights
model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)

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

# Training and evaluation loop
print("Starting training...")
num_epochs = 10
best_auc = 0.0
best_auprc = 0.0
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
        output = model(data)
        loss = criterion(output.view(-1), target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch+1}/{num_epochs} - Training loss: {running_loss / len(train_loader):.4f}")

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

    # Calculate AUC and AUPRC for the current epoch
    auc = roc_auc_score(val_targets, val_outputs)
    auprc = average_precision_score(val_targets, val_outputs)
    print(f"Epoch {epoch+1}/{num_epochs} - Validation AUC: {auc:.4f} - Validation AUPRC: {auprc:.4f}")

    # Save best model based on AUC and AUPRC
    if auc > best_auc or auprc > best_auprc:
        best_auc = max(auc, best_auc)
        best_auprc = max(auprc, best_auprc)
        best_model_state = model.state_dict()
        print(f"Best model updated with AUC: {best_auc:.4f} and AUPRC: {best_auprc:.4f}")

# Save the best model state to a file
if best_model_state is not None:
    torch.save(best_model_state, './best_resnet_finetune.pth')
    print(f"Best model saved with AUC: {best_auc:.4f} and AUPRC: {best_auprc:.4f}")

print("Training completed successfully.")

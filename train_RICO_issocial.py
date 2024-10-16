# -*- coding: utf-8 -*-
"""train_RICO_issocial.py

Revised Script for training and evaluating a ResNet model on the RICO dataset to predict 'Is_Social' with:
- Stratified sampling for train/test split
- Upsampling the minority class in the training set
- Saving predicted probabilities and generating a calibration plot
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score
from sklearn.calibration import calibration_curve
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from torch.utils.data import TensorDataset, DataLoader
from torchvision import models, transforms
import matplotlib.pyplot as plt

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

# Stratified train/test split
X_train, X_test, y_train, y_test = train_test_split(features_tensor, labels_tensor, test_size=0.2, stratify=labels_tensor)

# Upsampling the minority class in the training set
train_data = TensorDataset(X_train, y_train)
train_labels = pd.Series(y_train.cpu().numpy())

# Separate minority and majority classes
minority_class = train_labels[train_labels == 1]
majority_class = train_labels[train_labels == 0]

# Upsample minority class
minority_upsampled = resample(minority_class,
                              replace=True,
                              n_samples=len(majority_class),
                              random_state=42)

# Concatenate back the upsampled minority class and majority class
X_train_upsampled = torch.cat((X_train[minority_upsampled.index], X_train[majority_class.index]), dim=0)
y_train_upsampled = torch.cat((y_train[minority_upsampled.index], y_train[majority_class.index]), dim=0)

# Create the dataset and dataloaders
train_ds = TensorDataset(X_train_upsampled, y_train_upsampled)
val_ds = TensorDataset(X_test, y_test)

train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
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
all_pred_probs = []  # To store predicted probabilities for calibration plot

for epoch in range(num_epochs):
    # Training phase
    model.train()
    running_loss = 0.0
    for batch in train_loader:
        data, target = batch
        data = data.to(device)
        target = target.to(device).float().view(-1)

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
            all_pred_probs.extend(output.view(-1).cpu().numpy())  # Store predictions

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

# Calibration plot
print("Generating calibration plot...")
fraction_of_positives, mean_predicted_value = calibration_curve(val_targets, all_pred_probs, n_bins=10)

plt.plot(mean_predicted_value, fraction_of_positives, "s-", label="Model")
plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
plt.xlabel("Mean predicted value")
plt.ylabel("Fraction of positives")
plt.title("Calibration plot")
plt.legend()
plt.savefig('calibration_plot.png')  # Save the plot as PNG
plt.show()

print("Training completed successfully.")

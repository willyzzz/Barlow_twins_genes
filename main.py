import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from barlow_config import config
from dataset_input import Barlow_dataloader
from model_structure import Encoder, Projector, BarlowTwinsLoss, MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

import pandas as pd
import numpy as np

def train_barlow_twins(encoder, projector, dataloader, criterion, optimizer, device):
    encoder.train()
    projector.train()
    total_loss = 0
    for bulk, dist1, dist2, stage_gt in dataloader:
        bulk, dist1, dist2, stage_gt = bulk.to(device), dist1.to(device), dist2.to(device), stage_gt.to(device)

        z1 = encoder(dist1)
        z2 = encoder(dist2)

        p1 = projector(z1)
        p2 = projector(z2)

        loss = criterion(p1, p2)
        
        total_loss += loss.item()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    return total_loss / len(dataloader)

def train_mlp(model, X_train, y_train, criterion, optimizer, device, num_epochs=100, batch_size=32):
    model.train()
    for epoch in range(num_epochs):
        for i in range(0, len(X_train), batch_size):
            batch_X = X_train[i:i+batch_size].to(device)
            batch_y = y_train[i:i+batch_size].to(device)
            
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y - 1)  # Subtract 1 because labels start from 1
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

def evaluate_mlp(model, X_test, y_test, device):
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test.to(device)).argmax(dim=1).cpu().numpy() + 1
    
    print("Confusion Matrix:")
    print(confusion_matrix(y_test.cpu().numpy(), y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test.cpu().numpy(), y_pred, zero_division=0))

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Stage 1: Loading data...")
    # Load data
    dataloader, bulk_tensor, stages_tensor = Barlow_dataloader(config['sc_path'], config['bulk_path'],
                                                config['patient_path'], batchsize=config['batchsize'])
    input_dim = bulk_tensor.shape[1]
    output_dim = config['output_dim']
    proj_dim = config['project_dim']

    print("Stage 2: Initializing models...")
    # Initialize models
    encoder = Encoder(input_dim, output_dim).to(device)
    projector = Projector(output_dim, proj_dim).to(device)
    criterion = BarlowTwinsLoss(config['loss_lambda_param'])
    optimizer = optim.Adam(list(encoder.parameters()) + list(projector.parameters()), lr=config['learning_rate'])

    print("Stage 3: Training Barlow Twins...")
    # Train Barlow Twins
    losses = []
    for epoch in range(config['num_epochs']):
        epoch_loss = train_barlow_twins(encoder, projector, dataloader, criterion, optimizer, device)
        losses.append(epoch_loss)
        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch + 1} completed. Loss: {epoch_loss:.4f}")

    # Plot loss curve
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, config['num_epochs'] + 1), losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Barlow Twins Loss Curve')
    plt.legend()
    plt.grid(True)
    plt.show()

    print("Stage 4: Preparing data for classification...")
    # Prepare data
    X = bulk_tensor.to(device)
    y = stages_tensor.long().to(device)

    print("Stage 5: Generating Barlow Twins embeddings...")
    # Generate Barlow Twins embedding
    encoder.eval()
    with torch.no_grad():
        X_embedded = encoder(X)

    print("Stage 6: Preparing MLP classifier...")
    # Prepare MLP classifier
    hidden_dim = 64
    num_classes = len(torch.unique(y))
    
    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_embedded_train, X_embedded_test, _, _ = train_test_split(X_embedded, y, test_size=0.2, random_state=42)
    
    # Standardize
    scaler = StandardScaler()
    X_train_np = scaler.fit_transform(X_train.cpu().numpy())
    X_test_np = scaler.transform(X_test.cpu().numpy())
    X_train = torch.FloatTensor(X_train_np).to(device)
    X_test = torch.FloatTensor(X_test_np).to(device)
    X_embedded_train_np = scaler.fit_transform(X_embedded_train.cpu().numpy())
    X_embedded_test_np = scaler.transform(X_embedded_test.cpu().numpy())
    X_embedded_train = torch.FloatTensor(X_embedded_train_np).to(device)
    X_embedded_test = torch.FloatTensor(X_embedded_test_np).to(device)

    print("Stage 7: Training and evaluating MLP on original bulk data...")
    # MLP classification on original bulk data
    mlp_bulk = MLPClassifier(input_dim, hidden_dim, num_classes).to(device)
    criterion_mlp = nn.CrossEntropyLoss()
    optimizer_mlp = optim.Adam(mlp_bulk.parameters(), lr=0.001)

    print("Training MLP on original bulk data...")
    train_mlp(mlp_bulk, X_train, y_train, criterion_mlp, optimizer_mlp, device)
    print("Evaluating MLP on original bulk data:")
    evaluate_mlp(mlp_bulk, X_test, y_test, device)

    print("Stage 8: Training and evaluating MLP on Barlow Twins embeddings...")
    # MLP classification on Barlow Twins embedding
    mlp_embedded = MLPClassifier(output_dim, hidden_dim, num_classes).to(device)
    optimizer_mlp_embedded = optim.Adam(mlp_embedded.parameters(), lr=0.001)

    print("Training MLP on Barlow Twins embedding...")
    train_mlp(mlp_embedded, X_embedded_train, y_train, criterion_mlp, optimizer_mlp_embedded, device)
    print("Evaluating MLP on Barlow Twins embedding:")
    evaluate_mlp(mlp_embedded, X_embedded_test, y_test, device)

    print("All stages completed.")

if __name__ == "__main__":
    main()

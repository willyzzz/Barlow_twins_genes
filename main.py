import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import f1_score, precision_recall_curve, average_precision_score, confusion_matrix, classification_report
from sklearn.preprocessing import label_binarize
from collections import Counter
from barlow_config import config
from dataset_input import Barlow_dataloader, set_seed
from model_structure import Encoder, Projector, BarlowTwinsLoss, MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.pyplot as plt
import logging
from datetime import datetime
import os

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

        p1_norm = (p1 - p1.mean(0)) / p1.std(0)
        p2_norm = (p2 - p2.mean(0)) / p2.std(0)

        loss = criterion(p1_norm, p2_norm)
        
        total_loss += loss.item()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    return total_loss / len(dataloader)

def train_mlp(model, X_train, y_train, criterion, optimizer, device, num_epochs=100, batch_size=32):
    model.to(device)
    model.train()
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for i in range(0, len(X_train), batch_size):
            batch_X = X_train[i:i + batch_size]
            batch_y = y_train[i:i + batch_size]

            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * batch_X.size(0)

        if (epoch + 1) % 10 == 0:  
            avg_loss = epoch_loss / len(X_train)
            logging.info(f'Epoch [{epoch + 1}/{num_epochs}], Average Loss: {avg_loss:.4f}')

def evaluate_mlp(model, X_test, y_test, device):
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test.to(device)).argmax(dim=1).cpu().numpy()
    cm = confusion_matrix(y_test.cpu().numpy(), y_pred)
    cr = classification_report(y_test.cpu().numpy(), y_pred, zero_division=0)
    
    logging.info("Confusion Matrix:")
    logging.info(f"\n{cm}")
    logging.info("\nClassification Report:")
    logging.info(f"\n{cr}")

def setup_logging(testing_dataset_name):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join("log", testing_dataset_name)
    os.makedirs(log_dir, exist_ok=True)  # 确保目录存在
    log_file = os.path.join(log_dir, f"run_{timestamp}.log")
    
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        handlers=[logging.FileHandler(log_file),
                                  logging.StreamHandler()])
    return timestamp, log_file

def log_config_and_model(config, models):
    logging.info("Configuration Parameters:")
    for key, value in config.items():
        logging.info(f"{key}: {value}")
    
    logging.info("Model Architectures:")
    for model_name, model in models.items():
        logging.info(f"{model_name}: {model}")

def main():
    from barlow_config import config
    from model_structure import Encoder, Projector, MLPClassifier

    testing_dataset_name = config['testing_dataset_name']
    timestamp, log_file = setup_logging(testing_dataset_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    set_seed(42)

    logging.info("Stage 1: Loading data...")
    dataloader, bulk_tensor, stages_tensor, num_classes = Barlow_dataloader(config['sc_path'], config['bulk_path'],
                                                config['patient_path'], batchsize=config['batchsize'])
    input_dim = bulk_tensor.shape[1]
    output_dim = config['output_dim']
    proj_dim = config['project_dim']

    logging.info(f"Number of classes: {num_classes}")

    logging.info("Stage 2: Initializing models...")
    encoder = Encoder(input_dim, output_dim).to(device)
    projector = Projector(output_dim, proj_dim).to(device)
    mlp_bulk = MLPClassifier(input_dim, config['mlp_hidden_dim'], num_classes).to(device)
    mlp_embedded = MLPClassifier(output_dim, config['mlp_hidden_dim'], num_classes).to(device)
    criterion = BarlowTwinsLoss(config['loss_lambda_param'])
    optimizer = optim.Adam(list(encoder.parameters()) + list(projector.parameters()), lr=config['learning_rate'])

    # Log config and models
    models = {
        "Encoder": encoder,
        "Projector": projector,
        "MLP Bulk": mlp_bulk,
        "MLP Embedded": mlp_embedded
    }
    log_config_and_model(config, models)

    logging.info("Stage 3: Training Barlow Twins...")
    losses = []
    for epoch in range(config['num_epochs']):
        epoch_loss = train_barlow_twins(encoder, projector, dataloader, criterion, optimizer, device)
        losses.append(epoch_loss)
        if (epoch + 1) % 100 == 0:
            logging.info(f"Epoch {epoch + 1} completed. Loss: {epoch_loss:.4f}")

    # Plot loss curve
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, config['num_epochs'] + 1), losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Barlow Twins Loss Curve')
    plt.legend()
    plt.grid(True)
    plt.show()

    logging.info("Stage 4: Preparing data for classification...")
    X = bulk_tensor.to(device)
    y = stages_tensor.long().to(device)

    logging.info("Stage 5: Generating Barlow Twins embeddings...")
    encoder.eval()
    with torch.no_grad():
        X_embedded = encoder(X)
    
    # Save embeddings and bulk tensor
    result_dir = os.path.join("result", f"{testing_dataset_name}_{timestamp}")
    os.makedirs(result_dir, exist_ok=True)  
    bulk_file = os.path.join(result_dir, "bulk_tensor.pt")
    embedding_file = os.path.join(result_dir, "embedding.pt")
    torch.save(bulk_tensor, bulk_file)
    torch.save(X_embedded, embedding_file)
    logging.info(f"Bulk tensor saved to {bulk_file}")
    logging.info(f"Embeddings saved to {embedding_file}")

    logging.info("Stage 6: Preparing MLP classifier...")
    criterion_mlp = nn.CrossEntropyLoss()  
    optimizer_mlp = optim.Adam(mlp_bulk.parameters(), lr=config['learning_rate'])
    optimizer_mlp_embedded = optim.Adam(mlp_embedded.parameters(), lr=config['learning_rate'])
    
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

    logging.info("Stage 7: Training and evaluating MLP on original bulk data...")
    logging.info("Training MLP on original bulk data:")
    train_mlp(mlp_bulk, X_train, y_train, criterion_mlp, optimizer_mlp, device)
    logging.info("Evaluating MLP on original bulk data:")
    evaluate_mlp(mlp_bulk, X_test, y_test, device)

    logging.info("Stage 8: Training and evaluating MLP on Barlow Twins embeddings...")
    train_mlp(mlp_embedded, X_embedded_train, y_train, criterion_mlp, optimizer_mlp_embedded, device)
    logging.info("Evaluating MLP on Barlow Twins embeddings:")
    evaluate_mlp(mlp_embedded, X_embedded_test, y_test, device)

    logging.info("All stages completed.")

if __name__ == "__main__":
    main()

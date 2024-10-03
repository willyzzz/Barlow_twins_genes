import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score, precision_recall_curve, average_precision_score, confusion_matrix, classification_report
from barlow_config import config
from training import *
from dataset_input import Barlow_dataloader, set_seed
from model_structure import *
from sklearn.model_selection import train_test_split
import logging
from datetime import datetime
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

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
    dataloader, bulk_tensor, stages_tensor, survival_df, num_classes, bulk_df = Barlow_dataloader(config['sc_path'], config['bulk_path'],
                                                config['stage_path'], config['survival_path'], batchsize=config['batchsize'])
    input_dim = bulk_tensor.shape[1]
    output_dim = config['output_dim']
    hidden_dim = config['hidden_dim']
    proj_dim = config['project_dim']

    logging.info(f"Number of classes: {num_classes}")

    logging.info("Stage 2: Initializing models...")
    encoder = GeneResNetEncoder(input_dim, output_dim).to(device)
    projector = Projector(output_dim, hidden_dim, proj_dim).to(device)
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
    bulk_tensor_df = pd.DataFrame(bulk_tensor.cpu().numpy(), index = bulk_df.index)
    embedding_df = pd.DataFrame(X_embedded.cpu().numpy(), index= bulk_df.index)
    bulk_file = os.path.join(result_dir, "bulk_tensor.pt")
    embedding_file = os.path.join(result_dir, "embedding.pt")
    torch.save(bulk_tensor_df, bulk_file)
    torch.save(embedding_df, embedding_file)
    logging.info(f"Bulk tensor saved to {bulk_file}")
    logging.info(f"Embeddings saved to {embedding_file}")

    if config['eval_metric'] == 'stage':
        logging.info("Stage 6: Evaluate stages")
        logging.info("Preparing MLP classifier...")
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

    # elif config['eval_metric'] == 'survival':
    #     logging.info("Stage 6: Evaluate survival")
    #     logging.info("Preparing Cox model")
    #
    #     logging.info("Training Cox model on bulk data...")
    #     survival_analysis(survival_df, bulk_tensor)
    #
    #     logging.info("Training Cox model on Barlow Twins embeddings...")
    #     survival_analysis(survival_df, X_embedded)

    logging.info("All stages completed.")

if __name__ == "__main__":
    main()

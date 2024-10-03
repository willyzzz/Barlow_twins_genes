import torch
from sklearn.metrics import f1_score, precision_recall_curve, average_precision_score, confusion_matrix, classification_report
from barlow_config import config
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.util import Surv
from sksurv.nonparametric import kaplan_meier_estimator
from sksurv.metrics import concordance_index_censored

def train_barlow_twins(encoder, projector, dataloader, criterion, optimizer, device):
    encoder.train()
    projector.train()
    total_loss = 0
    for bulk, dist1, dist2 in dataloader:
        bulk, dist1, dist2 = bulk.to(device), dist1.to(device), dist2.to(device)

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


def train_mlp(model, X_train, y_train, criterion, optimizer, device, num_epochs=config['mlp_epochs'], batch_size=32):
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


def survival_analysis(surv, bulk_tensor):
    # 1. Data preprocessing: Ensure that the indices of surv and bulk are aligned to find common samples
    # 2. Convert survival data into the format required by scikit-survival

    bulk = bulk_tensor.cpu().numpy()
    y = Surv.from_dataframe("status", "time", surv)

    # Convert status column to boolean type
    surv['status'] = surv['status'].astype(bool)

    # 3. Standardize the gene expression data
    scaler = StandardScaler()
    X = scaler.fit_transform(bulk)

    # 4. Train the CoxPH model
    cox_model = CoxPHSurvivalAnalysis()
    cox_model.fit(X, y)

    # Step 4: Print model coefficients (risk ratio for each gene)
    coef_series = pd.Series(cox_model.coef_, index=bulk.columns)
    print("CoxPH Model Coefficients:\n", coef_series.sort_values(ascending=False))

    # Calculate C-index
    time = surv['time'].values
    event = surv['status'].values  # Use boolean type event status

    # Use the Cox model to predict risk scores
    predicted_risks = cox_model.predict(X)

    # Compute the C-index
    c_index = concordance_index_censored(event, time, predicted_risks)
    print(f"C-Index: {c_index[0]}")

    # Step 5: Use the model to predict and plot the survival function for a random sample
    sample = pd.DataFrame(X[0].reshape(1, -1), columns=bulk.columns)  # Convert the selected sample into a DataFrame
    pred_surv_func = cox_model.predict_survival_function(sample)

    # Estimate the survival probability
    time_points = np.linspace(0, surv['time'].max(), 100)
    plt.step(time_points, pred_surv_func[0](time_points), where="post")
    plt.ylim(0, 1)
    plt.ylabel(r"Estimated probability of survival $\hat{S}(t)$")
    plt.xlabel("Time")
    plt.title("Survival Function for Sample 1")
    plt.show()

    # Split samples into high-risk and low-risk groups
    risk_threshold = np.median(predicted_risks)  # Use the median risk score as the threshold
    high_risk = surv[predicted_risks > risk_threshold]
    low_risk = surv[predicted_risks <= risk_threshold]

    # Compute Kaplan-Meier survival curves for each group
    km_high = kaplan_meier_estimator(high_risk['status'], high_risk['time'])
    km_low = kaplan_meier_estimator(low_risk['status'], low_risk['time'])

    # Plot Kaplan-Meier survival curves
    plt.step(km_high[0], km_high[1], where="post", label="High Risk")
    plt.step(km_low[0], km_low[1], where="post", label="Low Risk")
    plt.ylim(0, 1)
    plt.ylabel(r"Estimated probability of survival $\hat{S}(t)$")
    plt.xlabel("Time")
    plt.title("Kaplan-Meier Survival Curves")
    plt.legend()
    plt.show()


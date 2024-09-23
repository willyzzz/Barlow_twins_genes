#!/usr/bin/env python
# coding: utf-8

import anndata as ad
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import random
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.feature_selection import VarianceThreshold
import matplotlib.pyplot as plt
import seaborn as sns
from barlow_config import *

def set_seed(seed_value):
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    np.random.seed(seed_value)
    random.seed(seed_value)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


### sampling single cell data as distortion input


def advanced_distort_gen(bulk_data, sc_df, lambda_noise, sample_size=1000):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} for distortion generation")

    n_repeats = bulk_data.shape[0]

    bulk_tensor = torch.tensor(np.array(bulk_data), dtype=torch.float32).to(device)
    sc_tensor = torch.tensor(np.array(sc_df), dtype=torch.float32).to(device)

    distortions = []
    
    for i in range(n_repeats):
        indices = torch.randint(0, sc_tensor.shape[0], (sample_size,), device=device)
        sampled_cells = sc_tensor[indices]
        mean_values = torch.mean(sampled_cells, dim=0)
        distorted = (1 - lambda_noise) * bulk_tensor[i] + lambda_noise * mean_values
        distortions.append(distorted)
    
    distortions_tensor = torch.stack(distortions)
    return distortions_tensor.cpu().numpy(), bulk_tensor.cpu().numpy()

def plot_bulk_vs_distortion_distribution(bulk_data, distortion1, distortion2, feature_indices=None, bins=50):
    """
    Visualize the distribution of bulk_data and two sets of distortions by plotting the selected features' distributions.

    Parameters:
    - bulk_data: Original bulk expression data.
    - distortion1: First set of distorted data.
    - distortion2: Second set of distorted data.
    - feature_indices: List of indices of features to plot. If None, plot the distribution of all features' mean.
    - bins: Number of bins to use in the histogram.
    """
    if feature_indices is None:
        # If no specific features are selected, plot the mean of all features
        bulk_mean = bulk_data.mean(axis=0)
        distortion1_mean = distortion1.mean(axis=0)
        distortion2_mean = distortion2.mean(axis=0)

        plt.figure(figsize=(10, 6))
        sns.histplot(bulk_mean, bins=bins, kde=True, label="Bulk Data (mean)", color="blue", stat="density")
        sns.histplot(distortion1_mean, bins=bins, kde=True, label="Distortion 1 (mean)", color="red", stat="density")
        sns.histplot(distortion2_mean, bins=bins, kde=True, label="Distortion 2 (mean)", color="green", stat="density")
    else:
        plt.figure(figsize=(10, 6))
        for feature_idx in feature_indices:
            sns.histplot(bulk_data[:, feature_idx], bins=bins, kde=True, label=f"Bulk Feature {feature_idx}", color="blue", stat="density")
            sns.histplot(distortion1[:, feature_idx], bins=bins, kde=True, label=f"Distortion 1 Feature {feature_idx}", color="red", stat="density")
            sns.histplot(distortion2[:, feature_idx], bins=bins, kde=True, label=f"Distortion 2 Feature {feature_idx}", color="green", stat="density")

    plt.title("Distribution Comparison: Bulk Data vs Distortion 1 vs Distortion 2")
    plt.xlabel("Expression Level")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True)
    plt.show()

class TensorDataset(Dataset):
    def __init__(self, tensor1, tensor2, tensor3, tensor4):
        assert tensor1.size(0) == tensor2.size(0) == tensor3.size(0) == tensor4.size(0)
        self.tensor1 = tensor1
        self.tensor2 = tensor2
        self.tensor3 = tensor3
        self.tensor4 = tensor4

    def __len__(self):
        return self.tensor1.size(0)  # Number of samples

    def __getitem__(self, idx):
        return self.tensor1[idx], self.tensor2[idx], self.tensor3[idx], self.tensor4

def pat_preprosses(patient_df):
    stage_list = ['American Joint Committee on Cancer Tumor Stage Code',
                  'Neoplasm Disease Lymph Node Stage American Joint Committee on Cancer Code',
                  'American Joint Committee on Cancer Metastasis Stage Code',
                  'Neoplasm Disease Stage American Joint Committee on Cancer Code', ]

    stage_info = patient_df[stage_list]
    overall_stages = stage_info.iloc[4:, 3:4]
    overall_stages.rename(columns={overall_stages.columns[0]: 'overall_stage'}, inplace=True)
    drop_indices = overall_stages[overall_stages['overall_stage'].isin(['[Not Available]', '[Discrepancy]'])].index
    overall_stages = overall_stages.drop(index=drop_indices)

    stage_mapping = {
        'Stage I': 0,
        'Stage IA': 0,
        'Stage IB': 0,
        'Stage II': 1,
        'Stage IIA': 1,
        'Stage IIB': 1,
        'Stage III': 2,
        'Stage IIIA': 2,
        'Stage IIIB': 2,
        'Stage IIIC': 2,
        'Stage IV': 3,
        'Stage X': 4
    }

    overall_stages['overall_stage_simplified'] = overall_stages['overall_stage'].map(stage_mapping)

    return overall_stages


def Barlow_dataloader(sc_path, bulk_path, patient_path, batchsize):
    set_seed(42)

    sc_data = ad.read_h5ad(sc_path)
    cell_names = sc_data.obs_names
    gene_names = sc_data.var['feature_name']
    sc_data_exp = sc_data.X.toarray()
    sc_df = pd.DataFrame(sc_data_exp, index=cell_names, columns=gene_names)
    print(f" Single-cell data loaded. Shape: {sc_df.shape}")

    if config['testing_dataset_name'].startswith('TCGA'):
        bulk = pd.read_csv(bulk_path, sep='\t', index_col=0)
        bulk.index = [i[:-3] for i in bulk.index]
        bulk = bulk.apply(lambda row: row.fillna(row.mean()), axis=1)

        patient_df = pd.read_csv(patient_path, sep='\t', index_col=0)
        overall_stages = pat_preprosses(patient_df)
        num_classes = len(overall_stages['overall_stage_simplified'].unique())

    elif config['testing_dataset_name'].startswith('Metabric'):
        bulk = pd.read_csv(bulk_path, index_col=0, sep='\t')
        bulk = bulk.drop(bulk.columns[0], axis=1).T
        bulk = bulk.apply(lambda row: row.fillna(row.mean()), axis=1)

        patient_df = pd.read_csv(patient_path, sep='\t', index_col=0)
        patient_df = patient_df.set_index(patient_df.columns[0])
        num_classes = len(patient_df['Tumor Stage'].unique())

    print(f"Bulk data loaded. Shape: {bulk.shape}")
    print(f"Number of unique classes: {num_classes}")

    common_genes = sc_df.columns.intersection(bulk.columns)
    sc_df = sc_df.loc[:, ~sc_df.columns.duplicated()]
    bulk = bulk.loc[:, ~bulk.columns.duplicated()]
    sc_df = sc_df[common_genes]
    bulk = bulk[common_genes]

    common_index = bulk.index.intersection(patient_df.index)
    bulk = bulk.loc[common_index]
    patient_df = patient_df.loc[common_index]
    
    bulk = bulk.sort_index()
    patient_df = patient_df.sort_index()
    
    print(f"After filtering: Bulk data shape: {bulk.shape}, Number of patients: {len(patient_df)}")
    
    if not (bulk.index == patient_df.index).all():
        raise ValueError("Bulk data and patient data indices do not match!")

    distortion1, bulk_tensor = advanced_distort_gen(bulk, sc_df=sc_df, lambda_noise=0.1, sample_size=config['sample_size'])
    distortion2, _ = advanced_distort_gen(bulk, sc_df=sc_df, lambda_noise=0.1, sample_size=config['sample_size'])
    print("Distortions generated.")

    # Apply variance threshold
    variance_threshold = config['variance_threshold']
    print('Cutting variance...')
    bulk_var = bulk.var(axis=0).values
    var_cutoff = np.sort(bulk_var)[-int(bulk.shape[1] * variance_threshold)]
    bulk = bulk.loc[:, bulk_var > var_cutoff]

    distortion1_var = np.var(distortion1, axis=0)
    var_cutoff = np.sort(distortion1_var)[-int(distortion1.shape[1] * variance_threshold)]
    distortion1 = distortion1[:, distortion1_var > var_cutoff]

    distortion2_var = np.var(distortion2, axis=0)
    var_cutoff = np.sort(distortion2_var)[-int(distortion2.shape[1] * variance_threshold)]
    distortion2 = distortion2[:, distortion2_var > var_cutoff]

    if config['minmax_icon']:
        scaler = MinMaxScaler()
        bulk = scaler.fit_transform(bulk.T).T
        distortion1 = scaler.fit_transform(distortion1.T).T
        distortion2 = scaler.fit_transform(distortion2.T).T
    elif config['standard_icon']:
        scaler = StandardScaler()
        bulk = scaler.fit_transform(bulk.T).T
        distortion1 = scaler.fit_transform(distortion1.T).T
        distortion2 = scaler.fit_transform(distortion2.T).T

    plot_bulk_vs_distortion_distribution(bulk, distortion1, distortion2, feature_indices=None, bins=50)

    bulk_tensor = torch.tensor(bulk, dtype=torch.float32)
    distortion1_tensor = torch.tensor(distortion1, dtype=torch.float32)
    distortion2_tensor = torch.tensor(distortion2, dtype=torch.float32)
    if config['testing_dataset_name'].startswith('TCGA'):
        stages_tensor = torch.tensor(np.array(overall_stages['overall_stage_simplified']), dtype=torch.float32)
    elif config['testing_dataset_name'].startswith('Metabric'):
        stages_tensor = torch.tensor(np.array(patient_df['Tumor Stage']), dtype=torch.float32)

    dataset = TensorDataset(bulk_tensor, distortion1_tensor, distortion2_tensor, stages_tensor)
    dataloader = DataLoader(dataset, batch_size=batchsize, shuffle=True)
    print(f"Dataloader created. Number of batches: {len(dataloader)}")

    return dataloader, bulk_tensor, stages_tensor, num_classes

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


def sample_single_cells(sc_df, sample_size=1000):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    indices = torch.randint(0, sc_df.shape[0], (sample_size,), device=device)
    sampled_cells = sc_df.iloc[indices.cpu().numpy()]
    return sampled_cells

def plot_bulk_vs_sampled_distribution(bulk_data, sampled_cells, feature_indices=None, bins=50):
    """
    Visualize the distribution of bulk_data and sampled single cells by plotting the selected features' distributions.

    Parameters:
    - bulk_data: Original bulk expression data.
    - sampled_cells: Sampled single cell data.
    - feature_indices: List of indices of features to plot. If None, plot the distribution of all features' mean.
    - bins: Number of bins to use in the histogram.
    """
    if feature_indices is None:
        # If no specific features are selected, plot the mean of all features
        bulk_mean = bulk_data.mean(axis=0)
        sampled_mean = sampled_cells.mean(axis=0)

        plt.figure(figsize=(10, 6))
        sns.histplot(bulk_mean, bins=bins, kde=True, label="Bulk Data (mean)", color="blue", stat="density")
        sns.histplot(sampled_mean, bins=bins, kde=True, label="Sampled Cells (mean)", stat="density")
    else:
        plt.figure(figsize=(10, 6))
        for feature_idx in feature_indices:
            sns.histplot(bulk_data[:, feature_idx], bins=bins, kde=True, label=f"Bulk Feature {feature_idx}", color="blue", stat="density")
            sns.histplot(sampled_cells[:, feature_idx], bins=bins, kde=True, label=f"Sampled Cells Feature {feature_idx}", stat="density")

    plt.title("Distribution Comparison: Bulk Data vs Sampled Single Cells")
    plt.xlabel("Expression Level")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True)
    plt.show()

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


def apply_scaling(bulk_data, sampled_cells, scaling_method):
    if scaling_method == 'minmax':
        scaler = MinMaxScaler()
    elif scaling_method == 'standard':
        scaler = StandardScaler()
    else:
        raise ValueError("Unsupported scaling method")

    bulk_data_scaled = scaler.fit_transform(bulk_data)
    sampled_cells_scaled = scaler.fit_transform(sampled_cells)
    return bulk_data_scaled, sampled_cells_scaled

def filter_genes_by_variance(bulk_data, sampled_cells, variance_threshold):
    bulk_var = np.var(bulk_data, axis=0)
    sampled_var = np.var(sampled_cells, axis=0)
    combined_var = bulk_var + sampled_var
    var_cutoff = np.sort(combined_var)[-int(combined_var.shape[0] * variance_threshold)]
    selected_genes = combined_var > var_cutoff
    return bulk_data[:, selected_genes], sampled_cells[:, selected_genes], selected_genes


def generate_distortions(bulk_data, sampled_cells, lambda_noise, sample_size):
    """
    Generate distortions for the given bulk data using a set of sampled single cell data.

    Parameters:
    - bulk_data: Bulk expression data (numpy array).
    - sampled_cells: Sampled single cell data (numpy array).
    - lambda_noise: Noise weight for distortion.

    Returns:
    - distortions_tensor: Distorted data as a numpy array.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Convert bulk data and sampled_cells to tensors
    bulk_tensor = torch.tensor(bulk_data, dtype=torch.float32).to(device)
    sampled_tensor = torch.tensor(sampled_cells, dtype=torch.float32).to(device)

    distortions = []

    # Generate different sampled cells for each bulk sample
    for i in range(bulk_tensor.shape[0]):
        # Randomly sample a new set of cells from sampled_tensor for each bulk sample
        indices = torch.randint(0, sampled_tensor.shape[0], (sample_size,), device=device)  # 1000 cells for each bulk sample
        random_sampled_cells = sampled_tensor[indices]

        # Calculate the mean values of the randomly sampled cells
        mean_values = torch.mean(random_sampled_cells, dim=0)

        # Generate the distorted sample using a combination of bulk sample and mean values of sampled cells
        distorted = bulk_tensor[i] + mean_values
        distortions.append(distorted)

    # Stack all distorted samples into a single tensor
    distortions_tensor = torch.stack(distortions)

    return distortions_tensor.cpu().numpy()


class TensorDataset(Dataset):
    def __init__(self, tensor1, tensor2, tensor3):
        assert tensor1.size(0) == tensor2.size(0) == tensor3.size(0)
        self.tensor1 = tensor1
        self.tensor2 = tensor2
        self.tensor3 = tensor3

    def __len__(self):
        return self.tensor1.size(0)  # Number of samples

    def __getitem__(self, idx):
        return self.tensor1[idx], self.tensor2[idx], self.tensor3[idx]

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

def filter_bulk_data(bulk, stage_df=None, survival_df=None, eval_metric='stage'):
    """
    Filter bulk data based on the evaluation metric.

    Parameters:
    - bulk: The bulk data DataFrame.
    - stage_df: The stage data DataFrame (optional).
    - survival_df: The survival data DataFrame (optional).
    - eval_metric: The evaluation metric to use ('stage' or 'survival').

    Returns:
    - filtered_bulk: The filtered bulk data.
    - relevant_data: The relevant stage or survival data.
    """
    if eval_metric == 'stage':
        common_index = bulk.index.intersection(stage_df.index)
        filtered_bulk = bulk.loc[common_index]
        relevant_data = stage_df.loc[common_index]

        filtered_bulk = filtered_bulk.sort_index()
        relevant_data = relevant_data.sort_index()

        print(f"After filtering: Bulk data shape: {filtered_bulk.shape}, Number of patients: {len(relevant_data)}")

        if not (filtered_bulk.index == relevant_data.index).all():
            raise ValueError("Bulk data and stage data indices do not match!")

    elif eval_metric == 'survival':
        common_index = bulk.index.intersection(survival_df.index)
        filtered_bulk = bulk.loc[common_index]
        relevant_data = survival_df.loc[common_index]

        filtered_bulk = filtered_bulk.sort_index()
        relevant_data = relevant_data.sort_index()

        print(f"After filtering: Bulk data shape: {filtered_bulk.shape}, Number of patients: {len(relevant_data)}")

    else:
        raise ValueError("Unsupported evaluation metric")

    return filtered_bulk, relevant_data

def Barlow_dataloader(sc_path, bulk_path, stage_path, survival_path, batchsize):
    set_seed(42)

    sc_data = ad.read_h5ad(sc_path)
    cell_names = sc_data.obs_names
    gene_names = sc_data.var['feature_name']
    sc_data_exp = sc_data.X.toarray()
    sc_df = pd.DataFrame(sc_data_exp, index=cell_names, columns=gene_names)
    survival_df = pd.read_csv(survival_path, sep=',', index_col=0)
    print(f" Single-cell data loaded. Shape: {sc_df.shape}")

    if config['testing_dataset_name'].startswith('TCGA'):
        bulk = pd.read_csv(bulk_path, sep='\t', index_col=0)
        bulk.index = [i[:-3] for i in bulk.index]
        bulk = bulk.apply(lambda row: row.fillna(row.mean()), axis=1)

        stage_df = pd.read_csv(stage_path, sep='\t', index_col=0)
        overall_stages = pat_preprosses(stage_df)
        num_classes = len(overall_stages['overall_stage_simplified'].unique())

    elif config['testing_dataset_name'].startswith('Metabric'):
        bulk = pd.read_csv(bulk_path, index_col=0, sep='\t')
        bulk = bulk.drop(bulk.columns[0], axis=1).T
        bulk = bulk.apply(lambda row: row.fillna(row.mean()), axis=1)

        stage_df = pd.read_csv(stage_path, sep='\t', index_col=0)
        stage_df = stage_df.set_index(stage_df.columns[0])
        num_classes = len(stage_df['Tumor Stage'].unique())

    print(f"Bulk data loaded. Shape: {bulk.shape}")
    print(f"Number of unique classes: {num_classes}")

    common_genes = sc_df.columns.intersection(bulk.columns)
    sc_df = sc_df.loc[:, ~sc_df.columns.duplicated()]
    bulk = bulk.loc[:, ~bulk.columns.duplicated()]
    sc_df = sc_df[common_genes]
    bulk = bulk[common_genes]

    # Filter bulk data based on evaluation metric
    if config['eval_metric'] == 'stage':
        bulk, stage_df = filter_bulk_data(bulk, stage_df=stage_df, eval_metric='stage')
    elif config['eval_metric'] == 'survival':
        bulk, survival_df = filter_bulk_data(bulk, survival_df=survival_df, eval_metric='survival')

    # Step 1: Apply scaling
    bulk_scaled, sampled_scaled = apply_scaling(bulk.values, sc_df.values, scaling_method=config['scaling_method'])

    # Step 2: Filter genes by variance
    bulk_filtered, sampled_filtered, selected_genes = filter_genes_by_variance(bulk_scaled, sampled_scaled, variance_threshold=config['variance_threshold'])

    plot_bulk_vs_sampled_distribution(bulk_filtered, sampled_filtered, feature_indices=None, bins=50)

    # Step 3: Generate distortions with different sampled single cells each time
    distortion1 = generate_distortions(bulk_filtered, sampled_filtered, lambda_noise=config['lambda_noise'], sample_size=config['sample_size'])
    distortion2 = generate_distortions(bulk_filtered, sampled_filtered, lambda_noise=config['lambda_noise'], sample_size=config['sample_size'])
    print("Distortions generated.")

    plot_bulk_vs_distortion_distribution(bulk_filtered, distortion1, distortion2)

    # Convert to tensors while keeping the original index
    bulk_tensor = torch.tensor(bulk_filtered, dtype=torch.float32)
    distortion1_tensor = torch.tensor(distortion1, dtype=torch.float32)
    distortion2_tensor = torch.tensor(distortion2, dtype=torch.float32)

    if config['testing_dataset_name'].startswith('TCGA'):
        stages_tensor = torch.tensor(np.array(overall_stages['overall_stage_simplified']), dtype=torch.float32)
    elif config['testing_dataset_name'].startswith('Metabric'):
        stages_tensor = torch.tensor(np.array(stage_df['Tumor Stage']), dtype=torch.float32)

    dataset = TensorDataset(bulk_tensor, distortion1_tensor, distortion2_tensor)
    dataloader = DataLoader(dataset, batch_size=batchsize, shuffle=True)
    print(f"Dataloader created. Number of batches: {len(dataloader)}")

    return dataloader, bulk_tensor, stages_tensor, survival_df, num_classes, bulk
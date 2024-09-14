#!/usr/bin/env python
# coding: utf-8

import anndata as ad
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

np.random.seed(42)


### sampling single cell data as distortion input
def distort_gen(n_repeats, sc_df, lambda_noise, noise_mean=0, noise_std=1, sample_size=1000, ):

    average_values_list = []

    for _ in range(n_repeats):
        # randomly choose 1000 cells
        sampled_cells = sc_df.sample(n=sample_size, replace=False)
        # averge value of these cells
        mean_values = sampled_cells.mean()
        noise = np.random.normal(loc=noise_mean, scale=noise_std, size=len(mean_values))
        mean_values_with_noise = mean_values + lambda_noise * noise
        average_values_list.append(mean_values_with_noise)

    distortion = pd.concat(average_values_list, axis=1)
    distortion = distortion.T

    return distortion

def advanced_distort_gen(bulk_data, sc_df, lambda_noise, sample_size=1000):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} for distortion generation")

    n_repeats = bulk_data.shape[0]
    bulk_tensor = torch.tensor(bulk_data.values, dtype=torch.float32).to(device)
    sc_tensor = torch.tensor(sc_df.values, dtype=torch.float32).to(device)

    distortions = []
    
    for i in range(n_repeats):
        # if i % 100 == 0:
        #     print(f"Generating distortion {i+1}/{n_repeats}")
        
        # Randomly sample cells
        indices = torch.randint(0, sc_tensor.shape[0], (sample_size,), device=device)
        sampled_cells = sc_tensor[indices]
        
        # Calculate mean
        mean_values = torch.mean(sampled_cells, dim=0)
        
        # # Generate noise
        # noise = torch.normal(mean=bulk_tensor.mean(), std=bulk_tensor.std(), size=mean_values.shape, device=device)
        
        # Combine bulk data, mean values, and noise
        distorted = (1 - lambda_noise) * bulk_tensor[i] + lambda_noise * mean_values
        distortions.append(distorted)
    
    distortions_tensor = torch.stack(distortions)
    return distortions_tensor.cpu().numpy()

def data_augmentation(bulk_data, sc_df):
    augmented_data = []
    for _, bulk_sample in bulk_data.iterrows():
        # 从单细胞数据中随机选择细胞
        sampled_cells = sc_df.sample(n=100, replace=True)
        
        # 计算bulk样本和单细胞样本的加权平均
        weight = np.random.beta(2, 2)  # 使用Beta分布生成权重
        augmented_sample = weight * bulk_sample + (1 - weight) * sampled_cells.mean()
        
        augmented_data.append(augmented_sample)
    
    return pd.DataFrame(augmented_data, columns=bulk_data.columns)

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
        'Stage I': 1,
        'Stage IA': 1,
        'Stage IB': 1,
        'Stage II': 2,
        'Stage IIA': 2,
        'Stage IIB': 2,
        'Stage III': 3,
        'Stage IIIA': 3,
        'Stage IIIB': 3,
        'Stage IIIC': 3,
        'Stage IV': 4,
        'Stage X': 5
    }

    overall_stages['overall_stage_simplified'] = overall_stages['overall_stage'].map(stage_mapping)

    return overall_stages


def Barlow_dataloader(sc_path, bulk_path, patient_path, batchsize):

    sc_data = ad.read_h5ad(sc_path)
    cell_names = sc_data.obs_names
    gene_names = sc_data.var['feature_name']
    sc_data_exp = sc_data.X.toarray()
    sc_df = pd.DataFrame(sc_data_exp, index=cell_names, columns=gene_names)
    print(f" Single-cell data loaded. Shape: {sc_df.shape}")

    bulk = pd.read_csv(bulk_path, sep='\t', index_col=0)
    bulk.index = [i[:-3] for i in bulk.index]
    print(f"Bulk data loaded. Shape: {bulk.shape}")

    patient_df = pd.read_csv(patient_path, sep='\t', index_col=0)
    overall_stages = pat_preprosses(patient_df)
    print(f"Patient information loaded. Number of patients: {len(overall_stages)}")

    common_genes = sc_df.columns.intersection(bulk.columns)
    sc_df = sc_df[common_genes]
    bulk = bulk[common_genes]

    common_index = bulk.index.intersection(overall_stages.index)
    overall_stages = overall_stages.loc[common_index]
    bulk = bulk.loc[common_index]
    print(f"After filtering: Bulk data shape: {bulk.shape}, Number of patients: {len(overall_stages)}")

    distortion1 = advanced_distort_gen(bulk, sc_df=sc_df, lambda_noise=0.1)
    distortion2 = advanced_distort_gen(bulk, sc_df=sc_df, lambda_noise=0.1)
    print("Distortions generated.")

    bulk_tensor = torch.tensor(np.array(bulk), dtype=torch.float32)
    distortion1_tensor = torch.tensor(distortion1, dtype=torch.float32)
    distortion2_tensor = torch.tensor(distortion2, dtype=torch.float32)
    stages_tensor = torch.tensor(np.array(overall_stages['overall_stage_simplified']), dtype=torch.float32)

    dataset = TensorDataset(bulk_tensor, distortion1_tensor, distortion2_tensor, stages_tensor)
    dataloader = DataLoader(dataset, batch_size=batchsize, shuffle=True)
    print(f"Dataloader created. Number of batches: {len(dataloader)}")

    return dataloader, bulk_tensor, stages_tensor

#!/usr/bin/env python
# coding: utf-8

import anndata as ad
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import random
from sklearn.preprocessing import MinMaxScaler, StandardScaler
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


def advanced_distort_gen(bulk_data, sc_df, lambda_noise, minmax_icon, standard_icon, sample_size=1000):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} for distortion generation")

    n_repeats = bulk_data.shape[0]
    if minmax_icon == True:
        scaler = MinMaxScaler()
        bulk_data = scaler.fit_transform(bulk_data.T).T

        scaler = MinMaxScaler()
        sc_df = scaler.fit_transform(sc_df.T).T
    
    elif standard_icon == True:
        scaler = StandardScaler()
        bulk_data = scaler.fit_transform(bulk_data.T).T

        scaler = StandardScaler()
        sc_df = scaler.fit_transform(sc_df.T).T

    else:
        bulk_data = np.array(bulk_data)
        sc_df = np.array(sc_df)

    bulk_tensor = torch.tensor(bulk_data, dtype=torch.float32).to(device)
    sc_tensor = torch.tensor(sc_df, dtype=torch.float32).to(device)

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
        
        ## choose to do minmax or not


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

    # 确保 bulk 和 patient_df 的索引一致
    common_index = bulk.index.intersection(patient_df.index)
    bulk = bulk.loc[common_index]
    patient_df = patient_df.loc[common_index]
    
    # 确保两者的索引顺序一致
    bulk = bulk.sort_index()
    patient_df = patient_df.sort_index()
    
    print(f"After filtering: Bulk data shape: {bulk.shape}, Number of patients: {len(patient_df)}")
    
    if not (bulk.index == patient_df.index).all():
        raise ValueError("Bulk data and patient data indices do not match!")

    distortion1 = advanced_distort_gen(bulk, sc_df=sc_df, lambda_noise=0.1, minmax_icon=config['minmax_icon'], standard_icon=config['standard_icon'])
    distortion2 = advanced_distort_gen(bulk, sc_df=sc_df, lambda_noise=0.1, minmax_icon=config['minmax_icon'], standard_icon=config['standard_icon'])
    print("Distortions generated.")

    bulk_tensor = torch.tensor(np.array(bulk), dtype=torch.float32)
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

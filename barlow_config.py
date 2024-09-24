import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

config = {
    'seed': 42,
    'batchsize': 128,
    'sample_size' : 1000,
    'lambda_noise': 0.5,
    'variance_threshold': 0.85,
    'output_dim': 64,
    'project_dim': 128,
    'mlp_hidden_dim': 64,
    'learning_rate': 1e-4,
    'weight_decay' : 1e-3,
    'num_epochs': 5000,
    'loss_lambda_param' : 0.005,
    'scaling_method' : 'minmax',
    'model_save_path': './model_checkpoints',
    'fra_save_path': './fra_pre',
    'training_dataset_name': 'All',
    'testing_dataset_name': 'Metabric',
    'device': device,
    'sc_path': './cancer_single_cell_data/92_sc_healthy.h5ad',
    'bulk_path': './cancer_brca_metabric_bulk_data/data_mrna_illumina_microarray_zscores_ref_diploid_samples.txt',
    'patient_path': './cancer_brca_metabric_bulk_data/Tumor_Stage.txt',
}



import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

config = {
    'seed': 42,
    'batchsize': 128,
    'output_dim': 64,
    'project_dim': 512,
    'mlp_hidden_dim': 64,
    'learning_rate': 1e-4,
    'weight_decay' : 1e-3,
    'num_epochs': 5000,
    'loss_lambda_param' : 0.005,
    'minmax_icon' : True,
    'model_save_path': './model_checkpoints',
    'fra_save_path': './fra_pre',
    'training_dataset_name': 'All',
    'testing_dataset_name': 'TCGA_2015_Cell',
    'device': device,
    'sc_path': '../cancer_data/92_sc_healthy.h5ad',
    'bulk_path': '../cancer_data/2015_bulk_rna_seq.txt',
    'patient_path': '../brca_tcga_pub2015/data_clinical_patient.txt',
}



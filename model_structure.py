import torch
import torch.nn as nn
import torch.optim as optim

# Define the Encoder part of Barlow Twins
class Encoder(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 1024),  # 更大的隐藏层
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(),

            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),

            nn.Linear(512, 512),  # 添加更多层
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),

            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        return self.layers(x)


# Define the Projector part of Barlow Twins
class Projector(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, output_dim)
        )

    def forward(self, x):
        return self.layers(x)


class MLPClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.layers(x)
        return self.softmax(x)


# Define the Barlow Twins loss function
class BarlowTwinsLoss(nn.Module):
    def __init__(self, lambda_param=5e-3):
        super().__init__()
        self.lambda_param = lambda_param

    def forward(self, z1, z2):
        c = torch.mm(z1.T, z2) / z1.shape[0]
        c_diff = (c - torch.eye(c.shape[0], device=c.device)).pow(2)
        c_diff[~torch.eye(c.shape[0], dtype=bool)] *= self.lambda_param
        loss = c_diff.sum()
        return loss
import torch
import torch.nn as nn
from dataset_input_back import set_seed

set_seed(42)

# Define the Encoder part of Barlow Twins
class Encoder(nn.Module):
    def __init__(self, input_dim, output_dim, dropout_rate=0.2):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            nn.Linear(256, output_dim),
        )

    def forward(self, x):
        return self.layers(x)


import torch
import torch.nn as nn


# 定义残差瓶颈块（Bottleneck Block）
class GeneResNetBlock(nn.Module):
    def __init__(self, in_features, mid_features, expansion=4, downsample=None):
        super(GeneResNetBlock, self).__init__()
        # 1x1 全连接层：降维
        self.fc1 = nn.Linear(in_features, mid_features)
        self.bn1 = nn.BatchNorm1d(mid_features)

        # 3x3 等效操作：保持维度
        self.fc2 = nn.Linear(mid_features, mid_features)
        self.bn2 = nn.BatchNorm1d(mid_features)

        # 1x1 全连接层：升维
        self.fc3 = nn.Linear(mid_features, mid_features * expansion)
        self.bn3 = nn.BatchNorm1d(mid_features * expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample  # 调整尺寸（如果需要）
        self.expansion = expansion

    def forward(self, x):
        identity = x

        # 1x1 降维 -> 3x3 变换 -> 1x1 升维
        out = self.fc1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.fc2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.fc3(out)
        out = self.bn3(out)

        # 如果输入输出维度不一致，则调整 identity
        if self.downsample is not None:
            identity = self.downsample(x)

        # 残差连接
        out += identity
        out = self.relu(out)

        return out


# 基于残差块构建的 `ResNet Encoder`
class GeneResNetEncoder(nn.Module):
    def __init__(self, input_dim, output_dim, dropout_rate=0.2):
        super(GeneResNetEncoder, self).__init__()

        # 第一个线性层（类似 ResNet 中的第一层 Conv）
        self.initial_layer = nn.Sequential(
            nn.Linear(input_dim, 512),  # 降低输入特征维度
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
        )

        # 残差层（ResNet 模块）
        self.layer1 = self._make_layer(512, 128, expansion=4, num_blocks=2)  # 第一组残差块
        self.layer2 = self._make_layer(512, 64, expansion=4, num_blocks=2)  # 第二组残差块
        self.layer3 = self._make_layer(256, 32, expansion=4, num_blocks=2)  # 第三组残差块

        # 最终输出层：将特征压缩至 output_dim
        self.fc = nn.Sequential(
            nn.Linear(128, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
        )

    def _make_layer(self, in_features, mid_features, expansion=4, num_blocks=2):
        """构建残差层（多个 Bottleneck Block 堆叠）"""
        layers = []
        downsample = None
        # 如果输入特征数与瓶颈块的输出特征数不同，则使用 downsample 进行调整
        if in_features != mid_features * expansion:
            downsample = nn.Sequential(
                nn.Linear(in_features, mid_features * expansion),
                nn.BatchNorm1d(mid_features * expansion),
            )

        # 堆叠多个残差瓶颈块
        layers.append(GeneResNetBlock(in_features, mid_features, expansion, downsample))

        for _ in range(1, num_blocks):
            layers.append(GeneResNetBlock(mid_features * expansion, mid_features, expansion))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.initial_layer(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.fc(x)
        return x


# Define the Projector part of Barlow Twins
class Projector(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),

            nn.Linear(hidden_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),

            nn.Linear(hidden_dim, output_dim, bias=False),
            nn.BatchNorm1d(output_dim, affine=False)
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

    def forward(self, x):
        return self.layers(x)


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
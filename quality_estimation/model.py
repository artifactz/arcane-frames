import torch
from torch import nn


class QualityModel(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        # self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.linear_1 = nn.Linear(in_features, 512)
        self.linear_2 = nn.Linear(512, 128)
        self.linear_out = nn.Linear(128, out_features)
        self.dropout = nn.Dropout()

    def forward(self, x):
        x = self.linear_1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear_2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear_out(x)
        return x


def pretrained() -> QualityModel:
    model = QualityModel(in_features=3048, out_features=1)
    model.load_state_dict(torch.load('quality_estimation/model.pth'))
    model.eval()
    return model

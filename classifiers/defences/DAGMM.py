from typing import Tuple
from torch import nn, Tensor, tanh, cat
import torch.nn.functional as F


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()

    def forward(self, x: Tensor) -> Tensor:
        pass


class DAGMM(nn.Module):
    def __init__(self, n_gmm=2, z_dim=1):
        """Network for DAGMM"""
        super(DAGMM, self).__init__()
        # Encoder network
        self.fc1 = nn.Linear(118, 60)
        self.fc2 = nn.Linear(60, 30)
        self.fc3 = nn.Linear(30, 10)
        self.fc4 = nn.Linear(10, z_dim)

        # Decoder network
        self.fc5 = nn.Linear(z_dim, 10)
        self.fc6 = nn.Linear(10, 30)
        self.fc7 = nn.Linear(30, 60)
        self.fc8 = nn.Linear(60, 118)

        # Estimation network
        self.fc9 = nn.Linear(z_dim + 2, 10)
        self.fc10 = nn.Linear(10, n_gmm)

    def encode(self, x: Tensor) -> Tensor:
        h = tanh(self.fc1(x))
        h = tanh(self.fc2(h))
        h = tanh(self.fc3(h))
        return self.fc4(h)

    def decode(self, x: Tensor) -> Tensor:
        h = tanh(self.fc5(x))
        h = tanh(self.fc6(h))
        h = tanh(self.fc7(h))
        return self.fc8(h)

    def estimate(self, z: Tensor) -> Tensor:
        h = F.dropout(tanh(self.fc9(z)), 0.5)
        return F.softmax(self.fc10(h), dim=1)

    def compute_reconstruction(self, x: Tensor, x_hat: Tensor) -> Tuple[Tensor, Tensor]:
        relative_euclidean_distance = (x - x_hat).norm(2, dim=1) / x.norm(2, dim=1)
        cosine_similarity = F.cosine_similarity(x, x_hat, dim=1)
        return relative_euclidean_distance, cosine_similarity

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        z_c = self.encode(x)
        x_hat = self.decode(z_c)
        rec_1, rec_2 = self.compute_reconstruction(x, x_hat)
        z = cat([z_c, rec_1.unsqueeze(-1), rec_2.unsqueeze(-1)], dim=1)
        gamma = self.estimate(z)
        return z_c, x_hat, z, gamma


class GMM(nn.Module):
    pass

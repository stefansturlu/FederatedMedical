from copy import deepcopy
import os
from utils.typings import Errors
from experiment.AggregatorConfig import AggregatorConfig
from torch import nn, Tensor
from client import Client
from logger import logPrint
from typing import List, Type
import torch
from aggregators.Aggregator import Aggregator
from datasetLoaders.DatasetInterface import DatasetInterface
from sklearn.cluster import KMeans
from utils.PCA import PCA
import matplotlib.pyplot as plt

# Group-Wise Aggregator based on clustering
# Even though it itself does not do aggregation, it makes programatic sense to inherit attributes and functions
# USE 10 EPOCHS PER ROUND FOR CLUSTERING
class ClusteringAggregator(Aggregator):
    def __init__(
        self,
        clients: List[Client],
        model: nn.Module,
        config: AggregatorConfig,
        useAsyncClients: bool = False,
    ):
        super().__init__(clients, model, config, useAsyncClients)

        self.config = config

        self.cluster_count = config.cluster_count
        self.cluster_centres: List[nn.Module] = [None] * self.cluster_count
        self.cluster_centres_len = torch.zeros(self.cluster_count)
        self.cluster_labels = [0] * len(self.clients)

        self.internalAggregator: Aggregator = None
        self.externalAggregator: Aggregator = None

        self.blocked_ps = []

    def trainAndTest(self, testDataset: DatasetInterface) -> Errors:
        roundsError = Errors(torch.zeros(self.config.rounds))
        for r in range(self.config.rounds):
            self.round = r
            logPrint("Round... ", r)
            self._shareModelAndTrainOnClients()

            models = self._retrieveClientModelsDict()

            # Perform Clustering
            with torch.no_grad():
                self.generate_cluster_centres(models)
                ps = self.cluster_centres_len / self.cluster_centres_len.sum()

                self.model = self.externalAggregator.aggregate(
                   [FakeClient(p, i) for (i, p) in enumerate(ps)], self.cluster_centres
                )

                roundsError[r] = self.test(testDataset)

        return roundsError


    def _init_aggregators(self, internal: Type[Aggregator], external: Type[Aggregator]) -> None:
        self.internalAggregator = internal(self.clients, self.model, self.config)
        self.externalAggregator = external(self.clients, self.model, self.config)


    def _gen_cluster_centre(self, indices: List[int], models: List[nn.Module]) -> nn.Module:
        """ Takes the average of the clients assigned to each cluster to generate a new centre """
        # Here you should be using other robust aggregation algorithms to perform the centre calculation and blocking

        return self.internalAggregator.aggregate([self.clients[i] for i in indices], [models[i] for i in indices])

    def _generate_weights(self, models: List[nn.Module]) -> List[Tensor]:
        X = []
        for model in models:
            X.append(self._generate_coords(model))

        return X

    def _generate_coords(self, model: nn.Module) -> Tensor:
        coords = torch.tensor([]).to(self.device)

        for param in model.parameters():
            coords = torch.cat((coords, param.data.view(-1)))

        return coords

    def generate_cluster_centres(self, models: List[nn.Module]) -> None:
        X = self._generate_weights(models)
        X = [model.tolist() for model in X]
        # self.__elbow_test(X, models)
        p = PCA.pca(X, 4) # 4
        kmeans = KMeans(n_clusters=self.cluster_count, random_state=0).fit(p)

        # if self.round % 5 == 0:
            # PCA.pca1D(X, self.clients)
        # PCA.pca2D(X, self.clients)
        # PCA.pca3D(X, self.clients)
        PCA.pca4D(X, self.clients)
        # if self.round % 5 == 0:
        # PCA.optimal_component_plot(X)

        self.cluster_labels = kmeans.labels_
        indices: List[List[int]] = [[] for _ in range(self.cluster_count)]
        self.cluster_centres_len.zero_()

        for i, l in enumerate(self.cluster_labels):
            self.cluster_centres_len[l] += 1
            indices[l].append(i)

        print(self.cluster_labels)

        self.cluster_centres_len /= len(self.clients)
        for i, ins in enumerate(indices):
            self.cluster_centres[i] = self._gen_cluster_centre(ins, models)


    def __elbow_test(self, X, models: List[nn.Module]) -> None:
        dispersion = []

        for h in range(len(self.clients)):
            kmeans = KMeans(n_clusters=h+1, random_state=0).fit(X)
            labels = kmeans.labels_

            indices: List[List[int]] = [[] for _ in range(h+1)]
            lens = torch.zeros(h+1)
            lens.zero_()

            centres: List[nn.Module] = []

            for i, l in enumerate(labels):
                lens[l] += 1
                indices[l].append(i)

            lens /= len(self.clients)
            d = 0
            for i, ins in enumerate(indices):
                centres.append(self._gen_cluster_centre(ins, models))

            for i, ins in enumerate(indices):
                ms = [models[j] for j in ins]
                c_coords = torch.tensor([]).to(self.device)
                for param in centres[i].parameters():
                    c_coords = torch.cat((c_coords, param.data.view(-1)))

                for m in ms:
                    m_coords = torch.tensor([]).to(self.device)
                    for param in m.parameters():
                        m_coords = torch.cat((m_coords, param.data.view(-1)))

                    d += (c_coords - m_coords).square().sum()

            dispersion.append(d)


        # plt.figure()
        # plt.plot(range(1, 31), dispersion)
        # plt.title(f"Sum of Distances from Cluster Centre as K Increases \n 20 Malicious - Round: {self.round}")
        # plt.xlabel("K-Value")
        # plt.ylabel("Sum of Distances")
        # if not os.path.exists("k_means_test/20_mal"):
        #     os.makedirs("k_means_test/20_mal")
        # plt.savefig(f"k_means_test/20_mal/{self.round}.png")


class FakeClient:
    def __init__(self, p: float, id: int):
        self.p = p
        self.id = id

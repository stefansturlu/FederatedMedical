from aggregators.FedMGDAPlus import FedMGDAPlusAggregator
from aggregators.AFA import AFAAggregator
from copy import deepcopy
from experiment.DefaultExperimentConfiguration import DefaultExperimentConfiguration
from torch import nn, device, Tensor
from client import Client
from logger import logPrint
from typing import Dict, List
import torch
from aggregators.Aggregator import Aggregator
from datasetLoaders.DatasetInterface import DatasetInterface
import matplotlib.pyplot as plt

# Group-Wise Aggregator based on clustering
# Even though it itself does not do aggregation, it makes programatic sense to inherit attributes and functions
class GroupWiseAggregation(Aggregator):
    def __init__(self, clients: List[Client], model: nn.Module, rounds: int, device: device, detectFreeRiders:bool, config: DefaultExperimentConfiguration, useAsyncClients:bool=False):
        super().__init__(clients, model, rounds, device, detectFreeRiders, useAsyncClients)

        self.config = config

        self.cluster_count = 3
        self.cluster_centres = None
        self.cluster_choices = torch.zeros(len(self.clients))

        self.internalAggregator = self._init_aggregator(config.internalAggregator)
        self.externalAggregator = self._init_aggregator(config.externalAggregator)

    def trainAndTest(self, testDataset: DatasetInterface) -> Tensor:
        roundsError = torch.zeros(self.config.rounds)
        for r in range(self.config.rounds):
            logPrint("Round... ", r)
            self._shareModelAndTrainOnClients()
            models = self._retrieveClientModelsDict()

            # Perform Clustering
            self.clustering(models)
            cluster_models = []
            cluster_model_count = []

            for choice in self.cluster_choices:
                empty_model = deepcopy(models[1])
                indices = self.cluster_choices == choice
                cluster_model_count.append(len(indices))

                comb = 0.0
                for i in indices:
                    self._mergeModels(
                        models[i].to(self.config.device),
                        empty_model.to(self.config.device),
                        self.clients[i].p,
                        comb,
                    )
                    comb = 1.0

                cluster_models.append(empty_model)

            # Assume p value is based on size of cluster
            self.model = self.externalAggregator.aggregate([{"p": val/len(self.clients)} for val in cluster_model_count], cluster_models)

            roundsError[r] = self.test(testDataset)

        return roundsError


    def _init_aggregator(self, aggregator: Aggregator) -> Aggregator:
        agg = aggregator(self.clients, self.model, self.rounds, self.device, self.detectFreeRiders)
        if isinstance(agg, AFAAggregator):
            agg.xi = self.config.xi
            agg.deltaXi = self.config.deltaXi
        elif isinstance(agg, FedMGDAPlusAggregator):
            agg.reinitialise(self.config.innerLR)

        return agg


    def _init_cluster_centres(self, models_weights: Tensor) -> None:
        if self.cluster_centres == None:
            indices: Tensor = torch.randint(high=len(models_weights), size=self.cluster_count)
            self.cluster_centres = models_weights[indices]

        else:
            for choice in range(self.cluster_count):
                indices = self.cluster_choices == choice
                cluster = models_weights[indices]
                self.cluster_centres[choice] = self._gen_cluster_centre(cluster)


    def _gen_cluster_centre(self, indices: Tensor) -> Tensor:
        """ Takes the average of the clients assigned to each cluster to generate a new centre """
        # Here you should be using other robust aggregation algorithms to perform the centre calculation and blocking

        model = self.internalAggregator.aggregate([self.clients[i] for i in indices], )

        return self._generate_weights({0: model})[0]


    def _generate_weights(self, models: Dict[int, nn.Module]) -> Tensor:
        X = []
        for id, model in models.items():
            coords = []

            for name, param in model.named_parameters():
                coords.append(param.data.view(-1))

            X.append(coords)

        return torch.tensor(X, device=self.config.device)


    # I use Cosine Similarity to cluster as I believe it to be a better similarity
    # metric than just Euclidean distance
    def clustering(self, models: Dict[int, nn.Module]):
        models_weights = self._generate_weights(models)
        self._init_cluster_centres(models_weights)
        cos = nn.CosineSimilarity(0)

        old_centres = self.cluster_centres

        # While the clusters are still converging
        while cos(old_centres, self.cluster_centres) < 0.99:
            old_centres = self.cluster_centres

            for i, model in enumerate(models_weights):
                best_sim = 0
                choice = -1
                for j, cluster in enumerate(self.cluster_centres):
                    sim = cos(model, cluster)
                    if sim > best_sim:
                        best_sim = sim
                        choice = j

                self.cluster_choices[i] = choice

            self._init_cluster_centres(models_weights)


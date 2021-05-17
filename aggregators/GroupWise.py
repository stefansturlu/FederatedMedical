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
from sklearn.cluster import KMeans
import math

# Group-Wise Aggregator based on clustering
# Even though it itself does not do aggregation, it makes programatic sense to inherit attributes and functions
class GroupWiseAggregation(Aggregator):
    def __init__(self, clients: List[Client], model: nn.Module, rounds: int, device: device, detectFreeRiders:bool, config: DefaultExperimentConfiguration, useAsyncClients:bool=False):
        super().__init__(clients, model, rounds, device, detectFreeRiders, useAsyncClients)

        self.config = config

        self.cluster_count = 3
        self.cluster_centres: List[nn.Module] = [None]*self.cluster_count
        self.cluster_centres_p = [0]*self.cluster_count
        self.cluster_labels = [0]*len(self.clients)

        self.internalAggregator = self._init_aggregator(config.internalAggregator)
        self.externalAggregator = self._init_aggregator(config.externalAggregator)

    def trainAndTest(self, testDataset: DatasetInterface) -> Tensor:
        roundsError = torch.zeros(self.config.rounds)
        for r in range(self.config.rounds):
            logPrint("Round... ", r)
            if r % 5 == 0:
                self._shareModelAndTrainOnClients()
            else:
                self._shareModelAndTrainOnClients(self.cluster_centres, self.cluster_labels)

            models = self._retrieveClientModelsDict()

            # Perform Clustering
            self.generate_cluster_centres(models)


            if r % 5 == 4:
                self._use_most_similar_clusters()
                # Assume p value is based on size of cluster
                class FakeClient:
                    def __init__(self, p:float, id:int):
                        self.p = p
                        self.id = id

                self.model = self.externalAggregator.aggregate([FakeClient(p/len(self.clients), i) for (i, p) in enumerate(self.cluster_centres_p)], self.cluster_centres)

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


    def _init_cluster_centres(self, models: List[nn.Module]) -> None:
        if self.cluster_centres == None:
            indices: Tensor = torch.randint(high=len(models), size=(self.cluster_count,))
            self.cluster_centres = [models[i] for i in indices]

        else:
            for choice in range(self.cluster_count):
                indices = [i for i,val in enumerate(self.cluster_labels) if val == choice]
                print(indices)
                self.cluster_centres[choice] = self._gen_cluster_centre(indices, models)
                self.cluster_centres_p[choice] = len(indices)


    def _gen_cluster_centre(self, indices: Tensor, models: List[nn.Module]) -> nn.Module:
        """ Takes the average of the clients assigned to each cluster to generate a new centre """
        # Here you should be using other robust aggregation algorithms to perform the centre calculation and blocking

        model = self.internalAggregator.aggregate([self.clients[i] for i in indices], models)

        return model


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
        kmeans = KMeans(n_clusters=self.cluster_count, random_state=0).fit(X)


        self.pca2D(X)



        self.cluster_labels = kmeans.labels_
        print(self.cluster_labels)
        indices = [[] for _ in range(self.cluster_count)]

        for i, l in enumerate(self.cluster_labels):
            self.cluster_centres_p[l] += 1
            indices[l].append(i)

        self.cluster_centres_p = [p/len(self.clients) for p in self.cluster_centres_p]

        for label in self.cluster_labels:
            self.cluster_centres[label] = self._gen_cluster_centre(indices[l], models)


    def _use_most_similar_clusters(self) -> None:
        num_to_take = math.floor(self.cluster_count) + 1

        X = self._generate_weights(self.cluster_centres)
        X = [model.tolist() for model in X]
        kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
        print(kmeans.labels_)

        exit(0)




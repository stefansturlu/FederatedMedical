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
            self._shareModelAndTrainOnClients()
            models = self._retrieveClientModelsDict()

            # Perform Clustering
            X = self._generate_weights(models)
            X = [model.tolist() for model in X]
            kmeans = KMeans(n_clusters=3, random_state=0).fit(X)
            self.cluster_labels = kmeans.labels_
            indices = [[] for _ in range(self.cluster_count)]

            for i, l in enumerate(self.cluster_labels):
                self.cluster_centres_p[l] += 1
                indices[l].append(i)

            self.cluster_centres_p = [p/len(self.clients) for p in self.cluster_centres_p]

            for label in self.cluster_labels:
                self.cluster_centres[label] = self._gen_cluster_centre(indices[l], models)

            exit(0)



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


    def _list_tensor_subtraction(self, a, b):
        # cos = nn.CosineSimilarity(0)
        # """ Computes the difference between a list of tensor representations of models """
        distances = [torch.mean(ai-bi) for (ai, bi) in zip(a, b)]
        print(distances)
        return abs(sum(distances) / len(distances))
        j = 0
        for d in distances:
            j += d.sum()
        # exit(0)
        # sim = sum(distances) / len(distances)

        return j

        for (ai, bi) in zip(a, b):
            print(ai-bi)
            print(torch.mean(ai-bi))


        cos = nn.CosineSimilarity(0)
        """ Computes the difference between a list of tensor representations of models """
        distances = [cos(ai, bi) for (ai, bi) in zip(a, b)]
        # print(distances)
        # exit(0)
        sim = sum(distances) / len(distances)
        print(sim)

        return sim


    # I use Cosine Similarity to cluster as I believe it to be a better similarity
    # metric than just Euclidean distance
    def clustering(self, models: List[nn.Module]):
        models_weights = self._generate_weights(models)
        self._init_cluster_centres(models)

        cluster_centres_weights = self._generate_weights(self.cluster_centres)
        old_centres: Tensor = None
        torch.set_printoptions(20)

        # While the clusters are still converging
        while old_centres is None or self._list_tensor_subtraction(old_centres, cluster_centres_weights) > 0.000000000001:
            print("START ITER")
            print(cluster_centres_weights)
            print(old_centres)
            if old_centres is not None:
                print(self._list_tensor_subtraction(old_centres, cluster_centres_weights))
            for i, model in enumerate(models_weights):
                # best_sim = 1000
                best_sim = 0
                choice = -1
                for j, cluster in enumerate(cluster_centres_weights):
                    sim = torch.mean(model - cluster)
                    # print(torch.square(model - cluster).sum())
                    # sim = nn.CosineSimilarity(0)(model, cluster)
                    # print(sim)
                    if sim < best_sim:
                        best_sim = sim
                        choice = j
                self.cluster_labels[i] = choice
            # print(self.cluster_labels)
            old_centres = cluster_centres_weights
            self._init_cluster_centres(models)
            cluster_centres_weights = self._generate_weights(self.cluster_centres)
            print(self._list_tensor_subtraction(old_centres, cluster_centres_weights))
            print("END ITER")
            print(cluster_centres_weights)
            print(old_centres)



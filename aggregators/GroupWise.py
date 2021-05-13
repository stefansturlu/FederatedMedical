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
# It does not inherit from the Aggregator class as it itself does not do the aggregation
class GroupWiseAggregation():
    def __init__(self, clients: List[Client], model: nn.Module, config: DefaultExperimentConfiguration, useAsyncClients:bool=False):
        self.model = model.to(device)
        self.clients: List[Client] = clients
        self.config = config

        self.useAsyncClients = useAsyncClients

        self.cluster_count = 3
        self.cluster_centres = None
        self.internalAggregator: Aggregator = None
        self.externalAggregator: Aggregator = None

    def trainAndTest(self, testDataset: DatasetInterface) -> Tensor:
        roundsError = torch.zeros(self.config.rounds)
        for r in range(self.config.rounds):
            logPrint("Round... ", r)
            self._shareModelAndTrainOnClients()
            models = self._retrieveClientModelsDict()
            # Merge models
            comb = 0.0
            self.clustering(models)
            for client in self.clients:
                self._mergeModels(
                    models[client.id].to(self.config.device),
                    self.model.to(self.config.device),
                    client.p,
                    comb,
                )
                comb = 1.0

            roundsError[r] = self.test(testDataset)

        return roundsError

    def _init_cluster_centres(self, models_weights: Tensor, choices: Tensor) -> None:
        if self.cluster_centres == None:
            indices: Tensor = torch.randint(high=len(models_weights), size=self.cluster_count)
            self.cluster_centres = models_weights[indices]

        else:
            for choice in range(self.cluster_count):
                indices = choices == choice
                cluster = models_weights[indices]
                self.cluster_centres[choice] = self._gen_cluster_centre(cluster)


    def _gen_cluster_centre(self, cluster: Tensor) -> Tensor:
        summation = cluster.sum(0)

        return summation / cluster.shape[0]


    # Taken from AFA
    def __modelSimilarity(self, mOrig: nn.Module, mDest: nn.Module) -> Tensor:
        cos = nn.CosineSimilarity(0)

        d2 = torch.tensor([]).to(self.config.device)
        d1 = torch.tensor([]).to(self.config.device)

        paramsOrig = mOrig.named_parameters()
        paramsDest = mDest.named_parameters()
        dictParamsDest = dict(paramsDest)

        for name1, param1 in paramsOrig:
            if name1 in dictParamsDest:
                d1 = torch.cat((d1, dictParamsDest[name1].data.view(-1)))
                d2 = torch.cat((d2, param1.data.view(-1)))
                # d2 = param1.data
                # sim = cos(d1.view(-1),d2.view(-1))
                # logPrint(name1,param1.size())
                # logPrint("Similarity: ",sim)
        sim: Tensor = cos(d1, d2)
        return d1, d2
        return sim


    def _generate_weights(self, models: Dict[int, nn.Module]) -> Tensor:
        X = []
        for id, model in models.items():
            coords = []

            for name, param in model.named_parameters():
                coords.append(param.data.view(-1))

            X.append(coords)

        return torch.tensor(X, device=self.config.device)


    def clustering(self, models: Dict[int, nn.Module]):
        models_weights = self.generate_weights(models)
        self._init_cluster_centres(models_weights)
        cos = nn.CosineSimilarity(0)
        cluster_choice = torch.zeros(len(self.clients))

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

                cluster_choice[i] = choice

            self._init_cluster_centres(models_weights, cluster_choice)








        # sim = self.__modelSimilarity(models[2], models[1])

        # plt.figure()
        # for client1, model1 in models.items():

        #     for client2, model2 in models.items():
        #         d1, d2 = self.__modelSimilarity(model1, model2)
        #         plt.scatter(client1, sim)
        # plt.show()
from torch import nn, device, Tensor
from client import Client
from logger import logPrint
from typing import Dict, List
import torch
from aggregators.Aggregator import Aggregator
from datasetLoaders.DatasetInterface import DatasetInterface
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# FEDERATED AVERAGING AGGREGATOR
class GroupWiseAggregation(Aggregator):
    def __init__(self, clients: List[Client], model: nn.Module, rounds: int, device: device, detectFreeRiders:bool, useAsyncClients:bool=False):
        super().__init__(clients, model, rounds, device, detectFreeRiders, useAsyncClients)

    def trainAndTest(self, testDataset: DatasetInterface) -> Tensor:
        roundsError = torch.zeros(self.rounds)
        for r in range(self.rounds):
            logPrint("Round... ", r)
            self._shareModelAndTrainOnClients()
            models = self._retrieveClientModelsDict()
            # Merge models
            comb = 0.0
            self.clustering(models)
            for client in self.clients:
                self._mergeModels(
                    models[client.id].to(self.device),
                    self.model.to(self.device),
                    client.p,
                    comb,
                )
                comb = 1.0

            roundsError[r] = self.test(testDataset)

        return roundsError


    # Taken from AFA
    def __modelSimilarity(self, mOrig: nn.Module, mDest: nn.Module) -> Tensor:
        cos = nn.CosineSimilarity(0)

        d2 = torch.tensor([]).to(self.device)
        d1 = torch.tensor([]).to(self.device)

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


    def clustering(self, models: Dict[int, nn.Module]):
        X = []
        for id, model in models.items():
            coords = []

            for name, param in model.named_parameters():
                coords.append(param.data.view(-1).mean().item())

            X.append(coords)

        # print(X[0])
        kmeans = KMeans(3, random_state=0).fit(X)
        print(kmeans.labels_)








        # sim = self.__modelSimilarity(models[2], models[1])

        # plt.figure()
        # for client1, model1 in models.items():

        #     for client2, model2 in models.items():
        #         d1, d2 = self.__modelSimilarity(model1, model2)
        #         plt.scatter(client1, sim)
        # plt.show()
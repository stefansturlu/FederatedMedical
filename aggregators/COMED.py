from torch import nn, device, Tensor
from client import Client
from logger import logPrint
from typing import List
import torch
import copy
from aggregators.Aggregator import Aggregator
from datasetLoaders.DatasetInterface import DatasetInterface


# ROBUST AGGREGATION ALGORITHM - computes the median of the clients updates
class COMEDAggregator(Aggregator):
    def __init__(self, clients: List[Client], model: nn.Module, rounds: int, device: device, detectFreeRiders:bool, useAsyncClients=False):
        super().__init__(clients, model, rounds, device, detectFreeRiders, useAsyncClients)

    def trainAndTest(self, testDataset: DatasetInterface) -> Tensor:
        roundsError = torch.zeros(self.rounds)

        for r in range(self.rounds):
            logPrint("Round... ", r)

            self._shareModelAndTrainOnClients()
            models = self._retrieveClientModelsDict()

            # Merge models
            self.model = self.__medianModels(models)

            roundsError[r] = self.test(testDataset)

        return roundsError

    def __medianModels(self, models: List[nn.Module]) -> nn.Module:
        client1 = self.clients[0]
        model = models[client1.id]
        modelCopy = copy.deepcopy(model)
        params = model.named_parameters()
        for name1, param1 in params:
            m = []
            for client2 in self.clients:
                params2 = models[client2.id].named_parameters()
                dictParams2 = dict(params2)
                m.append(dictParams2[name1].data.view(-1).to("cpu").numpy())
                # logPrint("Size: ", dictParams2[name1].data.size())
            m = torch.tensor(m)
            med = torch.median(m, dim=0)[0]
            dictParamsm = dict(modelCopy.named_parameters())
            dictParamsm[name1].data.copy_(med.view(dictParamsm[name1].data.size()))
            # logPrint("Median computed, size: ", med.size())
        return modelCopy.to(self.device)

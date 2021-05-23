from experiment.AggregatorConfig import AggregatorConfig
from torch import nn, Tensor
from client import Client
from logger import logPrint
from typing import List
import torch
from copy import deepcopy
from aggregators.Aggregator import Aggregator
from datasetLoaders.DatasetInterface import DatasetInterface


# ROBUST AGGREGATION ALGORITHM - computes the median of the clients updates
class COMEDAggregator(Aggregator):
    def __init__(self, clients: List[Client], model: nn.Module, config: AggregatorConfig, useAsyncClients=False):
        super().__init__(clients, model, config, useAsyncClients)

    def trainAndTest(self, testDataset: DatasetInterface) -> Tensor:
        roundsError = torch.zeros(self.rounds)

        for r in range(self.rounds):
            logPrint("Round... ", r)

            self._shareModelAndTrainOnClients()
            models = self._retrieveClientModelsDict()

            # Merge models
            self.model = self.aggregate(models)

            roundsError[r] = self.test(testDataset)

        return roundsError

    def aggregate(self, clients: List[Client], models: List[nn.Module]) -> nn.Module:
        client1 = clients[0]
        model = models[client1.id]
        modelCopy = deepcopy(model)
        params = model.named_parameters()
        for name1, param1 in params:
            m = []
            for client2 in clients:
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


from copy import deepcopy
from experiment.AggregatorConfig import AggregatorConfig
from datasetLoaders.DatasetInterface import DatasetInterface
from client import Client
from logger import logPrint
from typing import List
import torch
from aggregators.Aggregator import Aggregator
from torch import nn, Tensor


class MKRUMAggregator(Aggregator):
    def __init__(self, clients: List[Client], model: nn.Module, config: AggregatorConfig, useAsyncClients:bool=False):
        super().__init__(clients, model, config, useAsyncClients)

    def trainAndTest(self, testDataset: DatasetInterface):
        roundsError = torch.zeros(self.rounds)

        for r in range(self.rounds):
            logPrint("Round... ", r)

            self._shareModelAndTrainOnClients()
            models = self._retrieveClientModelsDict()

            # Merge models
            chosen_clients = [self.clients[i] for i in self.chosen_indices]
            self.model = self.aggregate(chosen_clients, models)

            roundsError[r] = self.test(testDataset)

        return roundsError

    def __computeModelDistance(self, mOrig: nn.Module, mDest: nn.Module) -> Tensor:
        paramsDest = mDest.named_parameters()
        dictParamsDest = dict(paramsDest)
        paramsOrig = mOrig.named_parameters()
        d1 = torch.tensor([]).to(self.device)
        d2 = torch.tensor([]).to(self.device)
        for name1, param1 in paramsOrig:
            if name1 in dictParamsDest:
                d1 = torch.cat((d1, dictParamsDest[name1].data.view(-1)))
                d2 = torch.cat((d2, param1.data.view(-1)))
        sim: Tensor = torch.norm(d1 - d2, p=2)
        return sim


    def aggregate(self, clients: List[Client], models: List[nn.Module]) -> nn.Module:
        empty_model = deepcopy(self.model)

        userNo = len(clients)
        # Number of Byzantine workers to be tolerated
        f = int((userNo - 3) / 2)
        th = userNo - f - 2
        mk = userNo - f
        # Compute distances for all users
        scores = torch.zeros(userNo)

        for i in range(userNo): # Client1 is i
            distances = torch.zeros((userNo, userNo))

            for j in range(userNo): # Client 2 is j
                if i != j:
                    distance = self.__computeModelDistance(
                        models[i].to(self.device),
                        models[j].to(self.device),
                    )
                    distances[i][j] = distance
            dd = distances[i][:].sort()[0]
            dd = dd.cumsum(0)
            scores[i] = dd[th]

        _, idx = scores.sort()
        selected_users = idx[: mk - 1] + 1
        # logPrint("Selected users: ", selected_users)

        comb = 0.0
        for i in range(userNo):
            if i in selected_users:
                self._mergeModels(
                    models[i].to(self.device),
                    empty_model.to(self.device),
                    1 / mk,
                    comb,
                )
                comb = 1.0

        return empty_model

from experiment.DefaultExperimentConfiguration import DefaultExperimentConfiguration
from torch import nn, Tensor
from client import Client
from logger import logPrint
from typing import List
import torch
from aggregators.Aggregator import Aggregator
from datasetLoaders.DatasetInterface import DatasetInterface
from copy import deepcopy

# FEDERATED AVERAGING AGGREGATOR
class FAAggregator(Aggregator):
    def __init__(self, clients: List[Client], model: nn.Module, config:DefaultExperimentConfiguration, useAsyncClients:bool=False):
        super().__init__(clients, model, config, useAsyncClients)

    def trainAndTest(self, testDataset: DatasetInterface) -> Tensor:
        roundsError = torch.zeros(self.rounds)
        for r in range(self.rounds):
            logPrint("Round... ", r)
            self._shareModelAndTrainOnClients()
            models = self._retrieveClientModelsDict()
            # Merge models
            self.model = self.aggregate(self.clients, models)

        roundsError[r] = self.test(testDataset)

        return roundsError

    def aggregate(self, clients: List[Client], models: List[nn.Module]) -> nn.Module:
        empty_model = deepcopy(self.model)

        comb = 0.0
        for client in clients:
            self._mergeModels(
                models[client.id].to(self.device),
                empty_model.to(self.device),
                client.p,
                comb,
            )
            comb = 1.0

        return empty_model



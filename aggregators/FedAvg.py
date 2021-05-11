from logger import logPrint
from typing import List
import torch
from aggregators.Aggregator import Aggregator

# FEDERATED AVERAGING AGGREGATOR
class FAAggregator(Aggregator):
    def __init__(self, clients, model, rounds, device, useAsyncClients=False):
        super().__init__(clients, model, rounds, device, useAsyncClients)

    def trainAndTest(self, testDataset):
        roundsError = torch.zeros(self.rounds)
        for r in range(self.rounds):
            logPrint("Round... ", r)
            self._shareModelAndTrainOnClients()
            models = self._retrieveClientModelsDict()
            # Merge models
            comb = 0.0
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

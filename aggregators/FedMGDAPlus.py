from utils.typings import Errors
from experiment.AggregatorConfig import AggregatorConfig
from client import Client
from typing import List
from logger import logPrint
import torch
import copy
from aggregators.Aggregator import Aggregator
from torch import nn
import torch.optim as optim
import numpy as np


class FedMGDAPlusAggregator(Aggregator):
    """
    FedMGDA+ Aggregator

    Uses a Linear Layer to perform predictions on the weighting of the clients
    """

    def __init__(
        self,
        clients: List[Client],
        model: nn.Module,
        config: AggregatorConfig,
        useAsyncClients: bool = False,
    ):
        super().__init__(clients, model, config, useAsyncClients)
        self.numOfClients = len(clients)
        self.lambdaModel = nn.Parameter(torch.rand(self.numOfClients), requires_grad=True)
        for client in self.clients:
            self.lambdaModel[client.id - 1].data = torch.tensor(1.0)

        self.learningRate = 0.001
        self.lambdatOpt = optim.SGD([self.lambdaModel], lr=self.learningRate, momentum=0.5)

        # self.delta is going to store the values of the g_i according to the paper FedMGDA
        self.delta = copy.deepcopy(model) if model else None

    def trainAndTest(self, testDataset) -> Errors:
        roundsError = Errors(torch.zeros(self.rounds))
        for r in range(self.rounds):
            logPrint("Round... ", r)
            self._shareModelAndTrainOnClients()
            sentClientModels = self._retrieveClientModelsDict()

            self.previousGlobalModel = copy.deepcopy(self.model) if self.model else None

            paramsDelta = self.delta.named_parameters()
            deltaParams = dict(paramsDelta)

            loss = 0.0
            # reset the gradients
            self.lambdatOpt.zero_grad()

            for client in self.clients:
                clientModel = sentClientModels[client].named_parameters()
                clientParams = dict(clientModel)
                paramsUntrained = self.previousGlobalModel.named_parameters()
                # compute the delta which is the difference between each client parameter and previous global model
                for name, paramPreviousGlobal in paramsUntrained:
                    if name in deltaParams:
                        deltaParams[name].data.copy_(
                            clientParams[name].cpu().data - paramPreviousGlobal.cpu().data
                        )

                # compute the loss = labda_i * delta_i for each client i
                if not (self.lambdaModel[client.id - 1] == 0):
                    loss += torch.norm(
                        torch.mul(
                            nn.utils.parameters_to_vector(self.delta.cpu().parameters()),
                            self.lambdaModel[client.id - 1],
                        )
                    )

                else:
                    blocked = [id for (id, _) in self.maliciousBlocked + self.benignBlocked]
                    if client.id not in blocked:
                        self.handle_blocked(client, r)

            loss.backward()
            self.lambdatOpt.step()
            for g in self.lambdatOpt.param_groups:
                g["lr"] = g["lr"] * 0.7

            comb = 0.0
            extractedVectors = np.array(list(self.lambdaModel.data))
            extractedVectors[extractedVectors < 0.001] = 0
            extractedVectors /= np.sum(extractedVectors)

            self.lambdaModel.data = torch.tensor(extractedVectors)

            for client in self.clients:
                self._mergeModels(
                    sentClientModels[client].to(self.device),
                    self.model.to(self.device),
                    extractedVectors[client.id - 1],
                    comb,
                )

                comb = 1.0

            roundsError[r] = self.test(testDataset)

        return roundsError

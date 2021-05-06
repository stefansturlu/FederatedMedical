from client import Client
from typing import List
from logger import logPrint
import torch
import copy
from aggregators.Aggregator import Aggregator
import numpy as np
from torch import nn, Tensor, device
import torch.optim as optim


class FedMGDAPlusAggregator(Aggregator):
    def __init__(
        self,
        clients:List[Client],
        model:nn.Module,
        rounds:int,
        device:device,
        useAsyncClients:bool=False,
        learningRateStart:float=0.1,
        learningRateEnd:float=0.1,
    ):
        super().__init__(clients, model, rounds, device, useAsyncClients)
        self.numOfClients = len(clients)
        self.lambdaModel = nn.Parameter(torch.ones(self.numOfClients), requires_grad=True)
        self.LR1 = learningRateStart
        self.LR2 = learningRateEnd
        self.lambdatOpt = optim.SGD([self.lambdaModel], lr=self.LR1, momentum=0.5)

        # self.delta is going to store the values of the g_i according to the paper FedMGDA
        # More accurately, it stores the difference between the previous model params and
        # the clients' params
        self.delta = copy.deepcopy(model) if model else False

    # Needed for when we set the config innerLR
    def reinitialise(self, lr1: float, lr2: float) -> None:
        self.LR1 = lr1
        self.LR2 = lr2
        self.lambdatOpt = optim.SGD([self.lambdaModel], lr=lr1, momentum=0.5)

    def trainAndTest(self, testDataset) -> Tensor[float]:
        roundsError = torch.zeros(self.rounds)
        lrs = torch.linspace(self.LR1, self.LR2, self.rounds)

        for r in range(self.rounds):
            logPrint("Round... ", r)
            logPrint("LR - Current: %.3f" % lrs[r])
            self._shareModelAndTrainOnClients()
            sentClientModels = self._retrieveClientModelsDict()

            self.previousGlobalModel = copy.deepcopy(self.model) if self.model else False

            loss = 0.0
            # reset the gradients
            self.lambdatOpt.zero_grad()

            # Keeping track of the blocked clients each round to ensure their weighting remains at 0 always
            blocked_clients = []

            for idx, client in enumerate(self.clients):
                self.lambdatOpt.zero_grad()
                if (client.blocked):
                    blocked_clients.append(idx)
                    continue
                clientModel = sentClientModels[client].named_parameters()
                clientParams = dict(clientModel)
                previousGlobalParams = dict(self.previousGlobalModel.named_parameters())

                with torch.no_grad():
                    for name, param in self.delta.named_parameters():
                        param.copy_(torch.tensor(0))
                        if name in clientParams:
                            param.copy_(
                                torch.abs(clientParams[name].cpu().data - previousGlobalParams[name].cpu().data)
                            )

                # Compute the loss = lambda_i * delta_i for each client i
                # Normalise the data
                loss_bottom = self.lambdaModel.max() - self.lambdaModel.min()
                # Handle case when initialised (or unlucky)
                if (loss_bottom == 0):
                    loss_bottom = 1

                loss = torch.norm(
                    torch.mul(
                        nn.utils.parameters_to_vector(self.delta.cpu().parameters()),
                        self.lambdaModel[client.id - 1] / loss_bottom
                    )
                )

                # If the client is blocked, we don't want to learn from it
                if not (self.lambdaModel[client.id - 1] == 0):
                    loss.backward()
                    self.lambdatOpt.step()

            # Thresholding and Normalisation
            clientWeights = np.array(list(self.lambdaModel.data))
            clientWeights[blocked_clients] = 0
            # Setting to zero no matter what if negative
            # If the weight gets below 0 then we don't want to count the client
            # The min might not be zero and so that's why we just don't take the max for the bottom
            clientWeights[clientWeights <= 0] = 0

            for idx, weight in enumerate(clientWeights):
                client = self.clients[idx]
                if (weight == 0) and not client.blocked:
                    self.handle_blocked(client, r)


            self.lambdaModel.data = torch.tensor(clientWeights)
            normalisedClientWeights = clientWeights / np.sum(clientWeights)

            comb = 0.0

            for client in self.clients:
                self._mergeModels(
                    sentClientModels[client].to(self.device),
                    self.model.to(self.device),
                    normalisedClientWeights[client.id - 1],
                    comb,
                )

                comb = 1.0

            roundsError[r] = self.test(testDataset)


            # Increasing / Decreasing LR each global round
            for g in self.lambdatOpt.param_groups:
                g['lr'] = lrs[r]



        return roundsError

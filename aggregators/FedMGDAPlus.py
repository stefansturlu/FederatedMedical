from experiment.AggregatorConfig import AggregatorConfig
from client import Client
from typing import List
from logger import logPrint
import torch
import copy
from aggregators.Aggregator import Aggregator
from torch import nn, Tensor
import torch.optim as optim


class FedMGDAPlusAggregator(Aggregator):
    def __init__(
        self,
        clients:List[Client],
        model:nn.Module,
        config: AggregatorConfig,
        useAsyncClients:bool=False,
        learningRate:float=0.1,
    ):
        super().__init__(clients, model, config, useAsyncClients)
        self.numOfClients = len(clients)
        self.lambdaModel = nn.Parameter(torch.ones(self.numOfClients), requires_grad=True)
        self.LR = learningRate
        self.lambdatOpt = optim.SGD([self.lambdaModel], lr=self.LR1, momentum=0.5)

        # self.delta is going to store the values of the g_i according to the paper FedMGDA
        # More accurately, it stores the difference between the previous model params and
        # the clients' params
        # Using a model to do this is overly complex but it allows for the parameter names to
        # be kept (probably still a better way to do it)
        self.delta = copy.deepcopy(model) if model else False

        self.std_multiplier = 1.5

    # Needed for when we set the config innerLR
    def reinitialise(self, lr: float) -> None:
        self.LR = lr
        self.lambdatOpt = optim.SGD([self.lambdaModel], lr=lr, momentum=0.5)

    def trainAndTest(self, testDataset) -> Tensor:
        roundsError = torch.zeros(self.rounds)
        previousBlockedClients = []
        old_error = 100

        for r in range(self.rounds):
            logPrint("Round... ", r)
            self.round = r
            self._shareModelAndTrainOnClients()
            sentClientModels = self._retrieveClientModelsDict()

            self.previousGlobalModel = copy.deepcopy(self.model) if self.model else False

            self.model = self.aggregate(self.clients, sentClientModels)

            roundsError[r] = self.test(testDataset)


            blockedCheck = []
            for idx, client in enumerate(self.clients):
                if (client.blocked):
                    blockedCheck.append(idx)

            size = len(blockedCheck) - len(previousBlockedClients)


            if (size != 0):
                # Increasing / Decreasing LR each global round
                for g in self.lambdatOpt.param_groups:
                    g["lr"] *= 0.9**size
                    print(f"New LR: {g['lr']}")

                    self.std_multiplier *= 1.05**size
                    print(f"New std: {self.std_multiplier}")

            elif (old_error <= roundsError[r]):
                # Increasing / Decreasing LR each global round
                for g in self.lambdatOpt.param_groups:
                    g["lr"] *= 1.3
                    print(f"New LR: {g['lr']}")

                    self.std_multiplier /= 1.05
                    print(f"New std: {self.std_multiplier}")
            else:
                for g in self.lambdatOpt.param_groups:
                    g["lr"] *= 0.9

            old_error = roundsError[r]

            previousBlockedClients = blockedCheck


        return roundsError


    def aggregate(self, clients: List[Client], models: List[nn.Module]) -> nn.Module:
        empty_model = copy.deepcopy(self.model)

        loss = 0.0
        # reset the gradients
        self.lambdatOpt.zero_grad()

        # Keeping track of the blocked clients each round to ensure their weighting remains at 0 always
        blocked_clients = []

        for idx, client in enumerate(clients):
            self.lambdatOpt.zero_grad()
            if (client.blocked):
                blocked_clients.append(idx)
                continue
            clientModel = models[client.id].named_parameters()
            clientParams = dict(clientModel)
            previousGlobalParams = dict(self.previousGlobalModel.named_parameters())

            with torch.no_grad():
                for name, param in self.delta.named_parameters():
                    param.copy_(torch.tensor(0))
                    if name in clientParams and name in previousGlobalParams:
                        if not torch.any(clientParams[name].isnan()):
                            param.copy_(
                                torch.abs(clientParams[name].data - previousGlobalParams[name].data)
                            )

            # Compute the loss = lambda_i * delta_i for each client i
            # Normalise the data
            loss_bottom = self.lambdaModel.max()
            if (loss_bottom == 0):
                loss_bottom = 1

            loss = torch.norm(
                torch.mul(
                    nn.utils.parameters_to_vector(self.delta.parameters()),
                    self.lambdaModel[client.id] / loss_bottom
                )
            )

            # If the client is blocked, we don't want to learn from it
            if not (self.lambdaModel[client.id] == 0):
                loss.backward()
                self.lambdatOpt.step()

        # Thresholding and Normalisation
        clientWeights = self.lambdaModel.data
        clientWeights[blocked_clients] = 0
        # Setting to zero no matter what if negative
        # If the weight gets below 0 then we don't want to count the client
        # The min might not be zero and so that's why we just don't take the max for the bottom
        clientWeights[clientWeights <= 0] = 0

        vals = clientWeights[torch.nonzero(clientWeights)]
        cutoff = vals.mean() - (self.std_multiplier*vals.std())

        clientWeights[clientWeights < cutoff] = 0

        for idx, weight in enumerate(clientWeights):
            client = clients[idx]
            if (weight == 0) and not client.blocked:
                self.handle_blocked(client, self.round)


        self.lambdaModel.data = clientWeights
        normalisedClientWeights = clientWeights / clientWeights.sum()


        comb = 0.0

        for client in clients:
            self._mergeModels(
                models[client.id].to(self.device),
                empty_model.to(self.device),
                normalisedClientWeights[client.id],
                comb,
            )

            comb = 1.0

        return empty_model
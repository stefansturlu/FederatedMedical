from logger import logPrint
import torch
import copy
from aggregators.Aggregator import Aggregator
import numpy as np
from torch import nn


class FedMGDAPlusAggregator(Aggregator):
    def __init__(
        self,
        clients,
        model,
        rounds,
        device,
        useAsyncClients=False,
        # Large learning rate needed for proper differentiation due to small changes for 1 layer
        learningRate=0.1,
        # Should not be hard set, should be based on number of users
        # ~0.033333 for 30 users (should not get higher than this for obvious reasons)
        threshold=0.0001,
    ):
        super().__init__(clients, model, rounds, device, useAsyncClients)
        self.numOfClients = len(clients)
        self.lambdaModel = nn.Parameter(torch.ones(self.numOfClients), requires_grad=True)
        self.learningRate = learningRate
        self.lambdatOpt = torch.optim.SGD([self.lambdaModel], lr=self.learningRate, momentum=0.5)

        # self.delta is going to store the values of the g_i according to the paper FedMGDA
        # More accurately, it stores the difference between the previous model params and
        # the clients' params
        self.delta = copy.deepcopy(model) if model else False
        self.threshold = threshold

    # Needed for when we set the config innerLR and threshold
    def reinitialise(self, lr: float, threshold: float):
        self.learningRate = lr
        self.lambdatOpt = torch.optim.SGD([self.lambdaModel], lr=lr, momentum=0.5)
        self.threshold = threshold

    def trainAndTest(self, testDataset):
        print(self.learningRate)
        roundsError = torch.zeros(self.rounds)

        for r in range(self.rounds):
            logPrint("Round... ", r)
            self._shareModelAndTrainOnClients()
            sentClientModels = self._retrieveClientModelsDict()

            self.previousGlobalModel = copy.deepcopy(self.model) if self.model else False

            loss = 0.0
            # reset the gradients
            self.lambdatOpt.zero_grad()

            any_blocked = False

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

                # compute the loss = lambda_i * delta_i for each client i
                # print("^^^^^^^^^^^^^^^^^^^^")
                loss_bottom = self.lambdaModel.max() - self.lambdaModel.min()
                if (loss_bottom == 0):
                    loss_bottom = 1
                loss = torch.norm(
                    torch.mul(
                        nn.utils.parameters_to_vector(self.delta.cpu().parameters()),
                        self.lambdaModel[client.id - 1] / loss_bottom
                    )
                )
                print(loss)
                if (self.lambdaModel[client.id - 1] == 0):
                    # if not client.blocked:
                    #     self.handle_blocked(client, r)
                    #     # any_blocked = True
                    pass

            # if (any_blocked):
                # oldWeights = np.array(list(self.lambdaModel.data))
                # print(oldWeights)
                # # indices = oldWeights < self.threshold
                # indices = [i for i, client in enumerate(self.clients) if client.blocked]
                # print(indices)
                # print(oldWeights[indices])
                # self.lambdaModel = nn.Parameter(torch.ones(self.numOfClients), requires_grad=True)
                # self.lambdatOpt = torch.optim.SGD([self.lambdaModel], lr=self.learningRate, momentum=0.5)

                # newWeights = np.array(list(self.lambdaModel.data))
                # newWeights[indices] = 0
                # self.lambdaModel.data = torch.tensor(newWeights)

                else:
                    loss.backward()
                    self.lambdatOpt.step()
            # for g in self.lambdatOpt.param_groups:
            #     g["lr"] = g["lr"] * 0.7

            # Thresholding and Normalisation
            clientWeights = np.array(list(self.lambdaModel.data))
            print(clientWeights)
            clientWeights[blocked_clients] = 0

            # norm = TODO

            clientWeights[clientWeights < self.threshold] = 0

            for idx, weight in enumerate(clientWeights):
                client = self.clients[idx]
                if (weight == 0) and not client.blocked:
                    self.handle_blocked(client, r)


            self.lambdaModel.data = torch.tensor(clientWeights)
            normalisedClientWeights = clientWeights / np.sum(clientWeights)
            print(normalisedClientWeights)
            # self.lambdaModel.data = torch.tensor(normalisedClientWeights)

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


        return roundsError

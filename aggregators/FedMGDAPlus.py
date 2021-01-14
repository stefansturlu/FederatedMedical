from logger import logPrint
from typing import List
import torch
import copy
from aggregators.Aggregator import Aggregator
import numpy as np
from torch import nn


class FedMGDAPlusAggregator(Aggregator):
    def __init__(self, clients, model, rounds, device, useAsyncClients=False):
        super().__init__(clients, model, rounds, device, useAsyncClients)
        self.numOfClients = len(clients)
        self.lambdaModel = nn.Parameter(torch.rand(self.numOfClients), requires_grad=True)
        for client in self.clients:
            self.lambdaModel[client.id - 1].data = torch.tensor(1.0)
        # self.learningRate = 0.0001
        self.learningRate = 0.001
        self.lambdatOpt = torch.optim.SGD([self.lambdaModel], lr=self.learningRate, momentum=0.5)
        # self.delta is going to store the values of the g_i according to the paper FedMGDA
        self.delta = copy.deepcopy(model) if model else False

    def trainAndTest(self, testDataset):
        roundsError = torch.zeros(self.rounds)
        for r in range(self.rounds):
            logPrint("Round... ", r)
            self._shareModelAndTrainOnClients()
            sentClientModels = self._retrieveClientModelsDict()

            self.previousGlobalModel = copy.deepcopy(self.model) if self.model else False

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
                        deltaParams[name].data.copy_(clientParams[name].cpu().data - paramPreviousGlobal.cpu().data)

                # compute the loss = labda_i * delta_i for each client i
                if not (self.lambdaModel[client.id - 1] == 0):
                    loss += torch.norm(
                        torch.mul(
                            nn.utils.parameters_to_vector(self.delta.cpu().parameters()),
                            self.lambdaModel[client.id - 1],
                        )
                    )

                else:
                    if not client.blocked:
                        self.handle_blocked(client, r)

                #
                # print(client.id)
                # print(torch.norm(nn.utils.parameters_to_vector(self.delta.parameters())))

                # print(torch.norm(torch.mul(nn.utils.parameters_to_vector(self.delta.parameters()),
                #                              self.lambdaModel[client.id - 1])))
                # print(self.lambdaModel[client.id - 1])

            loss.backward()
            # print(self.lambdaModel.grad)
            # print(self.lambdaModel)
            self.lambdatOpt.step()
            for g in self.lambdatOpt.param_groups:
                g["lr"] = g["lr"] * 0.7

            comb = 0.0
            extractedVectors = np.array(list(self.lambdaModel.data))
            extractedVectors[extractedVectors < 0.001] = 0
            extractedVectors /= np.sum(extractedVectors)
            # print(extractedVectors)

            self.lambdaModel.data = torch.tensor(extractedVectors)

            for client in self.clients:
                self._mergeModels(
                    sentClientModels[client].to(self.device),
                    self.model.to(self.device),
                    extractedVectors[client.id - 1],
                    comb,
                )
                client.lambdaList.append(extractedVectors[client.id - 1])

                comb = 1.0

            roundsError[r] = self.test(testDataset)

        # for client in self.clients:
        #     logPrint("Client ", client.id, " MU ", client.lambdaList)

        return roundsError

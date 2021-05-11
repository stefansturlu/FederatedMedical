from client import Client
from logger import logPrint
from typing import List
import torch
import copy
import numpy as np
from aggregators.Aggregator import Aggregator
from torch import nn
from scipy.stats import beta

# ADAPTIVE FEDERATED AVERAGING
class AFAAggregator(Aggregator):
    def __init__(self, clients, model, rounds, device, useAsyncClients=False):
        super().__init__(clients, model, rounds, device, useAsyncClients)
        self.xi = 2
        self.deltaXi = 0.25

    def trainAndTest(self, testDataset):
        roundsError = torch.zeros(self.rounds)

        for r in range(self.rounds):

            logPrint("Round... ", r)

            for client in self.clients:
                broadcastModel = copy.deepcopy(self.model)
                client.updateModel(broadcastModel)
                if not client.blocked:
                    error, pred = client.trainModel()

            models = self._retrieveClientModelsDict()

            badCount = 2
            slack = self.xi
            while badCount != 0:
                pT_epoch = 0.0
                for client in self.clients:
                    if self.notBlockedNorBadUpdate(client):
                        client.pEpoch = client.n * client.score
                        pT_epoch = pT_epoch + client.pEpoch

                for client in self.clients:
                    if self.notBlockedNorBadUpdate(client):
                        client.pEpoch = client.pEpoch / pT_epoch

                comb = 0.0
                for client in self.clients:
                    if self.notBlockedNorBadUpdate(client):
                        self._mergeModels(
                            models[client.id].to(self.device),
                            self.model.to(self.device),
                            client.pEpoch,
                            comb,
                        )
                        comb = 1.0

                sim = []
                for client in self.clients:
                    if self.notBlockedNorBadUpdate(client):
                        client.sim = self.__modelSimilarity(self.model, models[client.id])
                        sim.append(np.asarray(client.sim.to("cpu")))
                        # logPrint("Similarity user ", u.id, ": ", u.sim)

                sim = np.asarray(sim)

                meanS = np.mean(sim)
                medianS = np.median(sim)
                desvS = np.std(sim)

                if meanS < medianS:
                    th = medianS - slack * desvS
                else:
                    th = medianS + slack * desvS

                slack += self.deltaXi

                badCount = 0
                for client in self.clients:
                    if not client.badUpdate:
                        # Malicious self.clients are below the threshold
                        if meanS < medianS:
                            if client.sim < th:
                                # logPrint("Type1")
                                # logPrint("Bad update from user ", u.id)
                                client.badUpdate = True
                                badCount += 1
                                # Malicious self.clients are above the threshold
                        else:
                            if client.sim > th:
                                client.badUpdate = True
                                badCount += 1

            pT = 0.0
            for client in self.clients:
                if not client.blocked:
                    self.updateUserScore(client)
                    client.blocked = self.checkBlockedUser(client.alpha, client.beta)
                    if client.blocked:
                        self.handle_blocked(client, r)
                    else:
                        client.p = client.n * client.score
                        pT = pT + client.p

            for client in self.clients:
                client.p = client.p / pT
                # logPrint("Weight user", u.id, ": ", round(u.p,3))

            # Update model with the updated scores
            pT_epoch = 0.0
            for client in self.clients:
                if self.notBlockedNorBadUpdate(client):
                    client.pEpoch = client.n * client.score
                    pT_epoch = pT_epoch + client.pEpoch

            for client in self.clients:
                if self.notBlockedNorBadUpdate(client):
                    client.pEpoch = client.pEpoch / pT_epoch
            # logPrint("Updated scores:{}".format([client.pEpoch for client in self.clients]))
            comb = 0.0
            for client in self.clients:
                if self.notBlockedNorBadUpdate(client):
                    self._mergeModels(
                        models[client.id].to(self.device),
                        self.model.to(self.device),
                        client.pEpoch,
                        comb,
                    )
                    comb = 1.0

            # Reset badUpdate variable
            for client in self.clients:
                if not client.blocked:
                    client.badUpdate = False

            roundsError[r] = self.test(testDataset)

        return roundsError

    def __modelSimilarity(self, mOrig, mDest):
        cos = nn.CosineSimilarity(0)

        d2 = torch.tensor([]).to(self.device)
        d1 = torch.tensor([]).to(self.device)

        paramsOrig = mOrig.named_parameters()
        paramsDest = mDest.named_parameters()
        dictParamsDest = dict(paramsDest)

        for name1, param1 in paramsOrig:
            if name1 in dictParamsDest:
                d1 = torch.cat((d1, dictParamsDest[name1].data.view(-1)))
                d2 = torch.cat((d2, param1.data.view(-1)))
                # d2 = param1.data
                # sim = cos(d1.view(-1),d2.view(-1))
                # logPrint(name1,param1.size())
                # logPrint("Similarity: ",sim)
        sim = cos(d1, d2)
        return sim

    @staticmethod
    def checkBlockedUser(a, b, th=0.95):
        # return beta.cdf(0.5, a, b) > th
        s = beta.cdf(0.5, a, b)
        blocked = False
        if s > th:
            blocked = True
        return blocked

    @staticmethod
    def updateUserScore(client):
        if client.badUpdate:
            client.beta += 1
        else:
            client.alpha += 1
        client.score = client.alpha / client.beta

    @staticmethod
    def notBlockedNorBadUpdate(client):
        return client.blocked == False | client.badUpdate == False

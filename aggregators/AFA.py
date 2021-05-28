from random import uniform
from utils.typings import Errors
from experiment.AggregatorConfig import AggregatorConfig
from client import Client
from logger import logPrint
from typing import List
import torch
import copy
from aggregators.Aggregator import Aggregator
from torch import nn, Tensor
from scipy.stats import beta
import scipy
from datasetLoaders.DatasetInterface import DatasetInterface

# ADAPTIVE FEDERATED AVERAGING
class AFAAggregator(Aggregator):
    def __init__(
        self,
        clients: List[Client],
        model: nn.Module,
        config: AggregatorConfig,
        useAsyncClients: bool = False,
    ):
        super().__init__(clients, model, config, useAsyncClients)
        self.xi: float = 2
        self.deltaXi: float = 0.25

    def trainAndTest(self, testDataset: DatasetInterface) -> Errors:
        roundsError = Errors(torch.zeros(self.rounds))

        for r in range(self.rounds):

            logPrint("Round... ", r)

            if self.config.privacyAmplification:
                self.chosen_indices = [
                    i for i in range(len(self.clients)) if uniform(0, 1) <= self.config.amplificationP
                ]

            chosen_clients = [self.clients[i] for i in self.chosen_indices]

            for client in chosen_clients:
                broadcastModel = copy.deepcopy(self.model)
                client.updateModel(broadcastModel)
                if not client.blocked:
                    error, pred = client.trainModel()

            models = self._retrieveClientModelsDict()

            self.model = self.aggregate(chosen_clients, models)
            self.round = r

            roundsError[r] = self.test(testDataset)

        return roundsError

    def __modelSimilarity(self, mOrig: nn.Module, mDest: nn.Module) -> Tensor:
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
        sim: Tensor = cos(d1, d2)
        return sim

    @staticmethod
    def checkBlockedUser(a: float, b: float, th: float = 0.95) -> bool:
        return beta.cdf(0.5, a, b) > th

    @staticmethod
    def updateUserScore(client: Client) -> None:
        if client.badUpdate:
            client.beta += 1
        else:
            client.alpha += 1
        client.score = client.alpha / client.beta

    @staticmethod
    def notBlockedNorBadUpdate(client: Client) -> bool:
        return client.blocked == False | client.badUpdate == False

    def aggregate(self, clients: List[Client], models: List[nn.Module]) -> nn.Module:
        empty_model = copy.deepcopy(self.model)

        badCount: int = 2
        slack = self.xi
        while badCount != 0:
            pT_epoch = 0.0
            for client in clients:
                if self.notBlockedNorBadUpdate(client):
                    client.pEpoch = client.n * client.score
                    pT_epoch = pT_epoch + client.pEpoch

            for client in clients:
                if self.notBlockedNorBadUpdate(client):
                    client.pEpoch = client.pEpoch / pT_epoch

            comb = 0.0
            for i, client in enumerate(clients):
                if self.notBlockedNorBadUpdate(client):
                    self._mergeModels(
                        models[i].to(self.device),
                        empty_model.to(self.device),
                        client.pEpoch,
                        comb,
                    )
                    comb = 1.0

            sim = torch.zeros(len(clients)).to(self.device)
            for i, client in enumerate(clients):
                if self.notBlockedNorBadUpdate(client):
                    client.sim = self.__modelSimilarity(empty_model, models[i])
                    sim[i] = client.sim
                    # logPrint("Similarity user ", u.id, ": ", u.sim)

            meanS = torch.mean(sim)
            medianS = torch.median(sim)
            desvS = torch.std(sim)

            if meanS < medianS:
                th = medianS - slack * desvS
            else:
                th = medianS + slack * desvS

            slack += self.deltaXi

            badCount = 0
            for client in clients:
                if not client.badUpdate:
                    # Malicious clients are below the threshold
                    if meanS < medianS:
                        if client.sim < th:
                            # logPrint("Type1")
                            # logPrint("Bad update from user ", u.id)
                            client.badUpdate = True
                            badCount += 1
                            # Malicious clients are above the threshold
                    else:
                        if client.sim > th:
                            client.badUpdate = True
                            badCount += 1

        pT = 0.0
        for client in clients:
            if not client.blocked:
                self.updateUserScore(client)
                client.blocked = self.checkBlockedUser(client.alpha, client.beta)
                if client.blocked:
                    self.handle_blocked(client, self.round)
                else:
                    client.p = client.n * client.score
                    pT = pT + client.p

        for client in clients:
            client.p = client.p / pT
            # logPrint("Weight user", u.id, ": ", round(u.p,3))

        # Update model with the updated scores
        pT_epoch = 0.0
        for client in clients:
            if self.notBlockedNorBadUpdate(client):
                client.pEpoch = client.n * client.score
                pT_epoch = pT_epoch + client.pEpoch

        for client in clients:
            if self.notBlockedNorBadUpdate(client):
                client.pEpoch = client.pEpoch / pT_epoch
        # logPrint("Updated scores:{}".format([client.pEpoch for client in clients]))
        comb = 0.0
        for i, client in enumerate(clients):
            if self.notBlockedNorBadUpdate(client):
                self._mergeModels(
                    models[i].to(self.device),
                    empty_model.to(self.device),
                    client.pEpoch,
                    comb,
                )
                comb = 1.0

        # Reset badUpdate variable
        for client in clients:
            if not client.blocked:
                client.badUpdate = False

        return empty_model

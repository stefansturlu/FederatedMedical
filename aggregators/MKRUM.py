from logger import logPrint
from typing import List
import torch
from aggregators.Aggregator import Aggregator


class MKRUMAggregator(Aggregator):
    def __init__(self, clients, model, rounds, device, useAsyncClients=False):
        super().__init__(clients, model, rounds, device, useAsyncClients)

    def trainAndTest(self, testDataset):
        userNo = len(self.clients)
        # Number of Byzantine workers to be tolerated
        f = int((userNo - 3) / 2)
        th = userNo - f - 2
        mk = userNo - f

        roundsError = torch.zeros(self.rounds)

        for r in range(self.rounds):
            logPrint("Round... ", r)

            self._shareModelAndTrainOnClients()

            # Compute distances for all users
            scores = torch.zeros(userNo)
            models = self._retrieveClientModelsDict()
            for client in self.clients:
                distances = torch.zeros((userNo, userNo))
                for client2 in self.clients:
                    if client.id != client2.id:
                        distance = self.__computeModelDistance(
                            models[client.id].to(self.device),
                            models[client2.id].to(self.device),
                        )
                        distances[client.id - 1][client2.id - 1] = distance
                dd = distances[client.id - 1][:].sort()[0]
                dd = dd.cumsum(0)
                scores[client.id - 1] = dd[th]

            _, idx = scores.sort()
            selected_users = idx[: mk - 1] + 1
            # logPrint("Selected users: ", selected_users)

            comb = 0.0
            for client in self.clients:
                if client.id in selected_users:
                    self._mergeModels(
                        models[client.id].to(self.device),
                        self.model.to(self.device),
                        1 / mk,
                        comb,
                    )
                    comb = 1.0

            roundsError[r] = self.test(testDataset)

        return roundsError

    def __computeModelDistance(self, mOrig, mDest):
        paramsDest = mDest.named_parameters()
        dictParamsDest = dict(paramsDest)
        paramsOrig = mOrig.named_parameters()
        d1 = torch.tensor([]).to(self.device)
        d2 = torch.tensor([]).to(self.device)
        for name1, param1 in paramsOrig:
            if name1 in dictParamsDest:
                d1 = torch.cat((d1, dictParamsDest[name1].data.view(-1)))
                d2 = torch.cat((d2, param1.data.view(-1)))
        sim = torch.norm(d1 - d2, p=2)
        return sim

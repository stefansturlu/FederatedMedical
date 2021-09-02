from utils.typings import Errors
from experiment.AggregatorConfig import AggregatorConfig
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim.swa_utils import AveragedModel, SWALR
from torch.optim.lr_scheduler import CosineAnnealingLR
from client import Client
from logger import logPrint
from typing import List
import torch
from torch.distributions.dirichlet import Dirichlet
from torch.distributions.normal import Normal
from aggregators.Aggregator import Aggregator
from datasetLoaders.DatasetInterface import DatasetInterface
from torch.utils.data import DataLoader
from copy import deepcopy
from sklearn.metrics import confusion_matrix
from utils.KnowledgeDistiller import KnowledgeDistiller


class FedAFABEAggregator(Aggregator):
    """
    A novel aggregator which combines AFA and FedBE (FedAFABE), which fits a Dirichlet distribution to the models based on an alpha score, samples weighted combinations of models from the distribution and uses Knowledge Distillation to combine the ensemble into a global model.
    """

    def __init__(
        self,
        clients: List[Client],
        model: nn.Module,
        config: AggregatorConfig,
        useAsyncClients: bool = False,
    ):
        super().__init__(clients, model, config, useAsyncClients)

        logPrint("INITIALISING FedABE Aggregator!")
        # Unlabelled data which will be used in Knowledge Distillation
        self.distillationData = None  # data is loaded in __runExperiment function
        self.true_labels = None
        self.sampleSize = config.sampleSize
        self.method = config.samplingMethod
        self.samplingAlpha = config.samplingDirichletAlpha

        self.xi: float = config.xi
        self.deltaXi: float = config.deltaXi
        self.pseudolabelMethod = "medlogits"

    def trainAndTest(self, testDataset: DatasetInterface) -> Errors:
        roundsError = Errors(torch.zeros(self.rounds))
        for r in range(self.rounds):
            logPrint("Round... ", r)
            self._shareModelAndTrainOnClients()
            models = self._retrieveClientModelsDict()

            # Merge models
            chosen_clients = [self.clients[i] for i in self.chosen_indices]
            self.model = self.aggregate(chosen_clients, models)

            roundsError[r] = self.test(testDataset)

        return roundsError

    def aggregate(self, clients: List[Client], models: List[nn.Module]) -> nn.Module:

        logPrint(f"Step 1 of FedABE: Calculating client scores based on models.")
        # IDEA: Use server set predictions to compare similarities, instead of model weights. Maybe even both?
        self._updateClientScores(clients, models)

        # STEP 2: Construct distribution of models from which to sample, and sample M models
        logPrint(
            f"Step 2 of FedABE: Constructing distribution from {len(models)} models and sampling {self.sampleSize} models."
        )
        ensemble = self._sampleModels(clients, models, self.method)

        if self.true_labels is None:
            self.true_labels = self.distillationData.labels

        kd = KnowledgeDistiller(self.distillationData, method=self.pseudolabelMethod)

        ensembleError = 100 * (1 - self.ensembleAccuracy(kd._pseudolabelsFromEnsemble(ensemble)))
        modelsError = 100 * (1 - self.ensembleAccuracy(kd._pseudolabelsFromEnsemble(models)))
        logPrint(
            f"Step 2 of FedBE: Distilling knowledge (ensemble error: {ensembleError:.2f} %, models error: {modelsError:.2f})"
        )

        avg_model = self._averageModel(models, clients)
        # avg_model = self._medianModel(models)
        # avg_model = self._averageModel(ensemble)
        avg_model = kd.distillKnowledge(ensemble, avg_model)

        return avg_model

    def _updateClientScores(self, clients: List[Client], models: List[nn.Module]) -> None:
        """
        # Client scores can be kept in either .score, .pEpoch, or in .alpha & .beta
        # Might be wise to figure out what to use for probability weighing. Probably .score
        # In AFA.py:
            .p = proportion of data
            .pEpoch = p_{k_t} * n_k / N, N = sum(p_{k_t}*n_k)
            .score = p_{k_t}, i.e. alpha / (alpha+beta)
            .alpha, .beta
        """
        empty_model = deepcopy(self.model)
        self.renormalise_weights(clients)

        badCount = 1
        slack = self.xi

        # This while loop calculates number of bad updates
        while badCount != 0:
            pT_epoch = 0.0

            # Recalculate pEpoch
            for client in clients:
                if not (client.blocked | client.badUpdate):
                    client.pEpoch = client.n * client.score
                    pT_epoch += client.pEpoch
                else:
                    client.pEpoch = 0  # To prevent it affecting aggregated models
            # normalise pEpoch
            for client in clients:
                if not (client.blocked | client.badUpdate):
                    client.pEpoch /= pT_epoch

            # Combine good models using pEpoch as weights
            weights = [c.pEpoch for c in clients]
            empty_model = self._weightedAverageModel(models, weights)

            # Calculate similarities
            sim = torch.zeros(len(clients)).to(self.device)
            for i, client in enumerate(clients):
                if not (client.blocked | client.badUpdate):
                    client.sim = self.__modelSimilarity(empty_model, models[i])
                    sim[i] = client.sim

            # Calculate mean, median and std dev
            meanS = torch.mean(sim)
            medianS = torch.median(sim)
            stdevS = torch.std(sim)

            if meanS < medianS:
                th = medianS - slack * stdevS
            else:
                th = medianS + slack * stdevS
            slack += self.deltaXi

            badCount = 0
            for client in clients:
                if not client.badUpdate:
                    # Malicious clients are below the threshold
                    if meanS < medianS:
                        if client.sim < th:
                            client.badUpdate = True
                            badCount += 1
                    # Malicious clients are above the threshold
                    else:
                        if client.sim > th:
                            client.badUpdate = True
                            badCount += 1

        # Update scores for clients. For now: No blocking.
        pT = 0.0
        for client in clients:
            # update user score (alpha and beta)
            if client.badUpdate:
                client.beta += 1
            else:
                client.alpha += 1
            client.score = client.alpha / client.beta

            # no blocking for now
            client.p = client.n * client.score
            pT = pT + client.p

        # Normalise client's actual weighting
        for client in clients:
            client.p /= pT

        # Update model's epoch weights with the updated scores
        pT_epoch = 0.0
        for client in clients:
            if not (client.blocked | client.badUpdate):
                client.pEpoch = client.n * client.score
                pT_epoch += client.pEpoch

        for client in clients:
            if not (client.blocked | client.badUpdate):
                client.pEpoch /= pT_epoch

        print("Client scores:", [c.score for c in clients])
        # Now all the client weights (alphas, betas, p, pEpoch) have been updated and can be used for sampling
        return None

    def _sampleModels(
        self, clients: List[Client], models: List[nn.Module], method="dirichlet"
    ) -> List[nn.Module]:
        """
        Sampling models using Gaussian or Dirichlet distributiton. Dirichlet distribution can be used client-wise or elementwise

        Parameters:
        clients: List of clients.
        models: List of the clients' models.
        """

        self.renormalise_weights(clients)
        M = self.sampleSize
        sampled_models = [deepcopy(self.model) for _ in range(M)]
        client_p = torch.tensor([c.p for c in clients])

        if method == "dirichlet_elementwise":
            logPrint("Sampling using Dirichlet method elementwise")

            client_model_dicts = [m.state_dict() for m in models]

            for name1, param1 in self.model.named_parameters():
                x = torch.stack(
                    [c[name1] for c in client_model_dicts]
                )  # 30 x 512 x 784 or 30 x 512

                # Fit a diagonal gaussian distribution to clients
                alphas = client_p * len(client_p) * self.samplingAlpha
                d = Dirichlet(alphas)

                # Sample M weights for each parameter
                sample_shape = [M] + list(x[0].shape)  # M x 512 x 784
                weights = d.sample(sample_shape)  # M x 512 x 784 x 30
                perm = [0] + [
                    (i - 1) % (len(weights.shape) - 1) + 1 for i in range(len(weights.shape) - 1)
                ]
                weights = weights.permute(*perm).to(self.device)  # M x 30 x 512 x 784

                # Compute M linear combination of client models
                samp = (weights * x.unsqueeze(0)).sum(dim=1)

                # Update each model in ensemble in-place
                for i, e in enumerate(sampled_models):
                    params_e = e.state_dict()
                    params_e[name1].data.copy_(samp[i])

        if method == "dirichlet":
            logPrint("Sampling using Dirichlet method client-wise")
            # Sample weights for weighted average of client models
            # using a symmetrical Dirichlet distribution
            alphas = client_p * len(client_p) * self.samplingAlpha
            print("Dirichlet alpha:", alphas)
            d = Dirichlet(alphas)
            sample = d.sample([M])  # Shape: M x len(models)

            # Compute weighted averages based on Dirichlet sample
            for i, s_model in enumerate(sampled_models):
                comb = 0.0
                for j, c_model in enumerate(models):
                    # _mergeModels updates s_model in-place
                    self._mergeModels(
                        c_model.to(self.device),
                        s_model.to(self.device),
                        sample[i, j],
                        comb,
                    )
                    comb = 1.0

        return sampled_models

    def ensembleAccuracy(self, pseudolabels):
        _, predLabels = torch.max(pseudolabels, dim=1)
        mconf = confusion_matrix(self.true_labels.cpu(), predLabels.cpu())
        return 1.0 * mconf.diagonal().sum() / len(self.distillationData)

    def __modelSimilarity(self, mOrig: nn.Module, mDest: nn.Module) -> torch.Tensor:
        """
        Calculates model similarity based on the Cosine Similarity metric.
        Flattens the models into tensors before doing the comparison.
        """
        cos = nn.CosineSimilarity(0)
        d1 = nn.utils.parameters_to_vector(mOrig.parameters())
        d2 = nn.utils.parameters_to_vector(mDest.parameters())
        sim: torch.Tensor = cos(d1, d2)
        return sim

    @staticmethod
    def requiresData():
        """
        Returns boolean value depending on whether the aggregation method requires server data or not.
        """
        return True

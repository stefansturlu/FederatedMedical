import copy
from typing import Optional, Type
import torch
import torch.nn as nn
import torch.optim as optim
from torch import Tensor, cuda

from torch.utils.data import DataLoader
from numpy import clip, percentile

from scipy.stats import laplace

from logger import logPrint


class Client:
    """ An internal representation of a client """

    def __init__(
        self,
        epochs,
        batchSize,
        learningRate,
        trainDataset,
        p,
        idx,
        useDifferentialPrivacy,
        releaseProportion,
        epsilon1,
        epsilon3,
        needClip,
        clipValue,
        device,
        Optimizer,
        Loss,
        needNormalization,
        byzantine=None,
        flipping=None,
        freeRiding=False,
        model: Optional[nn.Module] = None,
        alpha=3.0,
        beta=3.0,
    ):

        self.name: str = "client" + str(idx)
        self.device: torch.device = device

        self.model: nn.Module = model
        self.trainDataset = trainDataset
        self.dataLoader = DataLoader(self.trainDataset, batch_size=batchSize, shuffle=True)
        self.n: int = len(trainDataset)  # Number of training points provided
        self.p: float = p  # Contribution to the overall model
        self.id: int = idx  # ID for the user
        self.byz: bool = byzantine  # Boolean indicating whether the user is faulty or not
        self.flip: bool = flipping  # Boolean indicating whether the user is malicious or not (label flipping attack)
        self.free: bool = freeRiding  # Boolean indicating whether the user is a free-rider or not

        # Used for computing dW, i.e. the change in model before
        # and after client local training, when DP is used
        self.untrainedModel: nn.Module = copy.deepcopy(model).to("cpu") if model else None

        # Used for free-riders delta weights attacks
        self.prev_model: nn.Module = None

        self.opt: optim.Optimizer = None
        self.sim: Tensor = None
        self.loss = None
        self.Loss = Loss
        self.Optimizer: Type[optim.Optimizer] = Optimizer
        self.pEpoch: float = None
        self.badUpdate: bool = False
        self.epochs: int = epochs
        self.batchSize: int = batchSize

        self.learningRate: float = learningRate
        self.momentum: float = 0.9

        # AFA Client params
        self.alpha: float = alpha
        self.beta: float = beta
        self.score: float = alpha / beta
        self.blocked: bool = False

        # DP parameters
        self.useDifferentialPrivacy = useDifferentialPrivacy
        self.epsilon1 = epsilon1
        self.epsilon3 = epsilon3
        self.needClip = needClip
        self.clipValue = clipValue
        self.needNormalization = needNormalization
        self.releaseProportion = releaseProportion

        # FedMGDA+ params

    def updateModel(self, model: nn.Module) -> None:
        """
        Updates the client with the new model and re-initialise the optimiser
        """
        self.prev_model = copy.deepcopy(self.model)
        self.model = model.to(self.device)
        if self.Optimizer == optim.SGD:
            self.opt = self.Optimizer(
                self.model.parameters(), lr=self.learningRate, momentum=self.momentum
            )
        else:
            self.opt = self.Optimizer(self.model.parameters(), lr=self.learningRate)
        self.loss = self.Loss()
        self.untrainedModel = copy.deepcopy(model)
        cuda.empty_cache()

    def trainModel(self):
        """
        Trains the client's model unless the client is a free-rider
        """
        if self.free:
            # If the use is a free rider then they won't have any data to train on (theoretically)
            # However, we have to initialise the grad weights and the only way I know to do that is to train
            return None, None

        self.model = self.model.to(self.device)
        for i in range(self.epochs):
            for x, y in self.dataLoader:
                x = x.to(self.device)
                y = y.to(self.device)
                err, pred = self._trainClassifier(x, y)

        cuda.empty_cache()
        self.model = self.model
        return err, pred

    def _trainClassifier(self, x: Tensor, y: Tensor):
        """
        Trains the classifier
        """
        x = x.to(self.device)
        y = y.to(self.device)
        # Reset gradients
        self.opt.zero_grad()
        pred = self.model(x).to(self.device)
        err = self.loss(pred, y).to(self.device)
        err.backward()
        # Update optimizer
        self.opt.step()
        return err, pred

    def retrieveModel(self) -> nn.Module:
        """
        Function used by aggregators to retrieve the model from the client
        """
        if self.free:
            # Free-rider update
            # The self.model won't update but this is just a logical check
            return self.untrainedModel

        if self.byz:
            # Faulty model update
            self.__manipulateModel()

        if self.useDifferentialPrivacy:
            self.__privacyPreserve()

        return self.model

    def __manipulateModel(self, alpha: int = 20) -> None:
        """
        Function to manipulate the model for byzantine adversaries
        """
        for param in self.model.parameters():
            noise = alpha * torch.randn(param.data.size()).to(self.device)
            param.data.copy_(param.data.to(self.device) + noise)

    def __privacyPreserve(
        self,
        eps1: int = 100,
        eps3: int = 100,
        clipValue: float = 0.1,
        releaseProportion: float = 0.1,
        needClip: bool = False,
        needNormalization: bool = False,
    ):
        """
        Implements differential privacy and applies it to the model
        """
        gamma = clipValue  # gradient clipping value
        s = 2 * gamma  # sensitivity
        Q = releaseProportion  # proportion to release

        # The gradients of the model parameters
        paramArr = nn.utils.parameters_to_vector(self.model.parameters())
        untrainedParamArr = nn.utils.parameters_to_vector(self.untrainedModel.parameters())

        paramNo = len(paramArr)
        shareParamsNo = int(Q * paramNo)

        r = torch.randperm(paramNo).to(self.device)
        paramArr = paramArr[r].to(self.device)
        untrainedParamArr = untrainedParamArr[r].to(self.device)
        paramChanges = (paramArr - untrainedParamArr).detach().to(self.device)

        # Normalising
        if needNormalization:
            paramChanges /= self.n * self.epochs

        # Privacy budgets for
        e1 = eps1  # gradient query
        e3 = eps3  # answer
        e2 = e1 * ((2 * shareParamsNo * s) ** (2 / 3))  # threshold

        paramChanges = paramChanges.cpu()
        tau = percentile(abs(paramChanges), Q * 100)
        paramChanges = paramChanges.to(self.device)
        # tau = 0.0001
        noisyThreshold = laplace.rvs(scale=(s / e2)) + tau

        queryNoise = laplace.rvs(scale=(2 * shareParamsNo * s / e1), size=paramNo)
        queryNoise = torch.tensor(queryNoise).to(self.device)

        releaseIndex = torch.empty(0).to(self.device)
        while torch.sum(releaseIndex) < shareParamsNo:
            if needClip:
                noisyQuery = abs(clip(paramChanges, -gamma, gamma)) + queryNoise
            else:
                noisyQuery = abs(paramChanges) + queryNoise
            noisyQuery = noisyQuery.to(self.device)
            releaseIndex = (noisyQuery >= noisyThreshold).to(self.device)

        filteredChanges = paramChanges[releaseIndex]

        answerNoise = laplace.rvs(
            scale=(shareParamsNo * s / e3), size=torch.sum(releaseIndex).cpu()
        )
        answerNoise = torch.tensor(answerNoise).to(self.device)
        if needClip:
            noisyFilteredChanges = clip(filteredChanges + answerNoise, -gamma, gamma)
        else:
            noisyFilteredChanges = filteredChanges + answerNoise
        noisyFilteredChanges = noisyFilteredChanges.to(self.device)

        # Demoralising the noise
        if needNormalization:
            noisyFilteredChanges *= self.n * self.epochs

        # logPrint("Broadcast: {}\t"
        #          "Trained: {}\t"
        #          "Released: {}\t"
        #          "answerNoise: {}\t"
        #          "ReleasedChange: {}\t"
        #          "".format(untrainedParamArr[releaseIndex][0],
        #                    paramArr[releaseIndex][0],
        #                    untrainedParamArr[releaseIndex][0] + noisyFilteredChanges[0],
        #                    answerNoise[0],
        #                    noisyFilteredChanges[0]))
        # sys.stdout.flush()

        paramArr = untrainedParamArr
        paramArr[releaseIndex][:shareParamsNo] += noisyFilteredChanges[:shareParamsNo]

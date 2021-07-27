from utils.typings import Errors
from experiment.AggregatorConfig import AggregatorConfig
from torch import nn
import  torch.nn.functional as F
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

class FedDFAggregator(Aggregator):
    """
    Federated Ensemble Distillation Aggregator that uses Knowledge Distillation to combine the client models into a global model.
    """
    
    def __init__(
        self,
        clients: List[Client],
        model: nn.Module,
        config: AggregatorConfig,
        useAsyncClients: bool = False,
    ):
        super().__init__(clients, model, config, useAsyncClients)
        
        logPrint("INITIALISING FedDF Aggregator!")
        # Unlabelled data which will be used in Knowledge Distillation
        self.distillationData = None # data is loaded in __runExperiment function
        self.sampleSize = config.sampleSize
        self.true_labels = None
        self.pseudolabelMethod = 'avglogits'
        
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
        
        if self.true_labels is None:
            self.true_labels = self.distillationData.labels
            
        kd = KnowledgeDistiller(self.distillationData, method=self.pseudolabelMethod)
        
        logPrint(f"FedDF: Distilling knowledge (ensemble error: {100*(1-self.ensembleAccuracy(kd._pseudolabelsFromEnsemble(models))):.2f} %)")
        
        avg_model = self._averageModel(models, clients)
        #avg_model = self._medianModel(models)
        avg_model = kd.distillKnowledge(models, avg_model)
        
        return avg_model
    
    def ensembleAccuracy(self, pseudolabels):
        _, predLabels = torch.max(pseudolabels,dim=1)
        mconf = confusion_matrix(self.true_labels.cpu(), predLabels.cpu())
        return 1.0 * mconf.diagonal().sum() / len(self.distillationData)
        
        
    @staticmethod
    def requiresData():
        """
        Returns boolean value depending on whether the aggregation method requires server data or not.
        """
        return True
        
    



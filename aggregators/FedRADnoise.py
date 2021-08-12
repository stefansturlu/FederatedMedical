from utils.typings import Errors
from experiment.AggregatorConfig import AggregatorConfig
from torch import nn
import  torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
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
from copy import deepcopy
from sklearn.metrics import confusion_matrix
from utils.KnowledgeDistiller import KnowledgeDistiller
from pandas import DataFrame
from datasetLoaders.DatasetInterface import DatasetInterface

class FedRADnoiseAggregator(Aggregator):
    """
    Federated Robust Adaptive Distillation aggregator using noise (FedRADnoise), which uses Knowledge Distillation using medians for pseudolabels and median-based weighted average to combine the client models into a global model.
    The server requires no data and only uses noise
    """
    
    def __init__(
        self,
        clients: List[Client],
        model: nn.Module,
        config: AggregatorConfig,
        useAsyncClients: bool = False,
    ):
        super().__init__(clients, model, config, useAsyncClients)
        
        logPrint("INITIALISING FedRADnoise Aggregator!")
        # Unlabelled data which will be used in Knowledge Distillation
        #self.distillationData = None # data is loaded in __runExperiment function
        self.sampleSize = config.sampleSize
        self.true_labels = None
        self.pseudolabelMethod = 'medlogits'
        
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
        
        class DataObject: # Quick and dirty solution to make a data-like object with random noise
            def __init__(self, data, labels):
                self.data = data
                self.labels = labels
            def __len__(self):
                return len(self.data)
            def __getitem__(self, index):
                return self.data[index], self.labels[index]
            
        # Random noise data. Labels are replaced by pseudolabels during knowledge distillation
        distillationData = DataObject(torch.rand(size=[10000,28*28]), torch.zeros(10000)) 
            
        kd = KnowledgeDistiller(distillationData, method=self.pseudolabelMethod, malClients = [i for i,c in enumerate(clients) if c.flip or c.byz])
        
        #logPrint(f"FedBRAD: Distilling knowledge (ensemble error: {100*(1-self.ensembleAccuracy(kd._pseudolabelsFromEnsemble(models))):.2f} %)")
        
        #client_p = torch.tensor([c.p for c in clients])
        weights = kd.medianBasedScores(models, clients)
        # Taking number of datapoints for clients into consideration
        #weights = weights*client_p
        #weights /= weights.sum()
        print("Median based scores:", ", ".join([f"{w*100:.1f}%" for w in weights]))
        avg_model = self._weightedAverageModel(models, weights)
        
        # We skip knowledge distillation on the random noise.
        avg_model = kd.distillKnowledge(models, avg_model)
        
        return avg_model
    
    def ensembleAccuracy(self, pseudolabels):
        _, predLabels = torch.max(pseudolabels,dim=1)
        mconf = confusion_matrix(self.true_labels.cpu(), predLabels.cpu())
        return 1.0 * mconf.diagonal().sum() / len(self.distillationData)
        
        
    #@staticmethod
    #def requiresData():
        #"""
        #Returns boolean value depending on whether the aggregation method requires server data or not.
        #"""
        #return False
        
    



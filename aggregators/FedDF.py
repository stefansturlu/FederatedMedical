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


class FedDFAggregator(Aggregator):
    """
    Federated Ensemble Distillation Aggregator that uses Knowledge Distillation to combine the client models into a global model.
    TODO: Refactor code! FedBE uses distillation as well. The Distillation part can be a module in the utils folder.
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
        #self.method = config.samplingMethod
        #self.alpha = config.samplingDirichletAlpha
        
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
        avg_model = self._avgModel(clients, models)

        T = 1 # Temperature of softmax
        pseudolabels = self._pseudolabelsFromEnsemble(models, T)
        if self.true_labels is None:
            self.true_labels = self.distillationData.labels
        self.distillationData.labels = pseudolabels
        
        #TODO: Implement knowledge distillation
        logPrint(f"FedDF: Distilling knowledge (ensemble error: {100*(1-self.ensembleAccuracy()):.2f} %)")
        avg_model = self._distillKnowledge(avg_model, T)
        
        return avg_model
    
    def _avgModel(self, clients: List[Client], models: List[nn.Module]) -> nn.Module:
        """
        Returns weighted average of clients' models.
        """
        avg_model = deepcopy(self.model)
        self.renormalise_weights(clients)

        comb = 0.0
        for i, client in enumerate(clients):
            self._mergeModels(
                models[i].to(self.device),
                avg_model.to(self.device),
                client.p,
                comb,
            )
            comb = 1.0
        return avg_model
            
        
    def _pseudolabelsFromEnsemble(self, ensemble, temperature):
        with torch.no_grad():
            pseudolabels = torch.zeros_like(ensemble[0](self.distillationData.data))
            for model in ensemble:
                pseudolabels += F.softmax(model(self.distillationData.data)/temperature, dim=1)
                if torch.isnan(pseudolabels).any():
                    print("WARNING! Something went wrong in _pseudolabelsFromEnsemble!")
            return pseudolabels/len(ensemble)
        
    #def loss_fn_kd(outputs, labels, teacher_outputs, temperature):
        #"""
        #Compute the knowledge-distillation (KD) loss given outputs, labels.
        #"Hyperparameters": temperature and alpha
        #NOTE: the KL Divergence for PyTorch comparing the softmaxs of teacher
        #and student expects the input tensor to be log probabilities! See Issue #2
        #"""
        #alpha = 0.5
        #T = temperature
        #KD_loss = nn.KLDivLoss()(F.log_softmax(outputs/T, dim=1),
                             #F.softmax(teacher_outputs/T, dim=1)) * (alpha * T * T) + \
              #F.cross_entropy(outputs, labels) * (1. - alpha)
        #return KD_loss
        
    def _distillKnowledge(self, student_model, temperature):
        tmptmp = deepcopy(student_model)
        epochs = 2
        lr = 0.0001
        momentum = 0.5
        opt = optim.SGD(student_model.parameters(), momentum=momentum, lr=lr, weight_decay=1e-4)
        Loss = nn.KLDivLoss # Config?
        #nn.KLDivLoss()(F.log_softmax(outputs/T, dim=1),
                             #F.softmax(teacher_outputs/T, dim=1)) * (alpha * T * T)
        loss = Loss(reduction='batchmean')
        
        # https://pytorch.org/blog/pytorch-1.6-now-includes-stochastic-weight-averaging/
        swa_model = AveragedModel(student_model)
        #scheduler = CyclicLR(opt, T_max=100)
        scheduler = CosineAnnealingLR(opt, T_max=100)
        swa_scheduler = SWALR(opt, swa_lr=0.005)
        
        # Tune the SWA stuff! It might just be the key to success
        
        dataLoader = DataLoader(self.distillationData, batch_size=16)
        
        for i in range(epochs):
            totalerr = 0
            for j, (x,y) in enumerate(dataLoader):
                opt.zero_grad()
                pred = student_model(x)
                if torch.isnan(pred).any():
                    print(f"WARNING! Something went wrong in prediction during distillation! Round {j}")
                    return None
                err = loss(F.log_softmax(pred/temperature, dim=1), y) * temperature * temperature
                err.backward()
                totalerr += err
                opt.step()
                with torch.no_grad():
                    pred = student_model(x)
                    if torch.isnan(pred).any():
                        print(f"WARNING! Something went wrong in pred after opt during distillation! Round {j}")
                    
            scheduler.step()
            swa_model.update_parameters(student_model)
            swa_scheduler.step()
            
            logPrint("Distillation epoch error:",totalerr)
            torch.optim.swa_utils.update_bn(dataLoader, swa_model)
        
        return swa_model.module # Return the averaged model
    
    
    def ensembleAccuracy(self):
        _, predLabels = torch.max(self.distillationData.labels,dim=1)
        mconf = confusion_matrix(self.true_labels.cpu(), predLabels.cpu())
        return 1.0 * mconf.diagonal().sum() / len(self.distillationData)
        
        
        
    



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

class FedBEAggregator(Aggregator):
    """
    Federated Bayesian Ensemble Aggregator that fits a distribution to the model weights, samples models from the distribution and uses Knowledge Distillation to combine the ensemble into a global model.
    """
    
    def __init__(
        self,
        clients: List[Client],
        model: nn.Module,
        config: AggregatorConfig,
        useAsyncClients: bool = False,
    ):
        super().__init__(clients, model, config, useAsyncClients)
        
        logPrint("INITIALISING FedBE Aggregator!")
        # Unlabelled data which will be used in Knowledge Distillation
        self.distillationData = None # data is loaded in __runExperiment function
        self.sampleSize = config.sampleSize
        self.method = config.samplingMethod
        self.samplingAlpha = config.samplingDirichletAlpha
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

        # STEP 1: Construct distribution of models from which to sample, and sample M models
        logPrint(f"Step 1 of FedBE: Constructing distribution from {len(models)} models and sampling {self.sampleSize} models.")
        ensemble = self._sampleModels(clients, models, self.method)
                
        if self.true_labels is None:
            self.true_labels = self.distillationData.labels
        
        kd = KnowledgeDistiller(self.distillationData, self.pseudolabelMethod)
        
        ensembleError = 100*(1-self.ensembleAccuracy(kd._pseudolabelsFromEnsemble(ensemble)))
        modelsError = 100*(1-self.ensembleAccuracy(kd._pseudolabelsFromEnsemble(models)))
        logPrint(f"Step 2 of FedBE: Distilling knowledge (ensemble error: {ensembleError:.2f} %, models error: {modelsError:.2f})")
        
        avg_model = self._averageModel(models, clients)
        #avg_model = self._medianModel(models)
        #avg_model = self._averageModel(ensemble)
        avg_model = kd.distillKnowledge(ensemble, avg_model)
        
        return avg_model
    
    
    def _sampleModels(self, clients: List[Client], models: List[nn.Module], method='dirichlet') -> List[nn.Module]:
        """
        Sampling models using Gaussian or Dirichlet distributiton. Dirichlet distribution can be used client-wise or elementwise
        
        Parameters:
        clients: List of clients.
        models: List of the clients' models.
        """
        
        self.renormalise_weights(clients)
        M = self.sampleSize
        sampled_models = [deepcopy(self.model) for _ in range(M)]
        
        if method == 'gaussian':
            logPrint("Sampling using Gaussian distribution")
            
            client_model_dicts = [m.state_dict() for m in models]
            client_p = torch.tensor([c.p for c in clients])
            
            for name1, param1 in self.model.named_parameters():
                x = torch.stack([c[name1] for c in client_model_dicts])
                p_shape = torch.tensor(x.shape)
                p_shape[1:] = 1
                client_p = client_p.view(list(p_shape))
                
                # Weighted mean and std dev
                x_mean = (x * client_p).sum(dim=0)
                x_std = ((x-x_mean.unsqueeze(0))**2 * client_p).sum(dim=0).sqrt()
                #x_std = x.std(dim=0)
                
                # Use small std instead of 0 std, since Normal doesn't support 0 std
                x_std[x_std==0] += 1e-10
                
                # Fit a diagonal gaussian distribution to values
                d = Normal(x_mean, x_std)
                # Sample M models
                samp = d.sample([M,])
                
                # Update each model in ensemble in-place
                for i, e in enumerate(sampled_models):
                    params_e = e.state_dict()
                    params_e[name1].data.copy_(samp[i])
            
        if method == 'dirichlet_elementwise':
            logPrint("Sampling using Dirichlet method elementwise")
            
            client_model_dicts = [m.state_dict() for m in models]
            client_p = torch.tensor([c.p for c in clients])
            
            for name1, param1 in self.model.named_parameters():
                x = torch.stack([c[name1] for c in client_model_dicts]) # 30 x 512 x 784 or 30 x 512
                
                alphas = client_p*len(client_p) * self.samplingAlpha
                # Fit a diagonal gaussian distribution to values
                d = Dirichlet(alphas)
                
                # Sample M weights for each parameter
                sample_shape = [M]+list(x[0].shape) # M x 512 x 784
                weights = d.sample(sample_shape) # M x 512 x 784 x 30 
                perm = [0] + [(i-1)%(len(weights.shape)-1)+1 for i in range(len(weights.shape)-1)]
                weights = weights.permute(*perm).to(self.device) # M x 30 x 512 x 784
                
                # Compute M linear combination of client models
                samp = (weights * x.unsqueeze(0)).sum(dim=1)
                
                # Update each model in ensemble in-place
                for i, e in enumerate(sampled_models):
                    params_e = e.state_dict()
                    params_e[name1].data.copy_(samp[i])
            
        
        if method == 'dirichlet':
            logPrint("Sampling using Dirichlet method client-wise")
            # Sample weights for weighted average of client models
            # using a symmetrical Dirichlet distribution
            alphas = self.samplingAlpha*torch.ones(len(models)).repeat(M).reshape(M,-1)
            d = Dirichlet(alphas)
            sample = d.sample() # Shape: M x len(models)
            
            # Take client dataset sizes into account (eq. 12 from FedBE paper) 
            # Note: Why isn't this done to the alphas before sampling? This is not equivalent.
            for i, c in enumerate(clients):
                sample[:,i] *= c.p
            sample = sample / sample.sum(dim=1).unsqueeze(1)

            # Compute weighted averages based on Dirichlet sample
            for i, s_model in enumerate(sampled_models):
                comb = 0.0
                for j, c_model in enumerate(models):
                    # _mergeModels updates s_model in-place
                    self._mergeModels(
                        c_model.to(self.device),
                        s_model.to(self.device),
                        sample[i,j],
                        comb,                
                    )
                    comb = 1.0
            
        return sampled_models
        
        
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
        



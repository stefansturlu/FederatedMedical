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

class FedABEDAggregator(Aggregator):
    """
    A novel aggregator called Federated Adaptive Bayesian Ensemble with Distillation (FedABED), which fits a Dirichlet distribution to the models based on an alpha score, samples weighted combinations of models from the distribution and uses Knowledge Distillation to combine the ensemble into a global model.
    Target: An aggregator that is more robust to attacks.
    """
    
    def __init__(
        self,
        clients: List[Client],
        model: nn.Module,
        config: AggregatorConfig,
        useAsyncClients: bool = False,
    ):
        super().__init__(clients, model, config, useAsyncClients)
        
        logPrint("INITIALISING FedABED Aggregator!")
        # Unlabelled data which will be used in Knowledge Distillation
        self.distillationData = None # data is loaded in __runExperiment function
        self.true_labels = None
        self.sampleSize = config.sampleSize
        self.method = config.samplingMethod
        self.samplingAlpha = config.samplingDirichletAlpha
        
        self.xi: float = config.xi
        self.deltaXi: float = config.deltaXi
        self.pseudolabelMethod = 'medlogits'
        self.threshold = 0.0 # For median-counting blocker
        
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
        
        kd = KnowledgeDistiller(self.distillationData, method=self.pseudolabelMethod,
                                malClients = [i for i,c in enumerate(clients) if c.flip or c.byz])
        
        logPrint("FedABED Step 1: Calculating scores")
        # Get weights for Dirichlet sampling. Make boolean mask for zero-values
        weights = kd.medianBasedScores(models, clients)
        
        # Filter out clients below threshold
        mask = ~(weights<=self.threshold)
        weights = weights[mask] / weights[mask].sum() # Filtered and normalised
        good_models = [models[i] for i, b in enumerate(mask) if b]
        
        logPrint(f"FedABED Step 2: Sampling ensemble of size {self.sampleSize} from {len(good_models)} models.")
        # STEP 1: Construct distribution of models from which to sample, and sample M models
        ensemble = self._sampleModels(good_models, weights, self.method) 
        
        err_e = 100*(1-self.ensembleAccuracy(kd._pseudolabelsFromEnsemble(ensemble)))
        err_m = 100*(1-self.ensembleAccuracy(kd._pseudolabelsFromEnsemble(models)))
        
        logPrint(f"FedABED Step 3: Distilling knowledge (Errors - models: {err_m:.1f}%, ensemble: {err_e:.1f}%)")
        avg_model = self._weightedAverageModel(good_models, weights)
        #avg_model = self._medianModel(models)
        #avg_model = self._averageModel(ensemble)
        avg_model = kd.distillKnowledge(ensemble, avg_model)
        
        return avg_model
    
    
    
    def _sampleModels(self, models: List[nn.Module], weights: torch.Tensor, method='dirichlet') -> List[nn.Module]:
        """
        Sampling models using Gaussian or Dirichlet distributiton. Dirichlet distribution can be used client-wise or elementwise
        
        Parameters:
        clients: List of clients.
        models: List of the clients' models.
        """
        
        M = self.sampleSize
        sampled_models = [deepcopy(self.model) for _ in range(M)]
        client_p = weights
        
        if method == 'dirichlet_elementwise':
            logPrint("Sampling using Dirichlet method elementwise")
            
            client_model_dicts = [m.state_dict() for m in models]
            
            for name1, param1 in self.model.named_parameters():
                x = torch.stack([c[name1] for c in client_model_dicts]) # 30 x 512 x 784 or 30 x 512
                
                # Fit a diagonal gaussian distribution to clients
                alphas = client_p*len(client_p) * self.samplingAlpha
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
            alphas = client_p*len(client_p) * self.samplingAlpha
            print("Dirichlet alpha:",alphas)
            d = Dirichlet(alphas)
            sample = d.sample([M]) # Shape: M x len(models)
            
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
        

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
        self.alpha = config.samplingDirichletAlpha
        self.true_labels = None
        
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

        # STEP 1-2: Construct distribution of models from which to sample, and sample M models
        logPrint(f"Step 1 of FedBE: Constructing distribution from {len(models)} models and sampling {self.sampleSize} models.")
        ensemble = self._sampleModels(clients, models, self.method)
        
        #params = self.model.named_parameters()
        #for name, param in params:
            #print("name:",name, "\nparam shape",param.shape)
            #for j, sample in enumerate(ensemble):
                #print(f"SAMPLE NUMBER {j}")
                #paramSamp = sample.named_parameters()
                #for name1, param1 in paramSamp:
                    #print(name1, param1)
                
        #TODO: Construct pseudolabelled dataset
        #logPrint(f"Step 2 of FedBE: Constructing pseudolabelled set")
        T = 1 # Temperature of softmax
        pseudolabels = self._pseudolabelsFromEnsemble(ensemble, T)
        if self.true_labels is None:
            self.true_labels = self.distillationData.labels
        self.distillationData.labels = pseudolabels
        
        #TODO: Implement knowledge distillation
        logPrint(f"Step 2 of FedBE: Distilling knowledge (ensemble error: {100*(1-self.ensembleAccuracy()):.2f} %)")
        avg_model = self._distillKnowledge(avg_model, T)
        
        return avg_model
    
    
    def _sampleModels(self, clients: List[Client], models: List[nn.Module], method='dirichlet') -> List[nn.Module]:
        """
        Sampling models using Dirichlet distributiton (client-wise)
        TODO: Add both Gaussian sampling and Dirichlet element-wise.
        
        Parameters:
        clients: List of clients.
        models: List of the clients' models.
        """
        # Are the Dirichlet parameters sampled element-wise or client-wise?
        # Try both, maybe? The Gaussian distribution samples are element-wise.
        
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
                
                alphas = client_p*len(client_p) * self.alpha
                # Fit a diagonal gaussian distribution to values
                d = Dirichlet(alphas)
                
                # Sample M weights for each parameter
                sample_shape = [M]+list(x[0].shape) # M x 512 x 784
                weights = d.sample(sample_shape) # M x 512 x 784 x 30 
                perm = [0] + [(i-1)%(len(weights.shape)-1)+1 for i in range(len(weights.shape)-1)]
                weights = weights.permute(*perm) # M x 30 x 512 x 784
                
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
            alphas = self.alpha*torch.ones(len(models)).repeat(M).reshape(M,-1)
            d = Dirichlet(alphas)
            sample = d.sample() # Shape: M x len(models)
            
            # Take client dataset sizes into account (eq. 12 from FedBE paper)
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
        mconf = confusion_matrix(self.true_labels, predLabels) 
        return 1.0 * mconf.diagonal().sum() / len(self.distillationData)
        
        
        
    



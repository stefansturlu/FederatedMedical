import torch
from torch import nn
import  torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim.swa_utils import AveragedModel, SWALR
from torch.optim.lr_scheduler import CosineAnnealingLR
from logger import logPrint
from typing import List
import gc

class KnowledgeDistiller:
    """
    A class for Knowledge Distillation using ensembles.
    """
    def __init__(self, dataset, epochs=2, batch_size = 16, temperature=1, method='avglogits', malClients=[], device='cpu'):
        self.dataset = dataset
        self.batch_size = batch_size
        self.T = temperature
        self.epochs = epochs
        self.lr = 0.0001
        self.momentum = 0.5
        self.swa_lr = 0.005
        self.method = method
        self.device=device
        #self.Optim = optim.SGD
        #self.Loss = nn.KLDivLoss
        self.malClients = malClients
    
    def distillKnowledge(self, teacher_ensemble, student_model):
        """
        Takes in a teacher ensemble (list of models) and a student model.
        Trains the student model using unlabelled dataset, then returns it.
        Args:
            teacher_ensemble is list of models used to construct pseudolabels using self.method
            student_model is models that will be trained
        """
            
        gc.collect()
        torch.cuda.empty_cache()
        # Set labels as soft ensemble prediction
        self.dataset.labels = self._pseudolabelsFromEnsemble(teacher_ensemble, self.method)
        
        opt = optim.SGD(student_model.parameters(),
                        momentum=self.momentum,
                        lr=self.lr,
                        weight_decay = 1e-4
                       )
        Loss = nn.KLDivLoss
        loss = Loss(reduction='batchmean')
        
        # https://pytorch.org/blog/pytorch-1.6-now-includes-stochastic-weight-averaging/
        swa_model = AveragedModel(student_model)
        scheduler = CosineAnnealingLR(opt, T_max=100)
        swa_scheduler = SWALR(opt, swa_lr=self.swa_lr)
        
        dataLoader = DataLoader(self.dataset, batch_size=self.batch_size)
        for i in range(self.epochs):
            total_err = 0
            for j, (x,y) in enumerate(dataLoader):
                
                opt.zero_grad()
                pred = student_model(x)
                err = loss(F.log_softmax(pred/self.T, dim=1), y)*self.T*self.T
                err.backward()
                total_err += err
                opt.step()
            logPrint(f"KD epoch {i}: {total_err}")
            scheduler.step()
            swa_model.update_parameters(student_model)
            swa_scheduler.step()
            
            torch.optim.swa_utils.update_bn(dataLoader, swa_model)
        
        gc.collect()
        torch.cuda.empty_cache()
        return swa_model.module
    
    
    
    def _pseudolabelsFromEnsemble(self, ensemble, method=None):
        """
            Combines the probabilities to make ensemble predictions.
            3 possibile methods: 
                avglogits: Takes softmax of the average outputs of the models 
                medlogits: Takes softmax of the median outputs of the models 
                avgprob: Averages the softmax of the outputs of the models
            
            Idea: Use median instead of averages for the prediction probabilities! 
                This might make the knowledge distillation more robust to confidently bad predictors.
        """
        if method is None:
            method = self.method
            
        with torch.no_grad():
            dataLoader = DataLoader(self.dataset, batch_size=self.batch_size)
            preds = []
            for i, (x,y) in enumerate(dataLoader):
                predsBatch = torch.stack([m(x)/self.T for m in ensemble])
                preds.append(predsBatch)
            preds = torch.cat(preds, dim=1)
            print(f"Final preds shape: {preds.shape}")

            #preds = torch.stack([m(self.dataset.data)/self.T for m in ensemble])
            
            if method == 'avglogits':
                pseudolabels = preds.mean(dim=0)
                return F.softmax(pseudolabels, dim=1)
            
            elif method == 'medlogits':
                pseudolabels, idx = preds.median(dim=0)
                return F.softmax(pseudolabels, dim=1)
            
            elif method == 'avgprob':
                preds = F.softmax(preds, dim=2)
                pseudolabels = preds.mean(dim=0)
                return pseudolabels
            
            else:
                raise ValueError("pseudolabel method should be one of: avglogits, medlogits, avgprob")
            
        
    
    def medianBasedScores(self, ensemble: List[nn.Module], clients: List[nn.Module] = []) -> torch.Tensor:
        """
            Gives scores reative to how often the models had the median logit.
        """
        logPrint(f"Calculating model scores based on frequency of median logits")
            
        gc.collect()
        torch.cuda.empty_cache()
        with torch.no_grad():
            dataLoader = DataLoader(self.dataset, batch_size=self.batch_size)
            preds = []
            for i, (x,y) in enumerate(dataLoader):
                predsBatch = torch.stack([m(x)/self.T for m in ensemble])
                preds.append(predsBatch)
            preds = torch.cat(preds, dim=1)
            pseudolabels, idx = preds.median(dim=0)
            
            counts = torch.bincount(idx.view(-1), minlength=preds.size(0)).to(self.device) #
            counts_p = counts/counts.sum()
            
            #logPrint(", ".join([f"{c*100:.1f}%" for c in counts_p]))
            logPrint("Counts:",counts)
            mask = torch.ones(counts.shape, dtype=bool)
            mask[self.malClients] = False
            mask = mask.to(self.device)
            print(f"Mean of attackers: {counts_p[~mask].mean()*100:.2f}%, healthy: {counts_p[mask].mean()*100:.2f}%")
            
            # If list of clients is provided, scale according to client.p
            if len(clients) > 0:
                client_p = torch.tensor([c.p for c in clients]).to(self.device)
                counts_p *= client_p
                counts_p /= counts_p.sum()
            
            gc.collect()
            torch.cuda.empty_cache()
            return counts_p

            
    
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
        

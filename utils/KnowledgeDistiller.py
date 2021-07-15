import torch
from torch import nn
import  torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim.swa_utils import AveragedModel, SWALR
from torch.optim.lr_scheduler import CosineAnnealingLR

class KnowledgeDistiller:
    """
    A class for Knowledge Distillation using ensembles.
    """
    def __init__(self, dataset, epochs=2, batch_size = 16, temperature=1):
        self.dataset = dataset
        self.batch_size = batch_size
        self.T = temperature
        self.epochs = epochs
        self.lr = 0.0001
        self.momentum = 0.5
        self.swa_lr = 0.005
        #self.Optim = optim.SGD
        #self.Loss = nn.KLDivLoss
    
    def distillKnowledge(self, teacher_ensemble, student_model):
        """
        Takes in a teacher ensemble (list of models) and a student model (optional?).
        Trains the student model using unlabelled dataset, then returns it.
        """
        # Set labels as soft ensemble prediction
        self.dataset.labels = self._pseudolabelsFromEnsemble(teacher_ensemble)
        
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
            print(f"KD epoch {i}: {total_err}")
            scheduler.step()
            swa_model.update_parameters(student_model)
            swa_scheduler.step()
            
            torch.optim.swa_utils.update_bn(dataLoader, swa_model)
        
        return swa_model.module
    
    
    
    def _pseudolabelsFromEnsemble(self, ensemble, method='logits'):
        """
            Combines the probabilities to make ensemble predictions.
            3 possibile methods: 
                logits: Takes softmax of the average outputs of the models 
                prob: Averages the softmax of the outputs of the models
                logprob: Averages the logsoftmax of the outputs of the models 
            
            Idea: Use median instead of averages for the prediction probabilities! 
                This might make the knowledge distillation more robust to confidently bad predictors.
                You might have to normalise it in the dimension.
        """
        with torch.no_grad():
            pseudolabels = torch.zeros_like(ensemble[0](self.dataset.data))
            preds = torch.stack([m(self.dataset.data)/self.T for m in ensemble])
            if method == 'logits':
                #pseudolabels = preds.mean(dim=0)  # Final error: 8.89 with no attacks, alpha=0.1
                pseudolabels, _ = preds.median(dim=0) # Final error: 8.66 with no attacks, alpha=0.1
                return F.softmax(pseudolabels, dim=1)
            
            elif method == 'prob': # Only mean. The median probabilities lead to predictions not summing up to 1.
                preds = F.softmax(preds, dim=2)
                pseudolabels = preds.mean(dim=0) # Final error: 12.56 with no attacks, alpha=0.1
                # Figure out why median works for logits but not prob
                # Probably: The median value of a softmax function is small. Also, probabilities don't add up to 1.
                #pseudolabels2 = preds.quantile(q=0.5,dim=0) 
                #pseudolabels2, _ = preds.median(dim=0) 
                return pseudolabels
            
            elif method == 'logprob': # Doesn't work, Probably doesn't make sense theoretically
                preds = F.log_softmax(preds, dim=2)
                pseudolabels = preds.mean(dim=0)
                pseudolabels = torch.exp(pseudolabels)
                #pseudolabels, _ = preds.median(dim=0)
                return pseudolabels
            
            return pseudolabels
        
    
    
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
        

from client import Client
from typing import List
from torch import nn
from experiment.AggregatorConfig import AggregatorConfig
from aggregators.FedBE import FedBEAggregator
from logger import logPrint

class FedBEmedAggregator(FedBEAggregator):
    """
    A more robust version of the FedDF aggregator, using median instead of mean logits for pseudolabels in knowledge distillation.
    """
    def __init__(
        self,
        clients: List[Client],
        model: nn.Module,
        config: AggregatorConfig,
        useAsyncClients: bool = False,
    ):
        super().__init__(clients, model, config, useAsyncClients)
        logPrint("correction: INITIALISING FedBEmed Aggregator!")
        self.pseudolabelMethod = 'medlogits'
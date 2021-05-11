import torch
from torch.utils.data import Dataset

class DatasetInterface(Dataset):
    def __init__(self, labels):
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        raise Exception("Method should be implemented in subclass.")

    def getInputSize(self):
        raise Exception(
            "Method should be implemented by subclasses where "
            "models requires input size update (based on dataset)."
        )

    def zeroLabels(self):
        self.labels = torch.zeros(len(self.labels), dtype=torch.long)

from torch import nn
from torchvision import models


class Classifier(nn.Module):
    def __init__(self, classes=3, model="resnet18"):
        super(Classifier, self).__init__()
        if model == "resnet18":
            self.cnn = models.resnet18(pretrained=True)
            self.cnn.fc = nn.Linear(512, classes)

        elif model == "resnext50":
            self.cnn = models.resnext50_32x4d(pretrained=True)
            self.cnn.classifier = nn.Linear(1280, classes)

    def forward(self, x):
        return self.cnn(x)

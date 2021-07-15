from torch import nn, Tensor


class Classifier(nn.Module):
    """
    Simple model with linear layers for mnist
    """

    def __init__(self):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(784, 512)
        self.relu1 = nn.LeakyReLU(negative_slope=0.1)
        self.drop1 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(512, 256)
        self.relu2 = nn.LeakyReLU(negative_slope=0.1)
        self.drop2 = nn.Dropout(p=0.5)
        self.out = nn.Linear(256, 10)
        #self.out_act = nn.Softmax(dim=1)

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.drop2(x)
        x = self.out(x)
        #x = self.out_act(x)
        return x

import torch
import torch.nn as nn


class ConvolutionalNeuralNetworkModel(nn.Module):

    def __init__(self):
        super(ConvolutionalNeuralNetworkModel, self).__init__()

        self.layer1 = nn.Sequential(nn.Conv2d(1, 32, kernel_size=5, padding=2), nn.MaxPool2d(kernel_size=2))
        self.layer2 = nn.Sequential(nn.Conv2d(32, 64, kernel_size=5, padding=2), nn.MaxPool2d(kernel_size=2))
        self.layer3 = nn.Linear(64 * 7 * 7, 10)

    def logits(self, x):
        x = self.conv(x)
        x = self.pool(x)
        return self.dense(x.reshape(-1, 32 * 14 * 14))

    # Predictor
    def f(self, x):
        return torch.softmax(self.logits(x), dim=1)

    # Cross Entropy loss
    def loss(self, x, y):
        return nn.functional.cross_entropy(self.logits(x), y.argmax(1))

    # Accuracy
    def accuracy(self, x, y):
        return torch.mean(torch.eq(self.f(x).argmax(1), y.argmax(1)).float())

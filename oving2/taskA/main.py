import torch
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

class NotOperatorModel:

    def __init__(self):
        self.W = torch.tensor([[0.0]], requires_grad=True)
        self.b = torch.tensor([[0.0]], requires_grad=True)
    
    def logits(self, x):
        return x @ self.W + self.b
    
    # Predictor
    def f(self, x):
        return torch.sigmoid(self.logits(x))
    
    # Cross Entropy loss
    def loss(self, x, y):
        return torch.nn.functional.binary_cross_entropy_with_logits(self.logits(x), y)

    def accuracy(self, x, y):
        return torch.mean(torch.eq(self.f(x).argmax(1),y.argmax(1)).float())


def train_and_visualize_model(epochs, learning_rate):
    model = NotOperatorModel()

    x_train = torch.tensor([[0], [1]], dtype=torch.float32)
    y_train = torch.tensor([[1], [0]], dtype=torch.float32)
    
    optimizer = torch.optim.SGD([model.W, model.b], learning_rate)
    
    for epoch in range(epochs):
        model.loss(x_train, y_train).backward()
        optimizer.step()
        optimizer.zero_grad()

    # model variables and loss
    print("W = %s, b = %s, loss = %s" % (model.W, model.b, model.loss(x_train, y_train)))
    
    x_range = torch.arange(0, 1.0, 0.01).reshape(-1, 1)
    plt.figure(figsize=(10, 6))
    plt.scatter(x_train, y_train, label='Observations', s=150)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.plot(x_range, model.f(x_range).detach(), label='Fitted Sigmoid Curve', color='red', linewidth=2)
    plt.title('NOT operator')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    train_and_visualize_model(epochs=10, learning_rate=40)
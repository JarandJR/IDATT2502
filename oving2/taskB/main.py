import torch
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

class NotAndOperatorModel:
    def __init__(self):
        self.W = torch.rand((2, 1), requires_grad=True)
        self.b = torch.rand((1, 1), requires_grad=True)
    
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
    model = NotAndOperatorModel()

    x_train = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
    y_train = torch.tensor([[1], [1], [1], [0]], dtype=torch.float32)

    optimizer = torch.optim.SGD([model.W, model.b], learning_rate)
    for epoch in range(epochs):
        model.loss(x_train, y_train).backward()
        optimizer.step()
        optimizer.zero_grad()

    # model variables and loss
    print("W = %s, b = %s, loss = %s" % (model.W, model.b, model.loss(x_train, y_train)))
    
    # Creating a grid of points to visualize the decision boundary
    x1 = np.linspace(0, 1, 100)
    x2 = np.linspace(0, 1, 100)
    x1, x2 = np.meshgrid(x1, x2)
    x_grid = np.column_stack((x1.ravel(), x2.ravel()))
    
    # Calculate model predictions on the grid
    y_grid = model.f(torch.tensor(x_grid, dtype=torch.float32)).detach().numpy()
    y_grid = y_grid.reshape(x1.shape)

    # Create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the surface representing the decision boundary
    ax.plot_surface(x1, x2, y_grid, cmap='gist_heat', alpha=0.8)
    ax.scatter(x_train[:, 0], x_train[:, 1], y_train, c=y_train.ravel())
    
    ax.set_xlabel('Input 1')
    ax.set_ylabel('Input 2')
    ax.set_zlabel('Predicted Output')
    plt.title('Decision Boundary')
    
    plt.show()

if __name__ == "__main__":
    train_and_visualize_model(50, 40)
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

class XorOperatorModel:
    def __init__(self):
        self.W1 = torch.rand((2, 2), requires_grad=True)
        self.b1 = torch.rand((1, 2), requires_grad=True)
        self.W2 = torch.rand((2, 1), requires_grad=True)
        self.b2 = torch.rand((1, 1), requires_grad=True)

    def f1(self, x):
        return torch.sigmoid(x @ self.W1 + self.b1)

    def f2(self, h):
        return torch.sigmoid(h @ self.W2 + self.b2)

    def f(self, x):
        return self.f2(self.f1(x))

    def loss(self, x, y):
        return torch.nn.functional.binary_cross_entropy(self.f(x), y)
    
    def accuracy(self, x, y):
        return torch.mean(torch.eq(self.f(x).round(), y).float())

def train_and_visualize_model(epochs, learning_rate):
    model = XorOperatorModel()
    x_train = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
    y_train = torch.tensor([[0], [1], [1], [0]], dtype=torch.float32)
    
    optimizer = torch.optim.SGD([model.W1, model.b1, model.W2, model.b2], learning_rate)

    for epoch in range(epochs):
        model.loss(x_train, y_train).backward()
        optimizer.step()
        optimizer.zero_grad()

    print("W1 = %s, b1 = %s, W2 = %s, b2 = %s, loss = %s" % (model.W1, model.b1, model.W2, model.b2, model.loss(x_train, y_train)))
    print(f"Accuracy on training data: {model.accuracy(x_train, y_train) * 100}%")

    plot_model(model, x_train, y_train)

def plot_model(model, x_train, y_train):
    x1 = np.linspace(0, 1, 100)
    x2 = np.linspace(0, 1, 100)
    x1, x2 = np.meshgrid(x1, x2)
    x_grid = np.column_stack((x1.ravel(), x2.ravel()))
    
    y_grid = model.f(torch.tensor(x_grid, dtype=torch.float32)).detach().numpy()
    y_grid = y_grid.reshape(x1.shape)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    ax.plot_surface(x1, x2, y_grid, cmap='flag', alpha=0.8)
    ax.scatter(x_train[:, 0], x_train[:, 1], y_train, c='blue')
    
    ax.set_xlabel('Input 1')
    ax.set_ylabel('Input 2')
    ax.set_zlabel('Predicted Output')
    plt.title('XOR Decision Boundary')
    plt.show()

if __name__ == "__main__":
    train_and_visualize_model(10_000, 1)

import torch
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Import for 3D plotting

# Load data from CSV file
data = pd.read_csv('oving1/taskB/day_length_weight.csv')

# Access data using column names
x_train = torch.tensor(data.iloc[:, 1:3].values, dtype=torch.float32)
y_train = torch.tensor(data.iloc[:, 0].values, dtype=torch.float32).view(-1, 1)

class LinearRegressionModel:
    def __init__(self):
        self.W = torch.tensor([[0.0], [0.0]], requires_grad=True)
        self.b = torch.tensor([[0.0]], requires_grad=True)

    def f(self, x):
        return x @ self.W + self.b

    def loss(self, x, y):
        return torch.mean(torch.square(self.f(x) - y))

model = LinearRegressionModel()

optimizer = torch.optim.SGD([model.W, model.b], 0.0001) 
for epoch in range(100_000): 
    model.loss(x_train, y_train).backward()
    optimizer.step()
    optimizer.zero_grad()

# Model variables and loss
print("W = %s, b = %s, loss = %s" % (model.W, model.b, model.loss(x_train, y_train)))

# 3D Visualization
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(x_train[:, 0], x_train[:, 1], y_train, c=y_train, cmap='viridis', marker='o', label='Observations')
ax.set_xlabel('Length')
ax.set_ylabel('Weight')
ax.set_zlabel('Day')
ax.plot_trisurf(x_train[:, 0], x_train[:, 1], model.f(x_train).detach().numpy().flatten(), color='red', alpha=0.5, label='Linear Regression')

handles, labels = scatter.legend_elements()
ax.legend(handles, labels, loc='best', title='3D visulization')

plt.show()

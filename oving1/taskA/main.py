import torch
import pandas as pd
import matplotlib.pyplot as plt

# Load data from CSV file
data = pd.read_csv(r'oving1\taskA\length_weight.csv')

# Access data using column names
x_train = torch.tensor(data.iloc[:, 0].values, dtype=torch.float32).view(-1, 1)
y_train = torch.tensor(data.iloc[:, 1].values, dtype=torch.float32).view(-1, 1)

class LinearRegressionModel:
    def __init__(self):
        self.W = torch.tensor([[0.0]], requires_grad=True)
        self.b = torch.tensor([[0.0]], requires_grad=True)

    def f(self, x):
        return x @ self.W + self.b

    def loss(self, x, y):
        return torch.mean(torch.square(self.f(x) - y))


model = LinearRegressionModel()

optimizer = torch.optim.SGD([model.W, model.b], 0.0001)  # Learning rate as needed
for epoch in range(200_000):  # Number of epochs as needed
    model.loss(x_train, y_train).backward()
    optimizer.step()
    optimizer.zero_grad()

# model variables and loss
print("W = %s, b = %s, loss = %s" % (model.W, model.b, model.loss(x_train, y_train)))

# Visualize the result
plt.scatter(x_train, y_train, label='Observations')
plt.xlabel('Length')
plt.ylabel('Weight')
plt.plot(x_train, model.f(x_train).detach(), label='Linear Regression', color='red')
plt.legend()
plt.show()

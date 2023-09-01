import torch
import pandas as pd
import matplotlib.pyplot as plt


class NonLinearRegressionModel:
    def __init__(self):
        self.W = torch.tensor([[0.0]], requires_grad=True)
        self.b = torch.tensor([[0.0]], requires_grad=True)

    def f(self, x):
        return 20 * torch.sigmoid(torch.matmul(x, self.W) + self.b) + 31

    def loss(self, x, y):
        return torch.nn.functional.mse_loss(self.f(x), y)


class Visulization:
    def __init__(self, file):
        data = pd.read_csv(file)

        # Access data using column names
        self.x_train = torch.tensor(data["# day"].values.reshape(-1, 1), dtype=torch.float32)
        self.y_train = torch.tensor(data["head circumference"].values.reshape(-1, 1), dtype=torch.float32)

    def train_and_visualize(self, epochs, learning_rate):
        model = NonLinearRegressionModel()
        self.train(epochs=epochs, learning_rate=learning_rate, model=model)
        self.visualize(model=model)

    def train(self, epochs, learning_rate, model:NonLinearRegressionModel):
        optimizer = torch.optim.SGD([model.W, model.b], learning_rate)
        for epoch in range(epochs):
            loss = model.loss(self.x_train, self.y_train)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Model variables and loss
        print("W = %s, b = %s, loss = %s" % (model.W, model.b, model.loss(self.x_train, self.y_train)))

    def visualize(self, model:NonLinearRegressionModel):
        plt.figure(figsize=(8, 6))
        plt.title("Predict head circumference based on age (in days)")
        plt.xlabel("Age (days)")
        plt.ylabel("Head Circumference")
        plt.scatter(self.x_train, self.y_train, marker="o")

        # Create a dense range of x-values
        x = torch.arange(torch.min(self.x_train), torch.max(self.x_train), 1.0).reshape(-1, 1)

        # Compute y-values using the model for the dense range of x-values
        y = model.f(x).detach()

        plt.plot(x, y, color="orange", linewidth=2, label=r"$f(x) = 20\sigma(xW + b) + 31$")

        plt.legend()
        plt.show()


if __name__ == "__main__":
    visulizer = Visulization(file='oving1/taskC/day_head_circumference.csv')
    visulizer.train_and_visualize(epochs=200_000, learning_rate=0.000001)
    
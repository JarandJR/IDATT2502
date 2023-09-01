import torch
import torchvision
import matplotlib.pyplot as plt

class HandWrittenPredicatorModel:
    def __init__(self):
        self.W = torch.rand((784, 10), requires_grad=True)
        self.b = torch.rand((1, 10), requires_grad=True)
    
    def logits(self, x):
        return x @ self.W + self.b
    
    def f(self, x):
        return torch.softmax(self.logits(x), dim=1)
 
    def loss(self, x, y):
        return torch.nn.functional.cross_entropy(self.logits(x), y)

    def accuracy(self, x, y):
        return torch.mean(torch.eq(self.f(x).argmax(1),y.argmax(1)).float())

def train_and_test_model(epochs, learning_rate):
    # Load observations from the mnist dataset. The observations are divided into a training set and a test set
    mnist_train = torchvision.datasets.MNIST('./oving2/taskD/data', train=True, download=True)
    x_train = mnist_train.data.reshape(-1, 784).float()  # Reshape input
    y_train = torch.zeros((mnist_train.targets.shape[0], 10))  # Create output tensor
    y_train[torch.arange(mnist_train.targets.shape[0]), mnist_train.targets] = 1  # Populate output

    mnist_test = torchvision.datasets.MNIST('./oving2/taskD/data', train=False, download=True)
    x_test = mnist_test.data.reshape(-1, 784).float()  # Reshape input
    y_test = torch.zeros((mnist_test.targets.shape[0], 10))  # Create output tensor
    y_test[torch.arange(mnist_test.targets.shape[0]), mnist_test.targets] = 1  # Populate output

    model = HandWrittenPredicatorModel()
    optimizer = torch.optim.SGD([model.W, model.b], learning_rate)
    for epoch in range(epochs):
        model.loss(x_train, y_train).backward()
        optimizer.step()
        optimizer.zero_grad()

    print(f"Accuracy on training data: {model.accuracy(x_train, y_train)*100}%")

    # test and show 10 random images from the test set with the model's prediction
    plt.figure(figsize=(10, 5))
    for i in range(10):
        index = torch.randint(0, x_test.shape[0], ())
        x = x_test[index]
        plt.subplot(2, 5, i + 1)
        plt.imshow(x.reshape(28, 28), cmap="gray")
        plt.title(f"Model: {model.f(x).argmax().item()}\nActual: {y_test[index].argmax().item()}")
        plt.xticks([])
        plt.yticks([])

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    train_and_test_model(epochs=100, learning_rate=1)
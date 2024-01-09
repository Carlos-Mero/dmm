import torch
from torch import nn
from torch import optim
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader
from accelerate import Accelerator
import matplotlib.pyplot as plt

class MNIST_tester(object):
    def __init__(self):
        self.accelerator = Accelerator()
        self.device = self.accelerator.device
        self.MNIST_train = datasets.MNIST(
            root="../datasets/",
            train=True,
            download=True,
            transform=transforms.ToTensor()
        )
        self.MNIST_test = datasets.MNIST(
            root="../datasets/",
            train=False,
            download=True,
            transform=transforms.ToTensor()
        )
        self.train_loader = DataLoader(self.MNIST_train, 128)
        self.test_loader = DataLoader(self.MNIST_test, 128)
        self.train_loader, self.test_loader = self.accelerator.prepare(
                self.train_loader, self.test_loader)
        self.loss_fn = nn.CrossEntropyLoss()
        self.losses = []
        self.accs = []

    def guide(self, model):
        optimizer = optim.SGD(model.parameters(), lr=0.2)       
        for (X, y) in self.train_loader:
            X = X.to(self.device)
            y = y.to(self.device)
            model = model.to(self.device)
            pred = model(X)
            loss = self.loss_fn(pred, y)
            self.accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()
            break

    def test(self, model):
        size = len(self.MNIST_test)
        num_batches = len(self.test_loader)
        model = model.to(self.device)
        model.eval()
        tloss, tcorrect = 0.0, 0.0
        with torch.no_grad():
            for X, y in self.test_loader:
                X = X.to(self.device)
                y = y.to(self.device)
                pred = model(X)
                tloss += self.loss_fn(pred, y)
                tcorrect += (pred.argmax(1) == y).type(torch.float).sum().item()
        tloss /= num_batches
        tloss = tloss.item()
        tcorrect /= size
        tcorrect *= 100
        print(f"Current Test Error: {tloss:>8f}")
        print(f"Current Test Accuracy: {tcorrect:>0.01f}%")
        self.losses.append(tloss)
        self.accs.append(tcorrect)

    def draw_curves(self):
        plt.plot(range(len(self.losses)), self.losses, color='red')
        plt.xlabel('time steps')
        plt.title("losses through backward progress")
        plt.savefig("./losses.png")
        plt.clf()
        plt.plot(range(len(self.accs)), self.accs, color='green')
        plt.xlabel('time steps')
        plt.title("accuracies through backward progress")
        plt.savefig("./accs.png")
        plt.clf()

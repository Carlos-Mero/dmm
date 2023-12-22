import torch
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader
from accelerate import Accelerator
import matplotlib.pyplot as plt

class MNIST_tester(object):
    def __init__(self):
        self.accelerator = Accelerator()
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
        self.losses = []
        self.accs = []

    def test(self, model):
        size = len(self.MNIST_test)
        num_batches = len(self.test_loader)
        model.eval()
        tloss, tcorrect = 0.0, 0.0
        with torch.no_grad():
            for X, y in self.test_loader:
                X = X.to(self.device)
                y = y.to(self.device)
                pred = self.model(X)
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
        plt.plot(len(self.losses), self.losses, color='red')
        plt.xlabel('time steps')
        plt.title("losses through backward progress")
        plt.savefig("./losses.png")
        plt.clf()
        plt.plot(len(self.accs), self.accs, color='green')
        plt.xlabel('time steps')
        plt.title("accuracies through backward progress")
        plt.savefig("./accs.png")
        plt.clf()

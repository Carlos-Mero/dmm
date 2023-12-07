import torch
import os
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from accelerate import Accelerator
from tqdm import tqdm

class CIFAR10_cnn(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        return self.net(x)

    def re_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.normal_(m.weight)



class trainer():
    def __init__(self):
        if not os.path.exists("../datasets/"):
            os.makedirs("../datasets/")
        if not os.path.exists("../datasets/CIFAR10_cnn_models/"):
            os.makedirs("../datasets/CIFAR10_cnn_models/")
        self.CIFAR10_train = datasets.CIFAR10(
            root="../datasets/",
            train=True,
            download=True,
            transform=transforms.ToTensor()
        )
        self.CIFAR10_test = datasets.CIFAR10(
            root="../datasets/",
            train=False,
            download=True,
            transform=transforms.ToTensor()
        )
        self.accelerator = Accelerator()
        self.device = self.accelerator.device
        self.loss_fn = nn.CrossEntropyLoss()
        self.model = CIFAR10_cnn().to(self.device)  # Update this to your CIFAR10 CNN model
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.01)
        self.train_loader = DataLoader(self.CIFAR10_train, 128)
        self.test_loader = DataLoader(self.CIFAR10_test, 128)
        self.model, self.optimizer = self.accelerator.prepare(
                self.model, self.optimizer)
        self.train_loader, self.test_loader = self.accelerator.prepare(
                self.train_loader, self.test_loader)
        self.model_count = 0

    def train(self):
        self.model.train()
        for _ in tqdm(range(30)):
            for (X, y) in self.train_loader:
                pred = self.model(X)
                loss = self.loss_fn(pred, y)
                self.accelerator.backward(loss)
                self.optimizer.step()
                self.optimizer.zero_grad()
        torch.save(self.model.state_dict(), f"../datasets/CIFAR10_cnn_models/{self.model_count}.pt")
        print(f"Finished training cnn on CIFAR10 [{self.model_count}/1024]")
        self.model_count += 1
    def test(self):
        size = len(self.CIFAR10_test)
        num_batches = len(self.test_loader)
        self.model.eval()
        tloss, tcorrect = 0.0, 0.0
        with torch.no_grad():
            for X, y in self.test_loader:
                X = X.to(self.device)
                y = y.to(self.device)
                pred = self.model(X)
                tloss += self.loss_fn(pred, y)
                tcorrect += (pred.argmax(1) == y).type(torch.float).sum().item()
        tloss /= num_batches
        #tloss = tloss.item()
        tcorrect /= size
        tcorrect *= 100
        print(f"Current Test Error: {tloss:>8f}")
        print(f"Current Test Accuracy: {tcorrect:>0.01f}%")
        #return (tloss, tcorrect)

    def generate_model_data(self):
        while self.model_count < 1024:
            self.train()

if __name__ == "__main__":
    t = trainer()
    t.generate_model_data()
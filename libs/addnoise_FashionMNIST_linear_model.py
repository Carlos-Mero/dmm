import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from matplotlib.colors import TwoSlopeNorm
from torch import optim
import matplotlib.pyplot as plt
from torchvision import datasets
from torchvision import transforms
from accelerate import Accelerator
import numpy as np

class MNIST_linear(nn.Module):
    def __init__(self):
        super().__init__()
        self.flat = nn.Flatten()
        self.net = nn.Linear(784, 10, bias=False)
    def forward(self, x):
        return self.net(self.flat(x))
    def re_init(self):
        nn.init.normal_(self.net.weight)

class test_with_noise():
    def __init__(self, noise, model, outputfile):
        self.MNIST_test = datasets.FashionMNIST(
            root="assets/datasets/",
            train=False,
            download=False,
            transform=transforms.ToTensor()
        )
        self.accelerator = Accelerator()
        self.device = self.accelerator.device
        self.loss_fn = nn.CrossEntropyLoss()
        self.test_loader = DataLoader(self.MNIST_test, 128)
        self.test_loader = self.accelerator.prepare(self.test_loader)
        self.model_count = 0
        self.noise = noise
        self.outputfile = outputfile
        self.model = model.to(self.device)
        self.accelerator = Accelerator()

    def preview_parameters(self):
        for name, param in self.model.named_parameters():
            if 'weight' in name:
                weights = param.cpu().detach().numpy()
                k = weights.shape[1] / weights.shape[0]
                cmap = plt.get_cmap('bwr')
                vmin = weights.min()
                vmax = weights.max()
                if vmin == vmax:
                    norm = None
                else:
                    norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)

                plt.imshow(weights, cmap=cmap, norm=norm, aspect=k)
                plt.colorbar()
                plt.title(f'Visualization with Noise: {self.cumulative_noise}')
                save_path = os.path.join(self.outputfile, f'{name}_with_noise_{self.cumulative_noise:.4f}.png')
                plt.savefig(save_path)
                plt.close()

    def test(self):
        size = len(self.MNIST_test)
        num_batches = len(self.test_loader)
        self.model.eval()
        tloss, tcorrect = 0.0, 0.0

        for param in self.model.parameters():
            param.data = param.data*np.sqrt(1-self.noise) + torch.randn_like(param.data) * self.noise

        self.cumulative_noise += self.noise  # Update the cumulative noise
                    
        with torch.no_grad():
            for X, y in self.test_loader:
                X = X.to(self.device)
                y = y.to(self.device)
                pred = self.model(X)
                tloss += self.loss_fn(pred, y)
                tcorrect += (pred.argmax(1) == y).type(torch.float).sum().item()
        tloss /= num_batches
        tcorrect /= size
        tcorrect *= 100
        print(f"Current Test Error: {tloss:>8f}")
        print(f"Current Test Accuracy: {tcorrect:>0.01f}%")
        self.preview_parameters()
        return tloss.item(), tcorrect

# Load the model
model = MNIST_linear()

step_of_model = 10
path = 'assets/datasets/FashionMNIST_linear_models/' + str(step_of_model) + '.pt'
model.load_state_dict(torch.load(path))

noise = 0
if not os.path.exists("assets/datasets/params_with_noise/FashionMNIST_linear_step_" + str(step_of_model) + "/"):
    os.makedirs("assets/datasets/params_with_noise/FashionMNIST_linear_step_" + str(step_of_model) + "/")
outputfile = "assets/datasets/params_with_noise/FashionMNIST_linear_step_" + str(step_of_model) + "/"
test_with_noise = test_with_noise(noise, model, outputfile)

noise_values = []
loss_values = []
acc_values = []
steps = []

#set num_diffusion_timesteps
n = 1000

config = {
    'diffusion': {
        'noise_schedule_type': 'linear',
        'noise_start': 0.02,
        'noise_end': 0.0001,
        'num_diffusion_timesteps': n
    }
}

def get_noise_schedule(noise_schedule_type, *, noise_start, noise_end, num_diffusion_timesteps):
    if noise_schedule_type == 'linear':
        noises = np.linspace(noise_start, noise_end, num_diffusion_timesteps, dtype=np.float64)
    else:
        raise NotImplementedError(noise_schedule_type)

    assert noises.shape == (num_diffusion_timesteps,)
    return noises

noise_schedule = get_noise_schedule(**config['diffusion'])

#add noise
for i in range(n):
    noise = noise_schedule[i]
    print(f"Current Noise: {noise}")
    test_with_noise.noise = noise
    loss, acc = test_with_noise.test()
    steps.append(i)
    noise_values.append(noise)
    loss_values.append(loss)
    acc_values.append(acc)


# Plot loss and accuracy
plt.plot(steps, loss_values, marker='o')
plt.xlabel('step')
plt.ylabel('Test Loss')
plt.title('Test Loss with Noise')
save_path = os.path.join(outputfile, 'loss_curve.png')
plt.savefig(save_path)
plt.close()

plt.plot(steps, acc_values, marker='o', color='r')
plt.xlabel('step')
plt.ylabel('Test Accuracy (%)')
plt.title('Test Accuracy with Noise')

save_path = os.path.join(outputfile, 'accuracy_curve.png')
plt.savefig(save_path)
plt.close()

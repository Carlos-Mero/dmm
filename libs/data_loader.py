import os
import glob
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

class param_set(Dataset):
    def __init__(self, data):
        self.data = data
    def __getitem__(self, index):
        return self.data[index]
    def __len__(self):
        return len(self.data)

def collect_single_file(model, state_path):
    # There's Many Things Left to DO!!
    model.load_state_dict(torch.load(state_path, map_location=torch.device("cpu")))
    parameters = model.state_dict()
    for name, parameters in parameters.items():
        return parameters.data

def get_dataset(config):
    # Not Completely Implemented Yet!!
    if config.dataset.name == 'mnist-linear':
        from assets.scripts.MNIST_linear_models import MNIST_linear
        model = MNIST_linear()
        tensors = []
        file_paths = glob.glob(os.path.join(os.getcwd(), config.dataset.path, "*"), recursive=False)
        for path in file_paths:
            data = collect_single_file(model, path)
            pad_amount = config.model.resolution[0] - data.shape[-2]
            pad_dims = (0, 0, 0, pad_amount)
            data = F.pad(data, pad_dims, value=0)
            data = data.view(1, config.model.resolution[0], config.model.resolution[1])
            tensors.append(data)
        data_loader = DataLoader(param_set(tensors), batch_size=config.train.batch_size, shuffle=True)
        return data_loader, None # We will not return a test set here.
    else:
        raise NotImplementedError('get_dataset')

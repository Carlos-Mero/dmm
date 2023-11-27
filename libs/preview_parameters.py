import torch
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import os

def preview_parameters(model, path, pic_name):
    os.makedirs(path, exist_ok=True)
    state_dict = torch.load(model)

    for name, weights in state_dict.items():
        if 'weight' in name:
            weights = weights.cpu().numpy()
            k = weights.shape[1]/weights.shape[0]
            cmap = plt.get_cmap('bwr')
            vmin = weights.min()
            vmax = weights.max()
            if vmin == vmax:
                norm = None
            else:
                norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
                
            plt.imshow(weights, cmap=cmap, norm=norm, aspect= k)
            plt.colorbar()
            plt.savefig(os.path.join(path, f'{pic_name}_{name}.png'))
            plt.clf()

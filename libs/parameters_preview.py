import torch
import matplotlib.pyplot as plt
import os
from torchvision.utils import save_image

def preview_parameters(model, path):
    if not os.path.exists(path):
        os.makedirs(path)

    for i, (name, param) in enumerate(model.named_parameters()):
        if param.requires_grad and len(param.shape) == 2:
            weights = (param - param.min()) / (param.max() - param.min())
            img = torch.zeros(weights.shape[0], weights.shape[1], 3)
            img[weights > 0.5, 0] = weights[weights > 0.5]
            img[weights <= 0.5, 2] = weights[weights <= 0.5]
            
            save_image(img, os.path.join(path, f'layer_{i}_{name}.png'))


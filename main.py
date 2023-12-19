import os
import torch
from configs.dmm_unet_small_mnistlinear import get_config as mnist_linear_configs
from libs.diffusion import Diffusion
import numpy as np
import argparse
import sys

def parse():
    parser = argparse.ArgumentParser(description=globals()["__doc__"])
    parser.add_argument("--config", type=str, required=True, help="Name of the config")
    parser.add_argument("--seed", type=int, default=1145, help="Random seed")
    parser.add_argument("--train", action="store_true", help="Enable training process")
    parser.add_argument("--sample", action="store_true", help="Enable sampling, which disables training")

    args = parser.parse_args()
    return args

def main():
    args = parse()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    if (args.config == "mnist-linear"):
        config = mnist_linear_configs()
    else:
        raise NotImplementedError("No such config")
    os.makedirs(config.log_path, exist_ok=True)
    diffuser = Diffusion(config)

    if args.sample:
        diffuser.sample()
    elif args.train:
        diffuser.train()
    else:
        raise NotImplementedError("No such method")

if __name__ == "__main__":
    sys.exit(main())

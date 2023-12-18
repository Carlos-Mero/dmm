import os
import logging
import numpy as np
import torch
from torch import optim
from tqdm import tqdm

from unet import Unet
from helper import EMAHelper
from accelerate import Accelerator
from data_loader import get_dataset
from losses import loss_registry

def get_beta_schedule(beta_schedule_type, *, beta_start, beta_end, num_diffusion_timesteps):
    if beta_schedule_type == 'linear':
        betas = np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64)
    else:
        raise NotImplementedError(beta_schedule_type)

    assert betas.shape == (num_diffusion_timesteps,)
    return betas

def get_optimizer(config, parameters):
    if config.optim.optimizer == 'adamw':
        return optim.AdamW(parameters, **config.optimizer)
    else:
        raise NotImplementedError(
            'Optimizer {} not understood.'.format(config.optim.optimizer))

class Diffusion(object):
    def __init__(self, config):
        self.config = config
        self.accelerator = Accelerator()
        self.device = self.accelerator.device
        betas = get_beta_schedule(
            beta_schedule=config.diffusion.beta_schedule,
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
        )
        betas = self.betas = torch.from_numpy(betas).float().to(self.device)
        self.num_timesteps = betas.shape[0]
        alphas = 1.0 - betas
        alphas_cumprod = alphas.cumprod(dim=0)
        alphas_cumprod_prev = torch.cat(
            [torch.ones(1).to(self.device), alphas_cumprod[:-1]], dim=0
        )
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        self.model_var_type = config.model.var_type
        if self.model_var_type == "fixedlarge":
            self.logvar = betas.log()
            # torch.cat(
            # [posterior_variance[1:2], betas[1:]], dim=0).log()
        elif self.model_var_type == "fixedsmall":
            self.logvar = posterior_variance.clamp(min=1e-20).log()

    def train(self):
        config = self.config
        train_loader, test_loader = get_dataset(config) # Currently we won't use test_loader
        model = Unet(config)
        if self.config.ema.enabled:
            ema_helper = EMAHelper(mu=self.config.ema.ema_rate)
            ema_helper.register(model)
        else:
            ema_helper = None
        optimizer = get_optimizer(self.config, model.parameters())
        model, optimizer, train_loader = self.accelerator.prepare(model, optimizer, train_loader)

        start_epoch, step = 0, 0
        # --------------------------------------------------------------------------------------
        # TODO
        # loading weights from previous checkpoints
        # --------------------------------------------------------------------------------------
        # --------------------------------------------------------------------------------------

        for epoch in tqdm(range(start_epoch, self.config.train.n_epochs)):
            data_time = 0
            for i, (x, y) in enumerate(train_loader):
                n = x.size(0)
                model.train()
                step += 1

                x = x.to(self.device)
                # x = data_transform(self.config, x)
                e = torch.randn_like(x)
                b = self.betas

                # antithetic sampling
                t = torch.randint(
                    low=0, high=self.num_timesteps, size=(n // 2 + 1,)
                ).to(self.device)
                t = torch.cat([t, self.num_timesteps - t - 1], dim=0)[:n]
                loss = loss_registry[config.train.loss_type](model, x, t, e, b)

                logging.info(
                    f"step: {step}, loss: {loss.item()}, data time: {data_time / (i+1)}"
                )

                optimizer.zero_grad()
                self.accelerator.backward(loss)

                try:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), config.optimizer.grad_clip
                    )
                except Exception:
                    pass
                optimizer.step()

                if self.config.ema.enabled:
                    ema_helper.update(model)

                if step % self.config.training.snapshot_freq == 0 or step == 1:
                    states = [
                        model.state_dict(),
                        optimizer.state_dict(),
                        epoch,
                        step,
                    ]
                    if self.config.ema.enabled:
                        states.append(ema_helper.state_dict())

                    torch.save(
                        states,
                        os.path.join(self.config.log_path, "ckpt_{}.pth".format(step)),
                    )
                    torch.save(states, os.path.join(self.config.log_path, "ckpt.pth"))

    def sample(self):
        # --------------------------------------------------------------------------------------
        # TODO
        # generating sample models
        # --------------------------------------------------------------------------------------
        raise NotImplementedError('Diffusion.sample')

    def test(self):
        raise NotImplementedError('Diffusion.test')

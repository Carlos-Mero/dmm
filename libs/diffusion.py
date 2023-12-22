import os
import logging
import numpy as np
import torch
from torch import optim
from tqdm import tqdm

from libs.unet import Unet
from libs.helper import EMAHelper, count_params
from accelerate import Accelerator
from libs.data_loader import get_dataset
from libs.losses import loss_registry

def get_beta_schedule(beta_schedule_type, *, beta_start, beta_end, num_diffusion_timesteps):
    if beta_schedule_type == 'linear':
        betas = np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64)
    else:
        raise NotImplementedError(beta_schedule_type)

    assert betas.shape == (num_diffusion_timesteps,)
    return betas

def get_optimizer(config, parameters):
    if config.optimizer_name == 'adamw':
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
            beta_schedule_type=config.diffusion.beta_schedule,
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
        count_params(model)
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

        for epoch in tqdm(range(start_epoch, self.config.train.n_epoches)):
            data_time = 0
            for i, x in enumerate(train_loader):
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

                if step % self.config.train.snapshot_freq == 0 or step == 1:
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

    def sample(self, nnet_path, sample_init=None, only_last=True):
        if nnet_path is None:
            raise EnvironmentError('Need to specfy the path to the pretrained model weight.')
        config = self.config
        model = Unet(config)
        model.load_state_dict(nnet_path)
        model = self.accelerator.prepare(model)
        model.eval()
        seq = torch.linspace(0, config.diffusion.num_diffusion_timesteps, config.sample.time_steps)
        x = torch.randn(
            config.sample.n_samples,
            config.model.in_channels,
            config.model.resolution,
            device=self.device
        ) if sample_init is None else sample_init
        from libs.helper import ddim_steps
        xs = ddim_steps(x, seq, model, self.betas, config.sample.eta)
        x = xs
        if only_last:
            x = x[0][-1]
        return x

    def visualize(self, nnet_path, sample_init):
        config = self.config
        xs = self.sample(nnet_path=nnet_path, sample_init=sample_init, only_last=False)
        xs = xs[0]
        steps = xs.shape[0]
        if config.name == 'dmm_mnistlinear':
            from libs.preview_parameters import preview_parameters as view
            from assets.scripts.MNIST_linear_models import MNIST_linear as Model
            from libs.eval import MNIST_tester
            model = Model()
            model.net.weight.data = xs[0]
            view(model, config.log_path, "init")
            model.net.weight.data = xs[steps // 2]
            view(model, config.log_path, "mid")
            model.net.weight.data = xs[-1]
            view(model, config.log_path, "end")
            tester = MNIST_tester()
            for i in range(steps):
                model.net.weight.data = xs[i]
                tester.test(model)
            tester.draw_curves()
        else:
            raise NotImplementedError()

    def test(self):
        raise NotImplementedError('Diffusion.test')

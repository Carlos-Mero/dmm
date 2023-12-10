import ml_collections


def d(**kwargs):
    """Helper of creating a config dict."""
    return ml_collections.ConfigDict(initial_dictionary=kwargs)

def get_config():
    config = ml_collections.ConfigDict()

    config.seed = 1145

    config.model = d(
        in_channels=1,
        out_channels=1,
        latent_channels=128,
        num_res_blocks=1,
        latent_channel_multipilers=(1, 2, 2),
        resolution=(16, 784),
        attn_resolutions = ((8, 392)),
        dropout=0.1,
        resamp_with_conv=True,
        var_type="fixedlarge",
    )

    config.train = d(
        loss_type='simple',
        n_epoches=10000,
        n_iters=5000000,
        snapshot_freq=5000,
        validation_frec=2000,
        batch_size=128,
    )

    config.optimizer = d(
        name='adamw',
        lr=0.0002,
        weight_decay=0.03,
        betas=(0.99, 0.99),
        grad_clip=1.0,
    )

    config.diffusion = d(
        beta_schedule='linear',
        beta_start=0.0001,
        beta_end=0.02,
        num_diffusion_timesteps=1000
    )

    config.dataset = d(
        name='mnist-linear',
        path='assets/datasets/MNIST_models/'
    )

    config.ema = d(
        enabled=False,
        ema_rate=0.9999,
    )

    return config

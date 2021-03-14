
def gan_hyperparams():
    hypers = dict(
        batch_size=53,
        z_dim=100,
        data_label=1,
        label_noise=0.1,
        discriminator_optimizer=dict(
            type="Adam",  # Any name in nn.optim like SGD, Adam
            lr=2e-4,
            betas=(0.5, 0.999),
            #weight_decay=2e-3
            # You an add extra args for the optimizer here
        ),
        generator_optimizer=dict(
            type="Adam",  # Any name in nn.optim like SGD, Adam
            lr=2e-4,
            betas=(0.5, 0.999)
            # You an add extra args for the optimizer here
        ),
    )
    return hypers

import torch
import torch.nn as nn
from typing import Callable
import os
from .BushDataLoader import BushDataLoader
from torch.utils.data import DataLoader
from torch.optim.optimizer import Optimizer
from .gan import Discriminator, SNDiscriminator, Generator
import torch.optim as optim
import IPython.display
import tqdm
import numpy as np
import sys
import matplotlib.pyplot as plt
import cs236781.plot as plot


def discriminator_loss_fn(y_data, y_generated, data_label=0, label_noise=0.0):
    """
    Computes the combined loss of the discriminator given real and generated
    data using a binary cross-entropy metric.
    This is the loss used to update the Discriminator parameters.
    :param y_data: Discriminator class-scores of instances of data sampled
    from the dataset, shape (N,).
    :param y_generated: Discriminator class-scores of instances of data
    generated by the generator, shape (N,).
    :param data_label: 0 or 1, label of instances coming from the real dataset.
    :param label_noise: The range of the noise to add. For example, if
    data_label=0 and label_noise=0.2 then the labels of the real data will be
    uniformly sampled from the range [-0.1,+0.1].
    :return: The combined loss of both.
    """
    assert data_label == 1 or data_label == 0
    # TODO:
    #  Implement the discriminator loss.
    #  See pytorch's BCEWithLogitsLoss for a numerically stable implementation.
    bce = nn.BCELoss()

    r1, r2 = data_label - label_noise / 2, data_label + label_noise / 2
    labels = torch.distributions.uniform.Uniform(r1, r2).sample(y_data.shape)
    loss_data = bce(y_data, labels.to(device=y_data.device))

    r1, r2 = 0 - label_noise / 2, 0 + label_noise / 2
    labels = torch.distributions.uniform.Uniform(r1, r2).sample(y_data.shape)
    loss_generated = bce(y_generated, labels.to(device=y_generated.device))

    return loss_data + loss_generated


def wgan_discriminator_loss_fn(y_data, y_generated):
    loss_data = y_data.mean(0).view(1)
    loss_generated = y_generated.mean(0).view(1)
    return loss_data, loss_generated


def generator_loss_fn(y_generated, data_label=0):
    """
    Computes the loss of the generator given generated data using a
    binary cross-entropy metric.
    This is the loss used to update the Generator parameters.
    :param y_generated: Discriminator class-scores of instances of data
    generated by the generator, shape (N,).
    :param data_label: 0 or 1, label of instances coming from the real dataset.
    :return: The generator loss.
    """
    assert data_label == 1 or data_label == 0
    # TODO:
    #  Implement the Generator loss.
    #  Think about what you need to compare the input to, in order to
    #  formulate the loss in terms of Binary Cross Entropy.
    labels = torch.full_like(y_generated, data_label, device=y_generated.device)
    bce = nn.BCELoss()
    loss = bce(y_generated, labels)
    return loss


def wgan_generator_loss_fn(y_generated):
    loss = y_generated.mean(0).view(1)
    return loss


def train_batch_discriminator(
        dsc_model: Discriminator,
        gen_model: Generator,
        dsc_loss_fn: Callable,
        dsc_optimizer: Optimizer,
        x_data: DataLoader,
):
    gen_data = gen_model.sample(len(x_data))

    dsc_loss = dsc_loss_fn(dsc_model(x_data), dsc_model(gen_data))
    dsc_loss.backward()
    dsc_optimizer.step()

    return dsc_loss

def w_train_batch_discriminator(
        dsc_model: Discriminator,
        gen_model: Generator,
        dsc_loss_fn: Callable,
        dsc_optimizer: Optimizer,
        x_data: DataLoader,
        n_critic=5,
        c=0.01
):
    batch_size = x_data.shape[0]
    dsc_batch_losses = []
    # Discrimnator update, n_critic iterations
    # clip discrimnator parameters, then update
    for d_iter in range(n_critic):
        dsc_model.zero_grad()
        for p in dsc_model.parameters():
            p.data.clamp_(-c, c)
        fake_imgs = gen_model.sample(n=batch_size, with_grad=False)
        y_data = dsc_model(x_data)
        y_generated = dsc_model(fake_imgs)
        dsc_loss_real, dsc_loss_fake = dsc_loss_fn(y_data, y_generated)
        dsc_loss_real.backward()
        dsc_loss_fake.backward(torch.Tensor([-1]).to(dsc_model.device))
        dsc_iter_loss = dsc_loss_real - dsc_loss_fake
        dsc_batch_losses.append(dsc_iter_loss.item())
        dsc_optimizer.step()
    dsc_loss = np.mean(dsc_batch_losses)

    return dsc_loss


def train_batch_generator(
        dsc_model: Discriminator,
        gen_model: Generator,
        gen_loss_fn: Callable,
        gen_optimizer: Optimizer,
        x_data: DataLoader,
):
    gen_data = gen_model.sample(len(x_data), with_grad=True)

    gen_loss = gen_loss_fn(dsc_model(gen_data))
    gen_loss.backward()
    gen_optimizer.step()

    return gen_loss



def train_batch(
        dsc_model: Discriminator,
        gen_model: Generator,
        dsc_loss_fn: Callable,
        gen_loss_fn: Callable,
        dsc_optimizer: Optimizer,
        gen_optimizer: Optimizer,
        x_data: DataLoader,
        n_critic=1,
        c=0.01
):
    """
    Trains a GAN for over one batch, updating both the discriminator and
    generator.
    :return: The discriminator and generator losses.
    """
    dsc_model.train(True)
    gen_model.train(True)

    # ====== YOUR CODE: ======
    if c == 0:
        dsc_loss = train_batch_discriminator(
                                                dsc_model,
                                                gen_model,
                                                dsc_loss_fn,
                                                dsc_optimizer,
                                                x_data)
    else:
        dsc_loss = w_train_batch_discriminator(
                                                dsc_model,
                                                gen_model,
                                                dsc_loss_fn,
                                                dsc_optimizer,
                                                x_data,
                                                n_critic,
                                                c)
    gen_loss = train_batch_generator(
                                    dsc_model,
                                    gen_model,
                                    gen_loss_fn,
                                    gen_optimizer,
                                    x_data)

    # ========================

    return dsc_loss.item(), gen_loss.item()


class Trainer:
    def __init__(self, hp, dsc_cls, gen_cls, name='gan', wgan=False):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        batch_size = hp['batch_size']
        z_dim = hp['z_dim']
        # Data
        self.dl_train = BushDataLoader(batch_size, shuffle=True)
        # im_size = self.dl_train.im_size

        # Model
        self.dsc = dsc_cls(self.dl_train.im_size).to(self.device)
        self.gen = gen_cls(z_dim, featuremap_size=4).to(self.device)
        print(self.gen)
        print(self.dsc)

        self.dsc_optimizer = self.create_optimizer(self.dsc.parameters(), hp['discriminator_optimizer'])
        self.gen_optimizer = self.create_optimizer(self.gen.parameters(), hp['generator_optimizer'])
        print(self.dsc_optimizer)

        # Loss
        if wgan:
            self.dsc_loss_fn = wgan_discriminator_loss_fn
            self.gen_loss_fn = wgan_generator_loss_fn
        else:
            self.dsc_loss_fn = lambda y_data, y_generated: discriminator_loss_fn(y_data, y_generated,
                                                                                 hp['data_label'], hp['label_noise'])
            self.gen_loss_fn = lambda y_generated: generator_loss_fn(y_generated, hp['data_label'])
            

        # Training
        self.checkpoint_file = f'checkpoints/{name}'
        self.checkpoint_file_final = f'{self.checkpoint_file}_final'
        if os.path.isfile(f'{self.checkpoint_file}.pt'):
            os.remove(f'{self.checkpoint_file}.pt')
        self.n_critic = hp['n_critic']
        self.c = hp['c']

    # Optimizer
    @staticmethod
    def create_optimizer(model_params, opt_params):
        opt_params = opt_params.copy()
        optimizer_type = opt_params['type']
        opt_params.pop('type')
        return optim.__dict__[optimizer_type](model_params, **opt_params)

    def train(self):
        num_epochs = 400
        max_inc_score = 0
        self.dsc.train(True)
        self.gen.train(True)
        self.inc_score_epochs=0
        try:
            dsc_avg_losses, gen_avg_losses = [], []
            for epoch_idx in range(num_epochs):
                # We'll accumulate batch losses and show an average once per epoch.
                dsc_losses, gen_losses = [], []
                print(f'--- EPOCH {epoch_idx + 1}/{num_epochs} ---')

                with tqdm.tqdm(total=len(self.dl_train.batch_sampler), file=sys.stdout) as pbar:
                    for batch_idx, (x_data, _) in enumerate(self.dl_train):
                        x_data = x_data.to(self.device)
                        dsc_loss, gen_loss = train_batch(self.dsc,
                                                        self.gen,
                                                        self.dsc_loss_fn,
                                                        self.gen_loss_fn,
                                                        self.dsc_optimizer,
                                                        self.gen_optimizer,
                                                        x_data,
                                                        self.n_critic,
                                                        self.c)
                        dsc_losses.append(dsc_loss)
                        gen_losses.append(gen_loss)
                        pbar.update()

                dsc_avg_losses.append(np.mean(dsc_losses))
                gen_avg_losses.append(np.mean(gen_losses))
                print(f'Discriminator loss: {dsc_avg_losses[-1]}')
                print(f'Generator loss:     {gen_avg_losses[-1]}')

                samples = self.gen.sample(5, with_grad=False)
                fig, _ = plot.tensors_as_images(samples.cpu(), figsize=(6, 2))
                IPython.display.display(fig)
                plt.close(fig)

                if num_epochs - epoch_idx <= self.inc_score_epochs:
                    print("Calculating inception score...")
                    #inc_score = 0#calc_inception_score(self.gen_model)
                    #print(f'Epoch {epoch_idx + 1}, Inception score:{inc_score}')
                    #if inc_score > max_inc_score:
                    #    max_inc_score = inc_score
                    #    torch.save(self.gen_model, self.checkpoint_file)
                    #    print(f'Saved checkpoint.')
            print(f"Finished training! best inception score is: {max_inc_score}")
            return max_inc_score
        except KeyboardInterrupt as e:
            print('\n *** Training interrupted by user')
        finally:
            self.dsc.eval()
            self.gen.eval()



    @staticmethod
    def save_checkpoint(gen_model, dsc_losses, gen_losses, checkpoint_file):
        """
        Saves a checkpoint of the generator, if necessary.
        :param gen_model: The Generator model to save.
        :param dsc_losses: Avg. discriminator loss per epoch.
        :param gen_losses: Avg. generator loss per epoch.
        :param checkpoint_file: Path without extension to save generator to.
        """

        saved = True
        checkpoint_file = f"{checkpoint_file}.pt"

        # TODO:
        #  Save a checkpoint of the generator model. You can use torch.save().
        #  You should decide what logic to use for deciding when to save.
        #  If you save, set saved to True.
        if saved:
            saved_state = dict(
                # best_acc=best_acc,
                # ewi=epochs_without_improvement,
                model_state=gen_model.state_dict(),
            )
            torch.save(gen_model, checkpoint_file)

        return saved


class GANTrainer(Trainer):
    def __init__(self, hp):
        super(GANTrainer, self).__init__(hp, Discriminator, Generator, name='gan', wgan=False)


class SNGANTrainer(Trainer):
    def __init__(self, hp):
        super().__init__(hp, SNDiscriminator, Generator, name='sngan', wgan=False)


class WGANTrainer(Trainer):
    def __init__(self, hp):
        super().__init__(hp, Discriminator, Generator, name='wgan', wgan=True)

import os
import numpy as np
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F

import visdom

from latentgoalexplo.actors.meta_actors import *


CUDA = torch.cuda.is_available()
device = torch.device("cuda" if CUDA else "cpu")


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class Channelize(nn.Module):
    def __init__(self, n_channels):
        super(Channelize, self).__init__()
        self.n_channels = n_channels

    def forward(self, input):
        return input.view(input.size(0), self.n_channels, 4, 4)


class BetaVAE(nn.Module):
    def __init__(self, n_latents, Ta, capacity, capacity_change_duration, n_channels, dim_ergo,
                 visdom_env="main", visdom_record=True):
        super(BetaVAE, self).__init__()

        self.n_latents = n_latents
        self.dim_ergo = dim_ergo

        self.n_iters = 0
        self.Ta = Ta
        self.capacity = capacity
        self.capacity_change_duration = capacity_change_duration

        self.n_channels = n_channels
        self.generate_architecture()

        self.visdom_record = visdom_record
        if self.visdom_record:
            self.visdom = visdom.Visdom(env=visdom_env)
        self.visdom_wins = dict()

    def generate_architecture(self):

        # Works only for n_channels * 64 * 64 images
        self.encoder = nn.Sequential(
                nn.Conv2d(self.n_channels, 32, kernel_size=4, stride=2, padding=1),  # b, 32, 32, 32
                nn.ReLU(),
                nn.Conv2d(32, 32, kernel_size=4, stride=2, padding=1),  # b, 32, 16, 16
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # b, 64, 8, 8
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1),  # b, 64, 4, 4
                nn.ReLU(),
                Flatten(),
                nn.Linear(64 * 4 * 4, 256),
                nn.ReLU(),
                nn.Linear(256, 2 * self.n_latents)
        )

        self.decoder = nn.Sequential(
                nn.Linear(self.n_latents, 4 * 4 * 64),
                nn.ReLU(),
                Channelize(n_channels=64),
                nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1),  # b, 32, 8, 8
                nn.ReLU(),
                nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # b, 32, 16, 16
                nn.ReLU(),
                nn.ConvTranspose2d(32, 32, kernel_size=4, stride=2, padding=1),  # b, 32, 32, 32
                nn.ReLU(),
                nn.ConvTranspose2d(32, self.n_channels, kernel_size=4, stride=2, padding=1)  # b, n_channels, 64, 64
        )

        self.ball_classifier = nn.Linear(1, 1)
        self.ergo_classifier = nn.Linear(self.dim_ergo, self.dim_ergo)
        self.setup_classifier = nn.Linear(3, 6)

    def encode(self, x):
        return torch.chunk(self.encoder(x), 2, dim=1)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        setup = self.setup_classifier(z[:, :3])
        ball_pos = self.ball_classifier(z[:, 3:4])
        ergo_pos = self.ergo_classifier(z[:, 4:4 + self.dim_ergo])
        return self.decode(z), mu, logvar, setup, ball_pos, ergo_pos

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def loss_function(self, recon_x, x, mu, logvar, setup, true_setup, ball_pos, true_ball_pos,
                      ergo_pos, true_ergo_pos, beta, c, capacity):
        BCE = F.binary_cross_entropy_with_logits(recon_x, x, size_average=False) / x.size()[0]

        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), 1)

        # Version 1
        # We force the latents of to have a KL=capacity, on average on a minibatch
        KLD_loss = c * beta * torch.abs(KLD.mean() - capacity)

        # Version 2
        # We force the latents of each sample to have a KL=capacity
        # KLD_loss = torch.abs(KLD - capacity).mean()
        # KLD_loss = c * beta * KLD_loss

        setup_loss = 10 * F.cross_entropy(setup, true_setup, ignore_index=-100)

        weights = (true_ball_pos != 0).max(dim=1)[0]
        ball_loss = 10 * F.binary_cross_entropy_with_logits(ball_pos, true_ball_pos, reduction='none').mean(1)
        ball_loss = (ball_loss * weights.float()).sum()

        weights = (true_ergo_pos != 0).max(dim=1)[0]
        ergo_loss = 10 * F.binary_cross_entropy_with_logits(ergo_pos, true_ergo_pos, reduction='none').mean(1)
        ergo_loss = (ergo_loss * weights.float()).sum()

        return BCE, KLD_loss, KLD.mean(), setup_loss, ball_loss, ergo_loss

    def KLD(self, data_loader):
        self.eval()
        KLD = torch.zeros(self.n_latents).to(device)
        with torch.no_grad():
            for i, (data, _, _, _) in enumerate(data_loader):
                if i > 100:
                    break
                data = data.to(device)
                _, mu, logvar, _, _, _ = self.forward(data)
                # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
                KLD += -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp(), dim=0)
        KLD /= (i + 1)
        return KLD

    def _perform_epoch(self, train_loader, optimizer, beta):
        self.train()
        for batch_idx, (images, ball_pos, apex_setup, ergo_pos) in enumerate(train_loader):
            images = images.to(device)
            ball_pos = ball_pos.to(device)
            apex_setup = apex_setup.to(device)
            ergo_pos = ergo_pos.to(device)
            self.n_iters += 1
            c = min([1., 0.01 + self.n_iters / self.Ta])
            capacity = self.calc_encoding_capacity()
            optimizer.zero_grad()
            recon_batch, mu, logvar, setup, ball, ergo = self.forward(images)
            recon_loss, kld_loss, kld, setup_loss, ball_loss, ergo_loss = self.loss_function(recon_batch, images, mu,
                                                                                             logvar, setup, apex_setup,
                                                                                             ball, ball_pos, ergo,
                                                                                             ergo_pos, beta,
                                                                                             c, capacity)
            if self.n_iters % 200 == 0 and self.visdom_record:
                self.plot_losses(recon_loss, kld_loss, kld, setup_loss, ball_loss, ergo_loss)
                self.plot_latents_kld_values(self.KLD(train_loader).cpu().detach().numpy(), self.n_iters)
                self.train()
            loss = recon_loss + kld_loss + setup_loss + ball_loss + ergo_loss
            loss.backward()
            optimizer.step()
        return

    def plot_losses(self, recon_loss, kld_loss, kld, setup_loss, ball_loss, ergo_loss):
        if 'total_loss' in self.visdom_wins and self.visdom.win_exists(self.visdom_wins['total_loss']):
            self.visdom.line(X=torch.ones(1) * self.n_iters, Y=(kld_loss + recon_loss).cpu().detach().unsqueeze(0),
                             win=self.visdom_wins['total_loss'], update='append')
        else:
            self.visdom_wins['total_loss'] = self.visdom.line(X=torch.zeros(1), Y=(recon_loss + kld_loss).cpu().detach().unsqueeze(0),
                                                              opts={'title': 'Total loss'})

        if 'recon_loss' in self.visdom_wins and self.visdom.win_exists(self.visdom_wins['recon_loss']):
            self.visdom.line(X=torch.ones(1) * self.n_iters, Y=recon_loss.cpu().detach().unsqueeze(0),
                             win=self.visdom_wins['recon_loss'], update='append')
        else:
            self.visdom_wins['recon_loss'] = self.visdom.line(X=torch.zeros(1), Y=recon_loss.cpu().detach().unsqueeze(0),
                                                              opts={'title': 'Reconstruction loss'})

        if 'kld_loss' in self.visdom_wins and self.visdom.win_exists(self.visdom_wins['kld_loss']):
            self.visdom.line(X=torch.ones(1) * self.n_iters, Y=kld_loss.cpu().detach().unsqueeze(0),
                             win=self.visdom_wins['kld_loss'], update='append')
        else:
            self.visdom_wins['kld_loss'] = self.visdom.line(X=torch.zeros(1), Y=kld_loss.cpu().detach().unsqueeze(0),
                                                            opts={'title': 'KLD loss with capacity and beta'})

        if 'kld' in self.visdom_wins and self.visdom.win_exists(self.visdom_wins['kld']):
            self.visdom.line(X=torch.ones(1) * self.n_iters,
                             Y=torch.cat([kld.cpu().detach().unsqueeze(0), torch.from_numpy(
                                     np.array([self.calc_encoding_capacity()])).float()]).unsqueeze(0),
                             win=self.visdom_wins['kld'], update='append')
        else:
            self.visdom_wins['kld'] = self.visdom.line(X=torch.zeros(1),
                                                       Y=torch.cat([kld.cpu().detach().unsqueeze(0), torch.from_numpy(np.array(
                                                               [self.calc_encoding_capacity()])).float()]).unsqueeze(0),
                                                       opts={'title': 'KLD'})

        if 'setup_loss' in self.visdom_wins and self.visdom.win_exists(self.visdom_wins['setup_loss']):
            self.visdom.line(X=torch.ones(1) * self.n_iters, Y=setup_loss.cpu().detach().unsqueeze(0),
                             win=self.visdom_wins['setup_loss'], update='append')
        else:
            self.visdom_wins['setup_loss'] = self.visdom.line(X=torch.zeros(1), Y=setup_loss.cpu().detach().unsqueeze(0),
                                                            opts={'title': 'prediction loss for setup'})

        if 'ball_loss' in self.visdom_wins and self.visdom.win_exists(self.visdom_wins['ball_loss']):
            self.visdom.line(X=torch.ones(1) * self.n_iters, Y=ball_loss.cpu().detach().unsqueeze(0),
                             win=self.visdom_wins['ball_loss'], update='append')
        else:
            self.visdom_wins['ball_loss'] = self.visdom.line(X=torch.zeros(1), Y=ball_loss.cpu().detach().unsqueeze(0),
                                                            opts={'title': 'prediction loss for ball'})

        if 'ergo_loss' in self.visdom_wins and self.visdom.win_exists(self.visdom_wins['ergo_loss']):
            self.visdom.line(X=torch.ones(1) * self.n_iters, Y=ergo_loss.cpu().detach().unsqueeze(0),
                             win=self.visdom_wins['ergo_loss'], update='append')
        else:
            self.visdom_wins['ergo_loss'] = self.visdom.line(X=torch.zeros(1), Y=ergo_loss.cpu().detach().unsqueeze(0),
                                                            opts={'title': 'prediction loss for ergo'})

    def plot_latent_space(self, data_loader, sorted_latents):
        self.eval()
        indices = sorted_latents
        z_means = []
        z_sigmas = []
        for batch_idx, (data, _, _, _) in enumerate(data_loader):
            data = data.to(device)
            if batch_idx * data.size(0) > 1000:
                break
            _, mu, logvar, _, _, _ = self.forward(data)
            z_means.append(mu)
            z_sigmas.append(logvar.exp())
        z_means = torch.cat(z_means)
        z_means = z_means[:, indices[:3]]
        z_means = z_means.cpu().detach()
        # Scatter plot of latent space
        if 'latent_space' in self.visdom_wins and self.visdom.win_exists(self.visdom_wins['latent_space']):
            self.visdom.scatter(z_means, win=self.visdom_wins['latent_space'],
                                opts={'title': 'latent space', 'markersize': 10})
        else:
            self.visdom_wins['latent_space'] = self.visdom.scatter(z_means,
                                                                   opts={'title': 'latent space', 'markersize': 10})
        # Values of the variance of latents
        z_sigmas = torch.cat(z_sigmas).mean(dim=0)

        # Values of the variance of the latents of one image
        # z_sigmas = z_sigmas[0][0]
        zss_str = "capacity=" + "{0:.2f}".format(self.calc_encoding_capacity()) + ", n_iters=" + str(self.n_iters)
        for i, zss in enumerate(z_sigmas.detach()):
            string = "z{0}={1:.4f}".format(i, zss)
            zss_str += ", " + string
        if 'latents_var' in self.visdom_wins and self.visdom.win_exists(self.visdom_wins['latents_var']):
            self.visdom.text(zss_str, win=self.visdom_wins['latents_var'], append=True)
        else:
            self.visdom_wins['latents_var'] = self.visdom.text(zss_str)

    def plot_reconstruction(self, test_loader, size=8):
        self.eval()
        imgs_sampled, _, _, _ = next(iter(test_loader))
        height = 64
        width = 64
        imgs_sampled = imgs_sampled.to(device)

        recons = self.forward(imgs_sampled)[0]
        recons = torch.sigmoid(recons)

        imgs_sampled = imgs_sampled.cpu().contiguous().detach().view([-1, self.n_channels, height, width])
        imgs_sampled = imgs_sampled[:size]
        recons = recons.cpu().contiguous().detach().view([-1, self.n_channels, height, width])
        recons = recons[:size]
        truth_vs_recons = torch.cat((imgs_sampled, recons))

        if 'recon' in self.visdom_wins and self.visdom.win_exists(self.visdom_wins['recon']):
            self.visdom.images(truth_vs_recons, nrow=size, win=self.visdom_wins['recon'],
                               opts={'title': 'x vs x_tilde',
                                     'jpgquality': 100,
                                     'height': 2 * 2 * height,
                                     'width': 2 * size * width})
        else:
            self.visdom_wins['recon'] = self.visdom.images(truth_vs_recons, nrow=size,
                                                           opts={'title': 'x vs x_tilde',
                                                                 'jpgquality': 100,
                                                                 'height': 2 * 2 * height,
                                                                 'width': 2 * size * width})

    def plot_reconstruction_vs_latent(self, data_loader, sorted_latents, img, n_sigma=10):
        self.eval()
        indices = sorted_latents

        # img = next(iter(data_loader))[:1]
        img = torch.from_numpy(img).float().unsqueeze(0)  # .permute(0, 2, 3, 1)
        height = 64
        width = 64

        img = img.to(device)
        sampled_imgs = []

        for indice in indices:
            # Recover model latent variables for img
            latents = self.encode(img)[0].cpu()
            latents.data[0, indice] = 0
            new_lat = torch.zeros_like(latents)
            new_lat.data[0, indice] = 1
            for sigma in np.linspace(-3, 3, n_sigma):
                latents_sample = latents + float(sigma) * new_lat
                latents_sample = latents_sample.to(device)
                sample = torch.sigmoid(self.decode(latents_sample))
                sampled_imgs.append(sample.cpu().detach().numpy().reshape(self.n_channels, height, width))

        if 'img_vs_latent' in self.visdom_wins and self.visdom.win_exists(self.visdom_wins['img_vs_latent']):
            self.visdom.images(sampled_imgs, nrow=n_sigma, win=self.visdom_wins['img_vs_latent'],
                               opts={'title': 'Reconstruction against latent value',
                                     'jpgquality': 100,
                                     'height': self.n_latents * 80,
                                     'width': n_sigma * 80})
        else:
            self.visdom_wins['img_vs_latent'] = self.visdom.images(sampled_imgs, nrow=n_sigma,
                                                                   opts={
                                                                       'title': 'Reconstruction against latent value',
                                                                       'jpgquality': 100,
                                                                       'height': self.n_latents * 80,
                                                                       'width': n_sigma * 80})

        random_img, _, _, _ = next(iter(data_loader))
        img = random_img[:1].to(device)
        sampled_imgs = []

        for indice in indices:
            # Recover model latent variables for img
            latents = self.encode(img)[0].cpu()
            latents.data[0, indice] = 0
            new_lat = torch.zeros_like(latents)
            new_lat.data[0, indice] = 1
            for sigma in np.linspace(-3, 3, n_sigma):
                latents_sample = latents + float(sigma) * new_lat
                latents_sample = latents_sample.to(device)
                sample = torch.sigmoid(self.decode(latents_sample))
                sampled_imgs.append(sample.cpu().detach().numpy().reshape(self.n_channels, height, width))

        if 'randomimg_vs_latent' in self.visdom_wins and self.visdom.win_exists(self.visdom_wins['randomimg_vs_latent']):
            self.visdom.images(sampled_imgs, nrow=n_sigma, win=self.visdom_wins['randomimg_vs_latent'],
                               opts={'title': 'Reconstruction against latent value (random image)',
                                     'jpgquality': 100,
                                     'height': self.n_latents * 80,
                                     'width': n_sigma * 80})
        else:
            self.visdom_wins['randomimg_vs_latent'] = self.visdom.images(sampled_imgs, nrow=n_sigma,
                                                                   opts={
                                                                       'title': 'Reconstruction against latent value (random image)',
                                                                       'jpgquality': 100,
                                                                       'height': self.n_latents * 80,
                                                                       'width': n_sigma * 80})

    def plot_latents_kld_values(self, kld_latents, epoch):
        KLD = kld_latents
        if 'latents_kld' in self.visdom_wins and self.visdom.win_exists(self.visdom_wins['latents_kld']):
            self.visdom.line(X=torch.ones(1) * epoch, Y=np.expand_dims(KLD, 0),
                             win=self.visdom_wins['latents_kld'], update='append')
        else:
            self.visdom_wins['latents_kld'] = self.visdom.line(X=torch.zeros(1), Y=np.expand_dims(KLD, 0),
                                                               opts={'title': 'KLD for each latent'})

    def calc_encoding_capacity(self):
        if self.n_iters > self.capacity_change_duration:
            return self.capacity
        else:
            return self.capacity * self.n_iters / self.capacity_change_duration


class PytorchBetaVAERepresentation(object):

    def __init__(self, n_latents, initial_epochs, beta, n_channels, dim_ergo,
                 batch_size=128, learning_rate=1e-4, Ta=1, capacity=0, capacity_change_duration=1,
                 visdom_record=False, visdom_env="main", log_interval=40, store_loader_gpu=False):

        self.beta = beta
        self.n_latents = n_latents

        self.model = None
        self.n_channels = n_channels
        self.batch_size = batch_size
        self.initial_epochs = initial_epochs

        self._optimizer = None
        self.learning_rate = learning_rate
        self.net_epochs = 0
        self.Ta = Ta
        self.capacity = capacity
        self.capacity_change_duration = capacity_change_duration
        self._store_loader_gpu = store_loader_gpu

        self._typical_img = None

        self.prediction = None
        self.representation = None
        self.kld_latents = None
        self.sorted_latents = None

        self._visdom_record = visdom_record
        self._visdom_env = visdom_env
        self._log_interval = log_interval

        self.dim_ergo = dim_ergo

    def reset(self, images, ball_pos, apex_setup, ergo_pos, typical_img=None, periodic_save=False):

        if typical_img is not None:
            self._typical_img = typical_img.transpose(2, 0, 1)
        else:
            self._typical_img = images.permute(0, 3, 1, 2)[0].numpy()

        self.model = BetaVAE(self.n_latents, self.Ta, self.capacity,
                             self.capacity_change_duration, self.n_channels, self.dim_ergo,
                             self._visdom_env, self._visdom_record)
        self.model.to(device)
        self._optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        ball_pos = ball_pos.unsqueeze(-1)
        if self.dim_ergo == 1:
            ergo_pos = ergo_pos.unsqueeze(-1)

        dataset = torch.utils.data.TensorDataset(images.permute(0, 3, 1, 2), ball_pos, apex_setup - 1, ergo_pos)
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        for i in range(self.initial_epochs):
            self.model._perform_epoch(data_loader, self._optimizer, self.beta)
            if self._visdom_record and self.net_epochs % self._log_interval == 0:
                self._update_latents_order(data_loader)
                self.model.plot_reconstruction(data_loader)
                self.model.plot_reconstruction_vs_latent(data_loader, self.sorted_latents, self._typical_img)
                self.model.plot_latent_space(data_loader, self.sorted_latents)
            self.net_epochs += 1
            if periodic_save and self.net_epochs == self.initial_epochs // 10:
                torch.save(self.model.state_dict(), 'weights/' + self._visdom_env + f'iter_{self.net_epochs}')

        self._update_prediction(data_loader)
        self._update_representation(data_loader)
        self._update_latents_order(data_loader)

    def train_representation(self, images, ball_pos, apex_setup, ergo_pos, epochs):
        ball_pos = ball_pos.unsqueeze(-1)
        ergo_pos = ergo_pos.unsqueeze(-1)

        dataset = torch.utils.data.TensorDataset(images.permute(0, 3, 1, 2), ball_pos, apex_setup - 1, ergo_pos)
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        for i in range(epochs):
            self.model._perform_epoch(data_loader, self._optimizer, self.beta)
            if self._visdom_record and self.net_epochs % self._log_interval == 0:
                self._update_latents_order(data_loader)
                self.model.plot_reconstruction(data_loader)
                self.model.plot_reconstruction_vs_latent(self.sorted_latents, self._typical_img)
                self.model.plot_latent_space(data_loader, self.sorted_latents)
            self.net_epochs += 1

        self._update_latents_order(data_loader)
        self._update_prediction(data_loader)
        self._update_representation(data_loader)

    def train_representation_n_iters(self, images, ball_pos, apex_setup, ergo_pos, n_iters):
        ball_pos = ball_pos.unsqueeze(-1)
        ergo_pos = ergo_pos.unsqueeze(-1)

        dataset = torch.utils.data.TensorDataset(images.permute(0, 3, 1, 2), ball_pos, apex_setup - 1, ergo_pos)
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        epochs = n_iters * self.batch_size // len(dataset)
        for i in range(epochs):
            self.model._perform_epoch(data_loader, self._optimizer, self.beta)
            if self._visdom_record and self.net_epochs % self._log_interval == 0:
                self._update_latents_order(data_loader)
                self.model.plot_reconstruction(data_loader)
                self.model.plot_reconstruction_vs_latent(self.sorted_latents, self._typical_img)
                self.model.plot_latent_space(data_loader, self.sorted_latents)
            self.net_epochs += 1

        self._update_latents_order(data_loader)

    def estimate_kld(self, images, ball_pos, apex_setup, ergo_pos):
        ball_pos = ball_pos.unsqueeze(-1)
        ergo_pos = ergo_pos.unsqueeze(-1)

        dataset = torch.utils.data.TensorDataset(images.permute(0, 3, 1, 2), ball_pos, apex_setup - 1, ergo_pos)
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self._update_latents_order(data_loader)

    def load_model(self, filepath, typical_img=None):
        if typical_img is not None:
            self._typical_img = typical_img.transpose(2, 0, 1)

        self.model = BetaVAE(self.n_latents, self.Ta, self.capacity,
                             self.capacity_change_duration, self.n_channels, self.dim_ergo,
                             self._visdom_env, self._visdom_record)
        if CUDA:
            self.model.to(device)
            self.model.load_state_dict(torch.load(filepath))
        else:
            self.model.load_state_dict(torch.load(filepath, map_location=lambda storage, loc: storage))

        self._optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        self.model.eval()

    def act(self, *args, X_pred=None, X_train=None, y_train=None, X_test=None, y_test=None):

        if X_train is not None and X_pred is not None and X_test is not None:
            raise Exception("Calling multiple modes at once is not possible.")

        if X_train is not None:
            if len(X_train.shape) == 3:
                X_train = np.expand_dims(X_train, axis=0)
            X_train = X_train.transpose(0, 3, 1, 2)

            images = torch.from_numpy(X_train).float()
            dataset = torch.utils.data.TensorDataset(images, torch.zeros_like(images), torch.zeros_like(images).long())
            data_loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
            self.model._perform_epoch(data_loader, self._optimizer, self.beta)

        elif X_test is not None:

            if len(X_test.shape) == 3:
                X_test = np.expand_dims(X_test, axis=0)
            X_test = X_test.transpose(0, 3, 1, 2)

            images = torch.from_numpy(X_test).float()
            dataset = torch.utils.data.TensorDataset(images, torch.zeros_like(images), torch.zeros_like(images).long())
            data_loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        elif X_pred is not None:
            if len(X_pred.shape) == 3:
                X_pred = np.expand_dims(X_pred, axis=0)
            X_pred = X_pred.transpose(0, 3, 1, 2)

            images = torch.from_numpy(X_pred).float()
            dataset = torch.utils.data.TensorDataset(images, torch.zeros_like(images), torch.zeros_like(images).long(), torch.zeros_like(images))
            data_loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
            self._update_prediction(data_loader)
            self._update_representation(data_loader)

    def _update_prediction(self, data_loader):

        self.model.eval()
        pred = []

        with torch.no_grad():
            for batch_idx, (data, _, _, _) in enumerate(data_loader):
                X_train = data.to(device)
                pred.append(self.model(X_train)[0].cpu().detach().numpy())

        self.prediction = np.concatenate(pred)

    def _update_representation(self, data_loader):

        self.model.eval()
        rep = []

        with torch.no_grad():
            for batch_idx, (data, _, _, _) in enumerate(data_loader):
                X_train = data.to(device)
                rep.append(self.model.encode(X_train)[0].cpu().detach().numpy())

        self.representation = np.concatenate(rep)

    def _update_latents_order(self, data_loader):
        KLD = self.model.KLD(data_loader)
        values, indices = KLD.sort(descending=True)
        self.kld_latents = values.cpu().detach().numpy()
        self.sorted_latents = indices.cpu().detach().numpy()


# Poppy representations
model_path = os.path.dirname(os.path.abspath(__file__)) + '/weights/SupPoppimage10_B20_C20_D600'
SupPoppimage10_B20_C20_D600 = PytorchBetaVAERepresentation(n_latents=10, initial_epochs=0, beta=1, n_channels=3,
                                                           dim_ergo=2, batch_size=256)
SupPoppimage10_B20_C20_D600.load_model(model_path)
SupPoppimage10_B20_C20_D600.sorted_latents = np.array([3, 1, 2, 0, 4, 5, 7, 8, 9, 6])

# model_path = os.path.dirname(os.path.abspath(__file__)) + '/weights/Poppimage_ent'
# SupPoppimage10_B20_C20_D600 = PytorchBetaVAERepresentation(n_latents=10, initial_epochs=0, beta=1, n_channels=3,
#                                                            dim_ergo=1, batch_size=256)
# SupPoppimage10_B20_C20_D600.load_model(model_path)
# SupPoppimage10_B20_C20_D600.sorted_latents = np.array([3, 1, 2, 0, 4, 5, 7, 6, 8, 9])


if __name__ == '__main__':
    images, ball_pos, apex_setup, ergo_pos = torch.load('full_dataset.pt')

    # We perform RGE-UGL
    rep = PytorchBetaVAERepresentation(beta=20, n_latents=10, n_channels=3,
                                       initial_epochs=20, Ta=200, capacity=25, capacity_change_duration=12,
                                       batch_size=32, learning_rate=1e-4,
                                       visdom_record=True, visdom_env="test", log_interval=1)
    rep.reset(images[:1000], ball_pos[:1000], apex_setup[:1000], ergo_pos[:1000])
    print('PytorchBetaVAERepresentation working')

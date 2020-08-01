import torch
from torch.utils.data import DataLoader, Dataset

from torch import nn, optim
from torch.nn import functional as F
from torchvision import transforms
from torchvision.utils import save_image, make_grid
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import os

from datetime import datetime


class Custom_Dataset(Dataset):

    def __init__(self, device, train=True):
        self.device = device
        try:
            all_data = np.load('data.npz')['X']
        except ImportError:
            print("You should first create the database file: data.npz")

        index = int(all_data.shape[0] * 0.85)

        np.random.seed(66)
        np.random.shuffle(all_data)
        if train:
            self.data = all_data[:index]
        else:
            self.data = all_data[index:]

    def __len__(self):

        """ Denotes the total number of samples """
        return self.data.shape[0]

    def __getitem__(self, index):
        """ Generates one sample of data """

        # Load data and get label
        img = self.data[index]
        transform = transforms.Compose([transforms.ToTensor()])

        return transform(img)


class Flatten(nn.Module):
    @staticmethod
    def forward(input_tensor):
        return input_tensor.view(input_tensor.size(0), -1)


class UnFlatten(nn.Module):
    @staticmethod
    def forward(input_tensor, size=128 * 7 * 7):
        return input_tensor.view(input_tensor.size(0), size, 1, 1)


class VAE(nn.Module):
    def __init__(self, image_channels=3, h_dim=6272, z_dim=256):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(image_channels, 32, kernel_size=6, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=6, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=6, stride=2),
            nn.ReLU(),
            Flatten()
        )

        self.fc1 = nn.Linear(h_dim, z_dim)
        self.fc2 = nn.Linear(h_dim, z_dim)
        self.fc3 = nn.Linear(z_dim, h_dim).to("cuda")

        self.decoder = nn.Sequential(
            UnFlatten(),
            nn.ConvTranspose2d(h_dim, 128, kernel_size=7, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=6, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=6, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(32, image_channels, kernel_size=6, stride=2),
            nn.Sigmoid(),
        )

    @staticmethod
    def reparameterize(mu, logvar):
        std = logvar.mul(0.5).exp_()
        esp = torch.randn_like(mu)
        z = (mu + std * esp).to("cuda")
        return z

    def bottleneck(self, h):
        mu, logvar = self.fc1(h), self.fc2(h)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def encode(self, x):
        h = self.encoder(x)
        z, mu, logvar = self.bottleneck(h)
        return z, mu, logvar

    def decode(self, z):
        z = self.fc3(z)
        z = self.decoder(z)
        return z

    def forward(self, x):
        z, mu, logvar = self.encode(x)
        z = self.decode(z)
        return z, mu, logvar


class ModelTraining:

    def __init__(self, model, epochs, train_loader, test_loader, device, batch_size, load_model=None, log_interval=100):
        if load_model:
            self.folder = load_model
            folder_path = 'saved_models/' + self.folder
            filename = os.listdir(folder_path)[-1]
            print(filename)
            checkpoint = torch.load(folder_path + filename)
            vae.load_state_dict(checkpoint['model_state_dict'])
            self.first_epoch = checkpoint['epoch']
        else:
            dt = datetime.today()
            self.folder = str(dt.hour) + "-" + str(dt.day) + \
                "-" + str(dt.month) + '/'
            self.make_folder()
            self.first_epoch = 1

        self.device = device
        self.batch_size = batch_size
        self.model = model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)

        self.final_epoch = epochs
        self.train_loader = train_loader
        self.test_loader = test_loader

        self.batch_size = batch_size
        self.log_interval = log_interval

        self.writer = SummaryWriter()
        images = next(iter(self.train_loader)).to(self.device)

        grid = make_grid(images)
        self.writer.add_image('images', grid, 0)
        self.writer.add_graph(self.model, images)

    def make_folder(self):
        if os.path.isdir('results/' + self.folder):
            print('Folders already created')
        else:
            os.mkdir('results/' + self.folder)
            os.mkdir('saved_models/' + self.folder)

    @staticmethod
    def loss_fn(recon_x, x, mu, logvar):
        BCE = F.binary_cross_entropy(recon_x, x, size_average=False)
        KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return BCE + KLD, BCE, KLD

    def train(self, epoch):
        self.model.train()
        total_train_loss = 0
        total_bce_loss = 0
        total_kld_loss = 0

        for batch_idx, data in enumerate(self.train_loader):
            data = data.to(self.device)
            self.optimizer.zero_grad()
            recon_batch, mu, logvar = self.model(data)
            total_loss, bce_loss, kld_loss = self.loss_fn(
                recon_batch, data, mu, logvar)
            total_loss.backward()
            total_train_loss += total_loss.item()
            total_bce_loss += bce_loss.item()
            total_kld_loss += kld_loss.item()

            self.optimizer.step()
            if batch_idx % self.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx *
                    len(data), len(self.train_loader.dataset),
                    100. * batch_idx / len(self.train_loader),
                    total_loss.item() / len(data)))

        total_train_loss /= len(self.train_loader.dataset)
        total_bce_loss /= len(self.train_loader.dataset)
        total_kld_loss /= len(self.train_loader.dataset)

        print('====> Epoch: {} Average loss: {:.4f}'.format(
                epoch, total_train_loss))

        self.writer.add_scalar('Train/Average Loss ', total_train_loss, epoch)
        self.writer.add_scalar('Train/BCE Loss ', total_bce_loss, epoch)
        self.writer.add_scalar('Train/KLD Loss ', total_kld_loss, epoch)

    def test(self, epoch):
        self.model.eval()
        total_test_loss = 0
        total_bce_loss = 0
        total_kld_loss = 0

        with torch.no_grad():
            for i, data in enumerate(self.test_loader):
                data = data.to(self.device)
                recon_batch, mu, logvar = self.model(data)
                total_loss, bce_loss, kld_loss = self.loss_fn(
                    recon_batch, data, mu, logvar)
                if i == 0:
                    n = min(data.size(0), 8)
                    comparison = torch.cat(
                        [data[:n], recon_batch.view(self.batch_size, 3, 84, 84)[:n]])
                    save_image(
                        comparison.cpu(),
                        'results/' +
                        self.folder +
                        'reconstruction_' +
                        str(epoch) +
                        '.png',
                        nrow=n)

                total_test_loss += total_loss.item()
                total_bce_loss += bce_loss.item()
                total_kld_loss += kld_loss.item()

        total_test_loss /= len(self.test_loader.dataset)
        total_bce_loss /= len(self.test_loader.dataset)
        total_kld_loss /= len(self.test_loader.dataset)

        print('====> Test set loss: {:.4f}'.format(total_test_loss))

        self.writer.add_scalar('Test/Average Loss', total_test_loss, epoch)
        self.writer.add_scalar('Test/BCE Loss ', total_bce_loss, epoch)
        self.writer.add_scalar('Test/KLD Loss ', total_kld_loss, epoch)

    def process(self, save_interval=50):
        for epoch in range(self.first_epoch + 1, self.final_epoch + 1):
            self.train(epoch)
            self.test(epoch)
            with torch.no_grad():
                sample = torch.randn(16, 256).to(self.device)
                sample = self.model.decode(sample).cpu()
                save_image(sample.view(16, 3, 84, 84), 'results/' + self.folder + 'sample_' + str(epoch) + '.png')

            if epoch != 0 and epoch % save_interval == 0:
                model_path = "saved_models/" + self.folder + \
                    "saved_vae_epoch_" + str(epoch) + ".pt"
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict()
                }, model_path)


if __name__ == '__main__':

    # PARAMETERS #
    bs = 32
    nb_epochs = 200
    device_type = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # MAIN #
    training_loader = DataLoader(Custom_Dataset(device_type), batch_size=bs, shuffle=True)
    testing_loader = DataLoader(Custom_Dataset(device_type, train=False), batch_size=bs, shuffle=True)
    vae = VAE().to(device_type)
    model_date = "13-31-7/"
    mt = ModelTraining(vae, nb_epochs, training_loader, testing_loader, device_type, bs, load_model=model_date)
    mt.process(save_interval=50)

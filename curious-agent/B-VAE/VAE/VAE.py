import numpy as np
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.utils import save_image, make_grid
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torch.utils.data import DataLoader, sampler


class VAE(nn.Module):

    def __init__(self, device, batch_size, loader_train, loader_test, latent_dim,
                 low_size, middle_size, high_size, learning_rate, path, writer):

        super(VAE, self).__init__()

        self.device = device
        self.loader_train = loader_train
        self.loader_test = loader_test
        self.batch_size = batch_size

        self.latent_dim = latent_dim
        self.low_size = low_size
        self.middle_size = middle_size
        self.high_size = high_size
        #######################################################################
        #                       ** Fully Connected **
        #######################################################################
        self.fc1 = nn.Linear(self.high_size, self.middle_size)
        self.fc2 = nn.Linear(self.middle_size, self.low_size)
        self.fc31 = nn.Linear(self.low_size, self.latent_dim)
        self.fc32 = nn.Linear(self.low_size, self.latent_dim)
        self.fc4 = nn.Linear(self.latent_dim, self.low_size)
        self.fc5 = nn.Linear(self.low_size, self.middle_size)
        self.fc6 = nn.Linear(self.middle_size, self.high_size)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        self.to(device)
        self.path = path

        self.images, _ = next(iter(self.loader_train))
        self.grid = make_grid(self.images)
        self.writer = writer

        self.writer.add_image('images', self.grid, 0)
        self.writer.add_graph(self, self.images)

    def encode(self, x):
        #######################################################################
        #                       ** ReLU Activation **
        #######################################################################
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))
        return self.fc31(h2), self.fc32(h2)

    @staticmethod
    def reparametrize(mu, logvar):
        #######################################################################
        #                       ** Reparametrization trick **
        #######################################################################
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        #######################################################################
        #                       ** ReLU Activation **
        #######################################################################
        h4 = F.relu(self.fc4(z))
        h5 = F.relu(self.fc5(h4))
        return torch.sigmoid(self.fc6(h5))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, self.high_size))
        z = self.reparametrize(mu, logvar)
        return self.decode(z), mu, logvar

    @staticmethod
    def loss_function(recon_x, x, mu, logvar):

        # Kullback_Leibler Divergence
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        # Binary Cross Entropy is used as an alternative to Negative Log Likelihood
        NLL = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')

        return [KLD, NLL]

    def summary(self):
        params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print("Total number of parameters is: {}".format(params))
        print(self)

    def test(self, epoch_index):

        self.eval()
        KLD_loss, NLL_loss = [0, 0]

        with torch.no_grad():
            for i, (data, _) in enumerate(self.loader_test):
                data = data.to(self.device)
                recon_batch, mu, logvar = self(data)
                loss_2d = self.loss_function(recon_batch, data, mu, logvar)
                KLD_loss += loss_2d[0].item()
                NLL_loss += loss_2d[1].item()
                if i == 0:
                    n = min(data.size(0), 8)
                    comparison = torch.cat([data[:n], recon_batch.view(self.batch_size, 1, 28, 28)[:n]])
                    if not os.path.exists(self.path + '/reconstructions/'):
                        os.makedirs(self.path + '/reconstructions/')
                    save_image(comparison.cpu(), self.path + '/reconstructions/reconstructions_epoch_'
                               + str(epoch_index) + '.png', nrow=n)

        KLD_loss /= len(self.loader_test.dataset)
        NLL_loss /= len(self.loader_test.dataset)
        return KLD_loss, NLL_loss

    def perform_training(self, num_epochs):
        self.train()
        total_loss_per_epoch_training = []
        total_loss_per_epoch_testing = []

        for epoch in range(num_epochs):

            KLD_loss, NLL_loss = [0, 0]

            for batch_idx, (data, _) in enumerate(self.loader_train):
                data = data.to(self.device)
                self.optimizer.zero_grad()
                recon_batch, mu, logvar = self(data)
                loss_2d = self.loss_function(recon_batch, data, mu, logvar)
                loss = sum(loss_2d)
                loss.backward()
                KLD_loss += loss_2d[0].item()
                NLL_loss += loss_2d[1].item()
                self.optimizer.step()
            kld_total_loss, nll_total_loss = [KLD_loss / len(self.loader_train.dataset),
                                              NLL_loss / len(self.loader_train.dataset)]

            self.writer.add_scalar('kld loss', kld_total_loss, epoch)
            self.writer.add_scalar('nll loss', nll_total_loss, epoch)

            total_loss_per_epoch_training.append([kld_total_loss, nll_total_loss])

            total_loss_per_epoch_testing.append(self.test(epoch))

            print(f"====> Epoch: {epoch} Average loss: {(KLD_loss + NLL_loss) / len(self.loader_train.dataset)}")

        return np.array(total_loss_per_epoch_training), np.array(total_loss_per_epoch_testing)

    def save(self, filename):
        torch.save(self.state_dict(), filename)


def get_loaders(train_dat, test_dat, batch_size):
    # Modify this line if you need to do any input transformations (optional).
    transform = transforms.Compose([transforms.ToTensor()])

    loader_train = DataLoader(train_dat, batch_size, shuffle=True)
    loader_test = DataLoader(test_dat, batch_size, shuffle=False)

    return loader_train, loader_test


def process_raw_data_from_npz(data_path):
    npz_files = [filename for filename in os.listdir(data_path) if filename.endswith('.npz')]
    list_img_matrix = [np.load(data_path + npz_file, allow_pickle=True)['obs'][:-1] for npz_file in npz_files]
    X_data = np.concatenate(list_img_matrix)
    return X_data.reshape(X_data.shape[0], 84, 84, 3)


def VAE_training(path, bs_size, n_epochs, lr_rate, latent_dimension, data_folder='collected_observations/'):

    #######################################################################
    #                       ** DEVICE SELECTION **
    #######################################################################

    GPU = False
    device_idx = 0
    if GPU:
        DEVICE = torch.device("cuda:" + str(device_idx) if torch.cuda.is_available() else "cpu")
    else:
        DEVICE = torch.device("cpu")

    print('Type of device use : ', DEVICE)

    # Random seed for reproducible results
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
    torch.manual_seed(0)

    #######################################################################
    #                       ** DATA LOADING **
    #######################################################################

    if not os.path.exists(path):
        os.makedirs(path)

    loader_train, loader_test = preprocess_data(data_folder, batch_size)
    sample_inputs, _ = next(iter(loader_test))
    fixed_input = sample_inputs[:2, :, :, :]

    save_img(fixed_input, path + '/image_original.png')

    #######################################################################
    #                       ** MODEL LOADING **
    #######################################################################
    writer = SummaryWriter()
    model = VAE(DEVICE, bs_size, loader_train, loader_test, latent_dimension,
                low_size, middle_size, high_size, lr_rate, path, writer)

    writer.close()
    model.summary()

    #######################################################################
    #                       ** MODEL TRAINING **
    #######################################################################

    training_loss, testing_loss = model.perform_training(n_epochs)
    plot(training_loss, testing_loss, path + '/training_losses.png')
    model.save(path + '/VAE_model.pth')


if __name__ == "__main__":

    #######################################################################
    #                       ** PARAMETERS **
    #######################################################################
    custom_path = 'results'

    #######################################################################
    #                       ** HYPERPARAMETERS **
    #######################################################################
    num_epochs = 1
    learning_rate = 0.001
    batch_size = 64
    latent_dim = 10

    low_size = 10 * 10
    middle_size = 28 * 28
    high_size = 56 * 56

    VAE_training(custom_path, batch_size, num_epochs, learning_rate, latent_dim)

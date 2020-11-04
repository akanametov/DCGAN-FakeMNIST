import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import make_grid

from tqdm.notebook import tqdm
from IPython.display import clear_output

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

###########################
#### HELPER FUNCTIONS #####
###########################

def show_images(pred, real, log, num_images=25):
    pred_unf = ((pred+1)/2).detach().cpu()
    real_unf = ((real+1)/2).detach().cpu()
    pred_grid = make_grid(pred_unf[:num_images], nrow=5)
    real_grid = make_grid(real_unf[:num_images], nrow=5)
    fig = plt.figure()
    ax1, ax2 = fig.subplots(1, 2)
    plt.title(log)
    ax1.imshow(pred_grid.permute(1, 2, 0).squeeze())
    ax2.imshow(real_grid.permute(1, 2, 0).squeeze())
    plt.show()
    
def initialize_weights(layer, mean=0.0, std=0.02):
    if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.ConvTranspose2d):
        torch.nn.init.normal_(layer.weight, mean, std)
    if isinstance(layer, nn.BatchNorm2d):
        torch.nn.init.normal_(layer.weight, mean, std)
        torch.nn.init.constant_(layer.bias, 0)
        
##################################
######## ConvBnReLU block ########
##################################

class ConvBnReLU(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size=3, stride=2):
        super().__init__()
        self.block=nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())
        
    def forward(self, x):
        return self.block(x)
    
##################################
##### ConvBnLeakyReLU block ######
##################################

class ConvBnLeakyReLU(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size=4, stride=2, alpha=0.2):
        super().__init__()
        self.block=nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(alpha))
        
    def forward(self, x):
        return self.block(x)
    
##################################
########### GENERATOR ############
##################################

class Generator(nn.Module):
    def __init__(self, in_channels, hid_channels, out_channels,
                 kernel_size=3, stride=2):
        super().__init__()
        self.model = nn.Sequential(
            ConvBnReLU(in_channels, 4*hid_channels, kernel_size, stride),
            ConvBnReLU(4*hid_channels, 2*hid_channels, kernel_size=4, stride=1),
            ConvBnReLU(2*hid_channels, hid_channels, kernel_size, stride),
            nn.ConvTranspose2d(hid_channels, out_channels, kernel_size=4, stride=stride),
            nn.Tanh())
        
    def forward(self, x):
        return self.model(x)
    
##################################
######### DISCRIMINATOR ##########
##################################

class Discriminator(nn.Module):
    def __init__(self, in_channels, hid_channels,
                 kernel_size=4, stride=2):
        super().__init__()
        self.model = nn.Sequential(
            ConvBnLeakyReLU(in_channels, hid_channels, kernel_size, stride),
            ConvBnLeakyReLU(hid_channels, 2*hid_channels, kernel_size, stride),
            nn.Conv2d(2*hid_channels, 1, kernel_size, stride))
        
    def forward(self, x):
        out = self.model(x)
        return out.view(out.size(0), -1)
    
##################################
############ TRAINER #############
##################################

class Trainer():
    def __init__(self, Generator, Discriminator, G_optimizer, D_optimizer,
                 criterion, device=device):
        self.G = Generator.to(device)
        self.D = Discriminator.to(device)
        self.G_optim = G_optimizer
        self.D_optim = D_optimizer
        
        self.criterion=criterion.to(device)
        self.results={'G_loss':[], 'D_loss':[]}
        
    def fit(self, generator, epochs=30, device=device):
        for epoch in range(1, epochs+1):
            G_losses=[]
            D_losses=[]

            log = f'::::: Epoch {epoch}/{epochs} :::::'
            for real, _ in tqdm(generator):
                real = real.to(device)
                # DISCRIMINATOR
                self.D_optim.zero_grad()
                # DISCRIMINATOR`s LOSS
                noise = torch.randn(real.size(0), 64, 1, 1).to(device)
                # Prediction on FAKE image
                fake = self.G(noise).detach()
                fake_pred = self.D(fake)
                fake_loss = self.criterion(fake_pred, torch.zeros_like(fake_pred))
                # Prediction on REAL image
                real_pred = self.D(real)
                real_loss = self.criterion(real_pred, torch.ones_like(real_pred))
                D_loss = (fake_loss + real_loss)/2

                D_losses.append(D_loss.item())
                D_loss.backward(retain_graph=True)
                self.D_optim.step()
                # GENERATOR
                self.G_optim.zero_grad()
                # GENERATOR`s LOSS
                fake = self.G(noise)
                fake_pred = self.D(fake)
                G_loss = self.criterion(fake_pred, torch.ones_like(fake_pred))

                G_losses.append(G_loss.item())
                G_loss.backward()
                self.G_optim.step()
                template = f'::: Generator Loss: {G_loss.item():.3f} | Discriminator Loss: {D_loss.item():.3f} :::'

            self.results['G_loss'].append(np.mean(G_losses))
            self.results['D_loss'].append(np.mean(D_losses))
            clear_output(wait=True)
            show_images(fake, real, log+template)
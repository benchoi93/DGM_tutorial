from matplotlib import pyplot as plt
import torch.optim as optim
import torch.nn as nn
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split

from vae import VariationalAutoencoder, plot_latent, plot_reconstructed
from pathlib import Path
Path('./vae').mkdir(exist_ok=True)

batch_size = 64

transform = transforms.Compose([transforms.ToTensor()])
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = VariationalAutoencoder(latent_dims=2).to(device)

optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 30
for epoch in range(num_epochs):
    for i, (images, _) in enumerate(train_loader):
        x_hat = model(images.to(device))
        loss = ((x_hat - images.to(device))**2).sum() + model.encoder.kl
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')

    plot_latent(model, train_loader, device)
    plt.savefig(f'./vae/latent_{epoch}.png')
    plt.clf()

    plot_reconstructed(model, device)
    plt.savefig(f'./vae/reconstructed_{epoch}.png')
    plt.clf()

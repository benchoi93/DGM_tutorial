from matplotlib import pyplot as plt
import torch.optim as optim
import torch.nn as nn
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split

from gan import Generator, Discriminator
from pathlib import Path
Path('./gan').mkdir(exist_ok=True)

batch_size = 64

transform = transforms.Compose([transforms.ToTensor()])
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
G = Generator(100, 784).to(device)
D = Discriminator(784).to(device)
criterion = nn.BCELoss()
lr = 0.0002
G_optimizer = optim.Adam(G.parameters(), lr=lr)
D_optimizer = optim.Adam(D.parameters(), lr=lr)

num_epochs = 200
for epoch in range(num_epochs):
    for i, (images, _) in enumerate(train_loader):

        x_fake = G(torch.randn(batch_size, 100).to(device))
        x_real = images.reshape(batch_size, 784).to(device)

        D_real_loss = criterion(D(x_real), torch.ones(batch_size, 1).to(device))
        D_fake_loss = criterion(D(x_fake), torch.zeros(batch_size, 1).to(device))

        D_loss = D_real_loss + D_fake_loss

        D_optimizer.zero_grad()
        D_loss.backward()
        D_optimizer.step()

        x_fake = G(torch.randn(batch_size, 100).to(device))
        G_loss = criterion(D(x_fake), torch.ones(batch_size, 1).to(device))

        G_optimizer.zero_grad()
        G_loss.backward()
        G_optimizer.step()

        if (i+1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], D_loss: {D_loss.item():.4f}, G_loss: {G_loss.item():.4f}')

    with torch.no_grad():
        x_fake = G(torch.randn(64, 100).to(device))
        x_fake = x_fake.reshape(-1, 1, 28, 28).to('cpu').detach()
        torchvision.utils.save_image(x_fake, f'./gan/sample_{epoch}.png', nrow=8, padding=0)

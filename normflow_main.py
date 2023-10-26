import normflow as nf
import torch.distributions as distributions
from matplotlib import pyplot as plt
import torch.optim as optim
import torch.nn as nn
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split

from normflow import RealNVP
from pathlib import Path
Path('./nf').mkdir(exist_ok=True)

batch_size = 64

transform = transforms.Compose([transforms.ToTensor()])
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = RealNVP().to(device)

optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 30

for epoch in range(num_epochs):

    with torch.no_grad():
        # sample
        sample = model.sample(64)
        torchvision.utils.save_image(sample, f'./nf/sample_{epoch}.png', nrow=8, padding=0)

    for i, (images, _) in enumerate(train_loader):
        images = images.to(device)

        # Forward pass
        z, ldj = model.encode(images)
        log_pz = model.prior.log_prob(z).sum(dim=[1, 2, 3])
        log_px = ldj + log_pz
        loss = -log_px.mean()

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')

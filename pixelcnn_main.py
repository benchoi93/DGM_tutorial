from matplotlib import pyplot as plt
import torch.optim as optim
import torch.nn as nn
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split

from pixelcnn import PixelCNN
from pathlib import Path
Path('./pixelcnn').mkdir(exist_ok=True)

batch_size = 64

transform = transforms.Compose([transforms.ToTensor()])
dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_dataset, val_dataset = random_split(dataset, [55000, 5000])
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = PixelCNN().to(device)

optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

num_epochs = 30
for epoch in range(num_epochs):
    for i, (images, _) in enumerate(train_loader):
        target = (images[:, 0, :, :] * 255).long()
        images = images.to(device)
        target = target.to(device)

        outputs = model(images)
        loss = criterion(outputs, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')

    with torch.no_grad():
        # sample
        sample = torch.zeros(64, 1, 28, 28).to(device)
        for i in range(28):
            for j in range(28):
                out = model(sample)
                probs = torch.softmax(out[:, :, i, j], dim=1)
                sample[:, :, i, j] = torch.multinomial(probs, 1).float() / 255.

        torchvision.utils.save_image(sample, f'./pixelcnn/sample_{epoch}.png', nrow=8, padding=0)

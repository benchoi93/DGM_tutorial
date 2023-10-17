import torch.nn as nn


class MaskedCNN(nn.Conv2d):
    """
    Implementation of Masked CNN Class as explained in A Oord et. al. 
    Taken and modified from https://github.com/jzbontar/pixelcnn-pytorch
    """

    def __init__(self, mask_type, *args, **kwargs):
        super(MaskedCNN, self).__init__(*args, **kwargs)
        self.register_buffer('mask', self.weight.data.clone())

        _, depth, height, width = self.weight.size()
        self.mask.fill_(1)
        if mask_type == 'A':
            self.mask[:, :, height // 2, width // 2:] = 0
            self.mask[:, :, height // 2 + 1:, :] = 0
        else:
            self.mask[:, :, height // 2, width // 2 + 1:] = 0
            self.mask[:, :, height // 2 + 1:, :] = 0

    def forward(self, x):
        self.weight.data *= self.mask
        return super(MaskedCNN, self).forward(x)


class PixelCNN(nn.Module):
    """
    Network of PixelCNN as described in A Oord et. al. 
    """

    def __init__(self, no_layers=8, kernel=7, channels=64):
        super(PixelCNN, self).__init__()

        self.layers = nn.ModuleList()

        # Initial layer with mask type 'A'
        self.layers.append(MaskedCNN('A', 1, channels, kernel, 1, kernel // 2, bias=False))
        self.layers.append(nn.BatchNorm2d(channels))
        self.layers.append(nn.ReLU(True))

        # Subsequent layers with mask type 'B'
        for _ in range(no_layers - 1):
            self.layers.append(MaskedCNN('B', channels, channels, kernel, 1, kernel // 2, bias=False))
            self.layers.append(nn.BatchNorm2d(channels))
            self.layers.append(nn.ReLU(True))

        # Final layer to produce outputs
        self.out = nn.Conv2d(channels, 256, 1)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.out(x)

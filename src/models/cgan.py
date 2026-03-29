"""
Conditional DCGAN (cGAN) for local synthetic data generation.

Each client trains a local cGAN on its private data partition.
The generator supports class-conditioned synthesis via one-hot label embedding.
"""

import torch
import torch.nn as nn


class Generator(nn.Module):
    """DCGAN-style conditional generator.

    Maps latent vector z and class label y to a synthetic image.

    Args:
        latent_dim: Dimension of the noise vector z.
        num_classes: Number of classes (for one-hot conditioning).
        feature_maps: Base number of feature maps.
        num_channels: Output image channels (1=MNIST, 3=CIFAR-10).
        image_size: Output image spatial size (28 or 32).
    """

    def __init__(
        self,
        latent_dim: int = 100,
        num_classes: int = 10,
        feature_maps: int = 64,
        num_channels: int = 1,
        image_size: int = 28,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.image_size = image_size

        # Input: z (latent_dim) + y (num_classes) concatenated
        input_dim = latent_dim + num_classes

        if image_size == 28:  # MNIST
            self.main = nn.Sequential(
                # input_dim -> feature_maps*4 x 7 x 7
                nn.ConvTranspose2d(input_dim, feature_maps * 4, 7, 1, 0, bias=False),
                nn.BatchNorm2d(feature_maps * 4),
                nn.ReLU(True),
                # feature_maps*4 -> feature_maps*2 x 14 x 14
                nn.ConvTranspose2d(
                    feature_maps * 4, feature_maps * 2, 4, 2, 1, bias=False
                ),
                nn.BatchNorm2d(feature_maps * 2),
                nn.ReLU(True),
                # feature_maps*2 -> num_channels x 28 x 28
                nn.ConvTranspose2d(
                    feature_maps * 2, num_channels, 4, 2, 1, bias=False
                ),
                nn.Tanh(),
            )
        else:  # CIFAR-10 (32x32)
            self.main = nn.Sequential(
                # input_dim -> feature_maps*8 x 4 x 4
                nn.ConvTranspose2d(input_dim, feature_maps * 8, 4, 1, 0, bias=False),
                nn.BatchNorm2d(feature_maps * 8),
                nn.ReLU(True),
                # feature_maps*8 -> feature_maps*4 x 8 x 8
                nn.ConvTranspose2d(
                    feature_maps * 8, feature_maps * 4, 4, 2, 1, bias=False
                ),
                nn.BatchNorm2d(feature_maps * 4),
                nn.ReLU(True),
                # feature_maps*4 -> feature_maps*2 x 16 x 16
                nn.ConvTranspose2d(
                    feature_maps * 4, feature_maps * 2, 4, 2, 1, bias=False
                ),
                nn.BatchNorm2d(feature_maps * 2),
                nn.ReLU(True),
                # feature_maps*2 -> num_channels x 32 x 32
                nn.ConvTranspose2d(
                    feature_maps * 2, num_channels, 4, 2, 1, bias=False
                ),
                nn.Tanh(),
            )

    def forward(self, z: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Generate synthetic images conditioned on class labels.

        Args:
            z: Noise vector of shape (batch_size, latent_dim).
            labels: Class indices of shape (batch_size,).

        Returns:
            Synthetic images of shape (batch_size, C, H, W).
        """
        # One-hot encode labels
        one_hot = torch.zeros(labels.size(0), self.num_classes, device=z.device)
        one_hot.scatter_(1, labels.unsqueeze(1), 1.0)

        # Concatenate z and one-hot label, reshape for ConvTranspose2d
        x = torch.cat([z, one_hot], dim=1).unsqueeze(-1).unsqueeze(-1)
        return self.main(x)


class Discriminator(nn.Module):
    """DCGAN-style conditional discriminator.

    Takes an image and class label, outputs real/fake probability.

    Args:
        num_classes: Number of classes.
        feature_maps: Base number of feature maps.
        num_channels: Input image channels (1=MNIST, 3=CIFAR-10).
        image_size: Input image spatial size (28 or 32).
    """

    def __init__(
        self,
        num_classes: int = 10,
        feature_maps: int = 64,
        num_channels: int = 1,
        image_size: int = 28,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.image_size = image_size

        # Label embedding: project label to a single-channel image
        self.label_embed = nn.Embedding(num_classes, image_size * image_size)

        # Input channels: image channels + 1 (label channel)
        in_channels = num_channels + 1

        if image_size == 28:  # MNIST
            self.main = nn.Sequential(
                nn.Conv2d(in_channels, feature_maps, 4, 2, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(feature_maps, feature_maps * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(feature_maps * 2),
                nn.LeakyReLU(0.2, inplace=True),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(feature_maps * 2, 1),
                nn.Sigmoid(),
            )
        else:  # CIFAR-10
            self.main = nn.Sequential(
                nn.Conv2d(in_channels, feature_maps, 4, 2, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(feature_maps, feature_maps * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(feature_maps * 2),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(feature_maps * 2, feature_maps * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(feature_maps * 4),
                nn.LeakyReLU(0.2, inplace=True),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(feature_maps * 4, 1),
                nn.Sigmoid(),
            )

    def forward(self, images: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Classify images as real or fake, conditioned on labels.

        Args:
            images: Image tensor of shape (batch_size, C, H, W).
            labels: Class indices of shape (batch_size,).

        Returns:
            Real/fake probability of shape (batch_size, 1).
        """
        # Create label channel
        label_channel = self.label_embed(labels).view(
            -1, 1, self.image_size, self.image_size
        )
        # Concatenate image and label channel
        x = torch.cat([images, label_channel], dim=1)
        return self.main(x)

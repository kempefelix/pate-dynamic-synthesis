"""
Local cGAN Training and Inference.

Each client trains a conditional GAN on its private data partition.
The generator can then produce class-conditioned synthetic samples
on request from the synthesis coordinator.
"""

import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from typing import Dict, List, Optional, Tuple

from src.models.cgan import Generator, Discriminator

logger = logging.getLogger(__name__)


class LocalCGAN:
    """Manages a local cGAN for one client.

    Handles training the generator on the client's private data
    and generating synthetic samples on demand.
    """

    def __init__(
        self,
        client_id: int,
        available_classes: List[int],
        num_classes: int = 10,
        latent_dim: int = 100,
        num_channels: int = 1,
        image_size: int = 28,
        feature_maps: int = 64,
        device: torch.device = torch.device("cpu"),
    ):
        self.client_id = client_id
        self.available_classes = available_classes
        self.num_classes = num_classes
        self.latent_dim = latent_dim
        self.device = device
        self.image_size = image_size

        self.generator = Generator(
            latent_dim=latent_dim,
            num_classes=num_classes,
            feature_maps=feature_maps,
            num_channels=num_channels,
            image_size=image_size,
        ).to(device)

        self.discriminator = Discriminator(
            num_classes=num_classes,
            feature_maps=feature_maps,
            num_channels=num_channels,
            image_size=image_size,
        ).to(device)

        self._is_trained = False

    def train(
        self,
        dataset: Subset,
        num_epochs: int = 200,
        batch_size: int = 64,
        lr_g: float = 0.0002,
        lr_d: float = 0.0002,
        beta1: float = 0.5,
    ) -> Dict[str, List[float]]:
        """Train the local cGAN on the client's private data.

        Args:
            dataset: Client's private data subset.
            num_epochs: Number of training epochs.
            batch_size: Batch size.
            lr_g: Generator learning rate.
            lr_d: Discriminator learning rate.
            beta1: Adam beta1 parameter.

        Returns:
            Dictionary with training loss history.
        """
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                                drop_last=True, num_workers=0)

        criterion = nn.BCELoss()
        opt_g = optim.Adam(self.generator.parameters(), lr=lr_g, betas=(beta1, 0.999))
        opt_d = optim.Adam(self.discriminator.parameters(), lr=lr_d, betas=(beta1, 0.999))

        history = {"g_loss": [], "d_loss": []}

        self.generator.train()
        self.discriminator.train()

        for epoch in range(num_epochs):
            epoch_g_loss = 0.0
            epoch_d_loss = 0.0
            num_batches = 0

            for real_images, real_labels in dataloader:
                batch_sz = real_images.size(0)
                real_images = real_images.to(self.device)
                real_labels = real_labels.to(self.device)

                real_target = torch.ones(batch_sz, 1, device=self.device)
                fake_target = torch.zeros(batch_sz, 1, device=self.device)

                # --- Train Discriminator ---
                opt_d.zero_grad()

                # Real samples
                d_real = self.discriminator(real_images, real_labels)
                loss_d_real = criterion(d_real, real_target)

                # Fake samples
                z = torch.randn(batch_sz, self.latent_dim, device=self.device)
                fake_labels = real_labels  # Use same labels for paired training
                fake_images = self.generator(z, fake_labels)
                d_fake = self.discriminator(fake_images.detach(), fake_labels)
                loss_d_fake = criterion(d_fake, fake_target)

                loss_d = loss_d_real + loss_d_fake
                loss_d.backward()
                opt_d.step()

                # --- Train Generator ---
                opt_g.zero_grad()

                z = torch.randn(batch_sz, self.latent_dim, device=self.device)
                fake_images = self.generator(z, fake_labels)
                d_fake = self.discriminator(fake_images, fake_labels)
                loss_g = criterion(d_fake, real_target)

                loss_g.backward()
                opt_g.step()

                epoch_g_loss += loss_g.item()
                epoch_d_loss += loss_d.item()
                num_batches += 1

            if num_batches > 0:
                history["g_loss"].append(epoch_g_loss / num_batches)
                history["d_loss"].append(epoch_d_loss / num_batches)

            if (epoch + 1) % 50 == 0:
                logger.info(
                    f"Client {self.client_id} | Epoch {epoch+1}/{num_epochs} | "
                    f"G_loss: {history['g_loss'][-1]:.4f} | "
                    f"D_loss: {history['d_loss'][-1]:.4f}"
                )

        self._is_trained = True
        logger.info(f"Client {self.client_id}: cGAN training complete.")
        return history

    @torch.no_grad()
    def generate(self, class_label: int, num_samples: int) -> torch.Tensor:
        """Generate synthetic samples for a specific class.

        Args:
            class_label: Target class to generate.
            num_samples: Number of samples to generate.

        Returns:
            Tensor of shape (num_samples, C, H, W).

        Raises:
            ValueError: If class_label is not in this client's available classes.
        """
        if class_label not in self.available_classes:
            raise ValueError(
                f"Client {self.client_id} cannot generate class {class_label}. "
                f"Available: {self.available_classes}"
            )

        self.generator.eval()
        z = torch.randn(num_samples, self.latent_dim, device=self.device)
        labels = torch.full((num_samples,), class_label, dtype=torch.long,
                            device=self.device)
        return self.generator(z, labels).cpu()

    @property
    def is_trained(self) -> bool:
        return self._is_trained

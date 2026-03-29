"""
Teacher and Student CNN architectures.

The base architecture (TeacherCNN) is adapted from the NEWSROOM/Saferlearn
framework (Thales Research & Technology, UCStubModel). Extended to support
both MNIST (1 channel) and CIFAR-10 (3 channels).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TeacherCNN(nn.Module):
    """CNN used for both Teacher and Student models.

    Architecture adapted from Saferlearn's UCStubModel:
    2x Conv2d -> MaxPool -> 2x Dropout -> 2x Linear -> LogSoftmax

    Supports MNIST (1x28x28) and CIFAR-10 (3x32x32).
    """

    def __init__(self, num_channels: int = 1, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(num_channels, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)

        # Compute flattened size: depends on input dimensions
        # MNIST: 28->26->24->12 => 64*12*12 = 9216
        # CIFAR: 32->30->28->14 => 64*14*14 = 12544
        if num_channels == 1:  # MNIST
            self._flat_size = 9216
        else:  # CIFAR-10
            self._flat_size = 12544

        self.fc1 = nn.Linear(self._flat_size, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Return class probabilities (softmax) instead of log-softmax."""
        self.eval()
        with torch.no_grad():
            logits = self.fc2(
                self.dropout2(
                    F.relu(
                        self.fc1(
                            torch.flatten(
                                self.dropout1(
                                    F.max_pool2d(
                                        F.relu(self.conv2(F.relu(self.conv1(x)))), 2
                                    )
                                ),
                                1,
                            )
                        )
                    )
                )
            )
        return F.softmax(logits, dim=1)

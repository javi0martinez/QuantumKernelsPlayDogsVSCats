"""
CNN model for feature extraction from cat and dog images.
"""

import torch.nn as nn


class CnnFeatureExtractor(nn.Module):
    def __init__(self):
        super(CnnFeatureExtractor, self).__init__()

        # Three convolutional layers with batch normalization and max pooling
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=0, stride=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=0, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=0, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        # Dense layers for feature reduction
        self.fc1 = nn.Linear(3 * 3 * 64, 250)
        self.fc2 = nn.Linear(250, 10)
        self.fc3 = nn.Linear(10, 2)
        self.relu = nn.ReLU()

    def forward(self, x):
        """Forward pass through the network."""
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.view(out.size(0), -1)
        out = self.relu(self.fc1(out))
        out = self.relu(self.fc2(out))
        features = out  # 10-dimensional features for quantum kernel
        out = self.fc3(out)
        return out, features

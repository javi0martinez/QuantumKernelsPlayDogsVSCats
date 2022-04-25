"""
Data loading and preprocessing utilities.
"""

import os
import glob
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split


class DogsCatsDataset(Dataset):
    """Dataset class for Dogs vs Cats dataset."""

    def __init__(self, file_list, transform=None):
        self.file_list = file_list
        self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_path = self.file_list[idx]
        img = Image.open(img_path)
        img_transformed = self.transform(img) if self.transform else img

        label = img_path.split("/")[-1].split(".")[0]
        if label == "dog":
            label = 1
        elif label == "cat":
            label = 0

        return img_transformed, label


def get_data_transforms(img_size=224):
    """Get data augmentation and normalization transformations."""

    train_transforms = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.RandomResizedCrop(img_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    )

    val_transforms = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.RandomResizedCrop(img_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    )

    test_transforms = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.RandomResizedCrop(img_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    )

    return train_transforms, val_transforms, test_transforms


def prepare_data_loaders(data_dir, batch_size=100, img_size=224):
    """Prepare data loaders for training and validation."""

    # Get file lists
    train_dir = os.path.join(data_dir, "train")
    test_dir = os.path.join(data_dir, "test")

    train_list = glob.glob(os.path.join(train_dir, "*.jpg"))
    test_list = glob.glob(os.path.join(test_dir, "*.jpg"))

    # Split train data into train and validation
    train_list, val_list = train_test_split(train_list, test_size=0.3)

    # Get transformations
    train_transforms, val_transforms, test_transforms = get_data_transforms(img_size)

    # Create datasets
    train_data = DogsCatsDataset(train_list, transform=train_transforms)
    val_data = DogsCatsDataset(val_list, transform=val_transforms)
    test_data = DogsCatsDataset(test_list, transform=test_transforms)

    # Create data loaders
    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True)

    return train_loader, val_loader, test_loader

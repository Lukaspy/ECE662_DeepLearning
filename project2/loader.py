#Modified loader to allow for per epoch augmentation
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from jax import numpy as jnp


class JAXBatchLoader:

    def __init__(self, torch_loader, one_hot: bool = False, num_classes: int = 10, flatten: bool = False):
        self.torch_loader = torch_loader
        self.one_hot = one_hot
        self.num_classes = num_classes
        self.flatten = flatten
        self._iterator = None

    def __iter__(self):
        # New iterator each epoch 
        self._iterator = iter(self.torch_loader)
        return self

    def __next__(self):
        images, labels = next(self._iterator)  # images: (B, C, H, W) for CIFAR-10
        images = images.numpy()
        labels = labels.numpy()

        if self.flatten:
            images = images.reshape(images.shape[0], -1)

        if self.one_hot:
            labels = jnp.eye(self.num_classes, dtype=jnp.float32)[labels]

        return jnp.array(images, dtype=jnp.float32), jnp.array(labels)


def make_cifar_datasets():
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    )

    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    train_data = datasets.CIFAR10(
        root="data",
        train=True,
        download=True,
        transform=train_transform,
    )
    test_data = datasets.CIFAR10(
        root="data",
        train=False,
        download=True,
        transform=test_transform,
    )
    return train_data, test_data


def load_data(batch_size: int = 128, shuffle: bool = True, flatten: bool = False):
    """Return (train_loader, test_loader) without one-hot labels."""
    train_data, test_data = make_cifar_datasets()

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=shuffle)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    return (
        JAXBatchLoader(train_loader, one_hot=False, flatten=flatten),
        JAXBatchLoader(test_loader, one_hot=False, flatten=flatten),
    )


def load_data_onehot(batch_size=(128, 256), shuffle: bool = True, num_classes: int = 10, flatten: bool = False):
    train_data, test_data = make_cifar_datasets()

    train_loader = DataLoader(train_data, batch_size=batch_size[0], shuffle=shuffle)
    test_loader = DataLoader(test_data, batch_size=batch_size[1], shuffle=False)

    return (
        JAXBatchLoader(train_loader, one_hot=True, num_classes=num_classes, flatten=flatten),
        JAXBatchLoader(test_loader, one_hot=True, num_classes=num_classes, flatten=flatten),
    )

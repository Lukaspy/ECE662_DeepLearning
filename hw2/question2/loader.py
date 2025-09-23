#Provided in mnist_jax code zip
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from jax import numpy as jnp



class JAXBatchLoader:
    def __init__(self, torch_loader, one_hot=False, num_classes=10, flatten=False):
        self.loader = iter(torch_loader)
        self.one_hot = one_hot
        self.num_classes = num_classes
        self.flatten = flatten

    def __iter__(self):
        self.loader = iter(self.loader)
        return self

    def __next__(self):
        images, labels = next(self.loader)
        images = images.numpy()  # shape: (batch, 1, 28, 28)
        labels = labels.numpy()
        images = images.squeeze(1)  # shape: (batch, 28, 28), normalize
        # images = images.squeeze(1) / 255.0  # shape: (batch, 28, 28), normalize
        if self.flatten:
            images = images.reshape(images.shape[0], -1)  # (batch, 784)
        if self.one_hot:
            labels = jnp.eye(self.num_classes)[labels]
        return jnp.array(images), jnp.array(labels)



def load_data(batch_size=64, shuffle=True, flatten=False):
    train_data = datasets.MNIST(
        root="data",
        train=True,
        download=True,
        transform=ToTensor(),
    )
    test_data = datasets.MNIST(
        root="data",
        train=False,
        download=True,
        transform=ToTensor(),
    )
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=shuffle)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=shuffle)
    return JAXBatchLoader(train_loader, flatten=flatten), JAXBatchLoader(test_loader, flatten=flatten)


def load_data_onehot(batch_size=[40_000, 10_000], shuffle=True, num_classes=10, flatten=False):
    train_data = datasets.MNIST(
        root="data",
        train=True,
        download=True,
        transform=ToTensor(),
    )
    test_data = datasets.MNIST(
        root="data",
        train=False,
        download=True,
        transform=ToTensor(),
    )
    train_loader = DataLoader(train_data, batch_size=batch_size[0], shuffle=shuffle)
    test_loader = DataLoader(test_data, batch_size=batch_size[1], shuffle=shuffle)
    return JAXBatchLoader(train_loader, one_hot=True, num_classes=num_classes, flatten=flatten), JAXBatchLoader(test_loader, one_hot=True, num_classes=num_classes, flatten=flatten)
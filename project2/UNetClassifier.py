#combines the in class UNet with a final pooling into MLP head for simple classification
import jax
from jax import numpy as jnp
from jax import random as jr
import equinox as eqx
from typing import Callable
from UNet import UNet

class UNetClassifier(eqx.Module):
    unet: UNet
    head: eqx.nn.Linear

    def __init__(
        self,
        num_classes: int,
        *,
        key,
        in_channels: int = 3,          
        num_spatial_dims: int = 2,
        hidden_channels: int = 32,
        num_levels: int = 3,
        activation: Callable = jax.nn.relu,
        feature_channels: int = 64,
    ):
        key_unet, key_head = jr.split(key)

        self.unet = UNet(
            num_spatial_dims=num_spatial_dims,
            in_channels=in_channels,
            out_channels=feature_channels,
            hidden_channels=hidden_channels,
            num_levels=num_levels,
            activation=activation,
            key=key_unet,
        )

        self.head = eqx.nn.Linear(
            in_features=feature_channels,
            out_features=num_classes,
            key=key_head,
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        #x: (B, C, H, W) batch of images

        # Linear head produces logits after pooling the output of the UNet
        logits = self.head(self.unet(x).mean(axis=(1, 2))) 
        return logits

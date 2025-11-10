import project2.UNetClassifier as UNetClassifier
import jax
from jax import numpy as jnp
from jax import random as jr



CIFAR10_model = UNetClassifier(
    num_spatial_dims=2,
    in_channels=1,
    out_channels=10,
    hidden_channels=32,
    num_levels=2,
    activation=jax.nn.relu,
    key=jr.key(0),
)


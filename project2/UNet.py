#In class implementation
import jax
from jax import numpy as jnp
from jax import random as jr
import equinox as eqx
from typing import Callable
import optax

class DoubleConv(eqx.Module):
    conv_1: eqx.nn.Conv
    conv_2: eqx.nn.Conv
    activation: Callable

    def __init__(
        self,
        num_spatial_dims: int, 
        in_channels: int,
        out_channels: int,
        activation: Callable,
        *,
        key
    ):
        c_1_key, c_2_key = jr.split(key)
        self.conv_1 = eqx.nn.Conv(
            num_spatial_dims=num_spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
            key=c_1_key,
        )
        self.conv_2 = eqx.nn.Conv(
            num_spatial_dims=num_spatial_dims,
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
            key=c_2_key,
        )        
        self.activation = activation
    
    def __call__(self, x: jax.Array):
        x = self.conv_1(x)
        x = self.activation(x)
        x = self.conv_2(x)
        return self.activation(x)


class UNet(eqx.Module):
    lifting: DoubleConv
    down_sampling_blocks: list[eqx.nn.Conv]
    left_arc_blocks: list[DoubleConv]
    right_arc_blocks: list[DoubleConv]
    up_sampling_blocks: list[eqx.nn.Conv]
    projection: eqx.nn.Conv

    def __init__(
        self,
        num_spatial_dims: int,
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        num_levels: int,
        activation: Callable,
        *,
        key,
    ):
        key, lifting_key, projection_key = jr.split(key, 3)
        self.lifting = DoubleConv(
            num_spatial_dims=num_spatial_dims,
            in_channels=in_channels,
            out_channels=hidden_channels,
            activation=activation,
            key=lifting_key,
        )
        self.projection = eqx.nn.Conv(
            num_spatial_dims=num_spatial_dims,
            in_channels=hidden_channels,
            out_channels=out_channels,
            kernel_size=1,
            key=projection_key,
        )

        channel_list = [hidden_channels * 2 ** i for i in range(0, num_levels+1)]

        self.down_sampling_blocks = []
        self.left_arc_blocks = []
        self.right_arc_blocks = []
        self.up_sampling_blocks = []

        for (upper_level_channels, lower_level_channels) in zip(channel_list[:-1], channel_list[1:]):
            key, down_key, left_key, up_key, right_key = jr.split(key, 5)
            self.down_sampling_blocks.append(
                eqx.nn.Conv(
                    num_spatial_dims=num_spatial_dims,
                    in_channels=upper_level_channels,
                    out_channels=upper_level_channels,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    key=down_key,
                )
            )
            self.left_arc_blocks.append(
                DoubleConv(
                    num_spatial_dims=num_spatial_dims,
                    in_channels=upper_level_channels,
                    out_channels=lower_level_channels,
                    activation=activation,
                    key=left_key,
                )
            )
            self.up_sampling_blocks.append(
                eqx.nn.ConvTranspose(
                    num_spatial_dims=num_spatial_dims,
                    in_channels=lower_level_channels,
                    out_channels=upper_level_channels,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1,
                    key=up_key,
                )
            )
            self.right_arc_blocks.append(
                DoubleConv(
                    num_spatial_dims=num_spatial_dims,
                    in_channels=lower_level_channels,
                    out_channels=upper_level_channels,
                    activation=activation,
                    key=right_key,
                )
            )

    def __call__(self, x: jax.Array):
        x = self.lifting(x)
        x_skip = []

        # Left part (contracting path)
        for down, left in zip(self.down_sampling_blocks, self.left_arc_blocks):
            x_skip.append(x)
            x = down(x)
            x = left(x)

        # Right part (expanding path)
        for up, right in zip(reversed(self.up_sampling_blocks), reversed(self.right_arc_blocks)):
            x = up(x)
            x = jnp.concatenate([x, x_skip.pop()], axis=0)
            x = right(x)

        x = self.projection(x)
        return x
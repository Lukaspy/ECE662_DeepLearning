"""
Lukas Crockett ECE 662 HW2
adapted from equinox docs
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import optax


"""
Three layer NN 
"""
class JaxNN(eqx.Module):
    layers: tuple

    def __init__(self, key):
        key1, key2 = jax.random.split(key, 2)
        self.layers = (eqx.nn.Linear(784, 30, key=key1),
                       eqx.nn.Linear(30, 10, key=key2))
    
    def __call__(self, x):
        for layer in self.layers[:-1]:
            x = jax.nn.sigmoid(layer(x))
        return self.layers[-1](x)

    def evaluate(self, x, y):
        pred = self(x)
        return jnp.argmax(pred).astype(int) == jnp.argmax(y).astype(int)







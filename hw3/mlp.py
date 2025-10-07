import equinox as eqx
import jax

class MLP(eqx.Module):
    layers: tuple

    def __init__(self, key):
        key1, key2, key3 = jax.random.split(key,3)
        self.layers = (eqx.nn.Linear(2, 10),
                       eqx.nn.Linear(10, 3))
        
    def __call__(self, x):
        for layer in self.layers[:-1]:
            x = jax.nn.softmax(layer(x))
        return self.layers[-1](x)
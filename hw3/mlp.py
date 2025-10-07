import equinox as eqx
import jax

class MLP(eqx.Module):
    layers: tuple

    def __init__(self, key):
        key1, key2 = jax.random.split(key,2)
        self.layers = (eqx.nn.Linear(2, 100, key=key1),
                       jax.nn.sigmoid,
                       eqx.nn.Linear(100, 3, key=key2),
                       jax.nn.softmax)
        
    def __call__(self, x):
        for layer in self.layers[:-1]:
            x = layer(x)
        return self.layers[-1](x)
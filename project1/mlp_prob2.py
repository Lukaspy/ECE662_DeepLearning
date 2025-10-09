import equinox as eqx
import jax

class MLP(eqx.Module):
    layers: tuple

    def __init__(self, key):
        key1, key2, key3 = jax.random.split(key,3)
        self.layers = (eqx.nn.Linear(3, 200, key=key1),
                       jax.nn.tanh,
                       eqx.nn.Linear(200, 100, key=key2),
                       jax.nn.relu,
                       eqx.nn.Linear(100, 2, key=key3))
        
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
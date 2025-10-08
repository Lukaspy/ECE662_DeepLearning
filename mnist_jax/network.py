from functools import partial
import jax
from jax import numpy as jnp

def mse_loss(params, x, y, model):
    output = model(params, x)
    return 0.5 * jnp.sum((output - y) ** 2)

def cross_entropy_loss(params, x, y, model):
    output = model(params, x)
    log_preds = output - jax.nn.logsumexp(output)
    # return -jnp.sum(y * jnp.log(output + 1e-12))
    return -jnp.sum(y * log_preds) # or jnp.mean(y * log_preds)
    # return -jnp.mean(y * log_preds)

class Network:
    def __init__(self, layer_sizes, loss_fn=cross_entropy_loss, activation=jax.nn.sigmoid):
        self.loss = partial(loss_fn, model=self)
        self.layer_sizes = layer_sizes
        self.num_layers = len(layer_sizes) - 1
        self.activation = activation
        self.activation_grad = jax.grad(lambda z: jnp.sum(self.activation(z)))

    def __call__(self, params, x):
        for i in range(self.num_layers):
            x = jnp.dot(params['w'][i], x) + params['b'][i]
            x = self.activation(x)
        return x

    def forward(self, params, x):
        activations = [x]
        pre_activations = []
        for i in range(self.num_layers):
            z = jnp.dot(params['w'][i], activations[-1]) + params['b'][i]
            pre_activations.append(z)
            a = self.activation(z)
            activations.append(a)
        return activations, pre_activations

    def backward(self, params, x, y):
        activations, pre_activations = self.forward(params, x)
        loss = 0.5 * jnp.sum((activations[-1] - y) ** 2)
        delta = (activations[-1] - y) * self.activation_grad(pre_activations[-1])
        grad_w = []
        grad_b = []
        for l in reversed(range(self.num_layers)):
            a_prev = activations[l]
            w = params['w'][l]
            grad_w.insert(0, jnp.outer(delta, a_prev))
            grad_b.insert(0, delta)
            if l > 0:
                delta = jnp.dot(w.T, delta) * self.activation_grad(pre_activations[l-1])
        grads = {'w': grad_w, 'b': grad_b}
        return grads, loss

    def update(self, params, grads, lr):
        new_w = [w - lr * gw for w, gw in zip(params['w'], grads['w'])]
        new_b = [b - lr * gb for b, gb in zip(params['b'], grads['b'])]
        return {'w': new_w, 'b': new_b}
    
    @partial(jax.jit, static_argnames=['self', 'backward_fn', 'update_fn'])
    def batch_update(self, params, X_batch, Y_batch, lr, backward_fn, update_fn):
        # g, _ = jax.vmap(backward_fn, in_axes=(None, 0, 0))(params, X_batch, Y_batch)
        # mean_g = jax.tree.map(lambda x: jnp.mean(x, axis=0), g)
        mean_g = jax.grad(lambda p: self.batch_loss(p, X_batch, Y_batch))(params)
        return update_fn(params, mean_g, lr)
    
    def evaluate(self, params, x, y):
        pred = self(params, x)
        return jnp.argmax(pred).astype(int) == jnp.argmax(y).astype(int)

    def batch_evaluate(self, params, X_batch, Y_batch):
        correct = jax.vmap(self.evaluate, in_axes=(None, 0, 0))(params, X_batch, Y_batch)
        return jnp.sum(correct) / X_batch.shape[0] * 100

    def batch_loss(self, params, X_batch, Y_batch):
        losses = jax.vmap(self.loss, in_axes=(None, 0, 0))(params, X_batch, Y_batch)
        return jnp.mean(losses)

    def sgd(self, params, X, Y, X_test, Y_test, epochs=3, batch_size=32, lr=1.0):
        for epoch in range(epochs):
            for i in range(0, int(X.shape[0]), batch_size):
                X_batch = X[i:i+batch_size]
                Y_batch = Y[i:i+batch_size]
                params = self.batch_update(params, X_batch, Y_batch, lr, self.backward, self.update)
            train_acc = self.batch_evaluate(params, X, Y)
            test_acc = self.batch_evaluate(params, X_test, Y_test)
            loss = self.batch_loss(params, X, Y)
            print(f"Epoch {epoch+1}, Loss: {loss:.4f}, Train Acc: {train_acc:.2f}%, Test Acc: {test_acc:.2f}%")
        return params

def init_network_params(layer_sizes, key):
    w_list = []
    b_list = []
    keys = jax.random.split(key, len(layer_sizes) - 1)
    for i in range(len(layer_sizes) - 1):
        w_key, b_key = jax.random.split(keys[i])
        weight_shape = (layer_sizes[i + 1], layer_sizes[i])
        bias_shape = (layer_sizes[i + 1],)
        w = jax.random.normal(w_key, shape=weight_shape) * jnp.sqrt(1.0 / layer_sizes[i])
        b = jax.random.normal(b_key, shape=bias_shape) * jnp.sqrt(1.0 / layer_sizes[i])
        w_list.append(w)
        b_list.append(b)
    return {'w': w_list, 'b': b_list}


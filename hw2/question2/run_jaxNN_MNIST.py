"""
Lukas Crockett ECE 662 
Adapted from provided mnist_jax code
"""

import jax,optax
import jax.numpy as jnp
from loader import load_data, load_data_onehot
from jaxNN import JaxNN
from train_jaxNN import train


#Load data and one hot encoded labels
train_iter, test_iter = load_data_onehot(flatten=True)
X_train, Y_train = next(train_iter)
X_test, Y_test = next(test_iter)

#initialize model
key = jax.random.PRNGKey(0)
model = JaxNN(key)

#train model
optim = optax.sgd(0.3)
model = train(model, optim, X_train, Y_train, X_test, Y_test, batch_size=128, epochs=30)




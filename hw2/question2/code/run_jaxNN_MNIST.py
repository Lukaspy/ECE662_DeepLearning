"""
Lukas Crockett ECE 662 
Adapted from provided mnist_jax code
and Nielsen plots
"""

import jax,optax
import jax.numpy as jnp
from loader import load_data, load_data_onehot
from jaxNN import JaxNN
from train_jaxNN import train
import matplotlib.pyplot as plt


#Load data and one hot encoded labels
train_iter, test_iter = load_data_onehot(flatten=True)
X_train, Y_train = next(train_iter)
X_test, Y_test = next(test_iter)

#initialize model
key = jax.random.PRNGKey(0)
model = JaxNN(key)

#train model
optim = optax.sgd(0.3)
model = train(model, optim, X_train, Y_train, X_test, Y_test, batch_size=10, epochs=3)


#extract weights, W[0]=w2, W[1]=w3
W = [0] * 3
for i, layer in enumerate(model.layers):
    W[i] = layer.weight
    print(f"W{i+2}: {W[i].shape}")


#plot weights
rows, cols = 5, 6
fig, axs = plt.subplots(rows, cols)
plt.subplots_adjust(hspace = 0.3)
for j in range(30):
    h1_image = W[0][j].reshape(28,28)
    print()
    #print("The weights into h{0} have image ".format(j))
    ax = axs.flat[j]
    im = ax.imshow(h1_image,cmap='BuPu', origin='upper')
    ax.set_title("h{0} image".format(j))
#    plt.xticks(())
#    plt.yticks(())
#    print("plt.show():")
    print()
    if j==1: fig.colorbar(im,ax=axs, fraction=0.03, pad=0.02)
fig.suptitle("W2 weights Jax/Equinox")
plt.show()


print()
print("Plot the W3 weights.")
print()
rows, cols = 5, 2
fig, axs = plt.subplots(rows, cols)
plt.subplots_adjust(hspace = 0.8)
for j in range(10):
    #print("The weights into output {0} are".format(j))
    ax = axs.flat[j]
    ax.plot(W[1][j],'o')
    ax.set_title("Output {0} weights".format(j))
    print()

fig.suptitle("W3 weights Jax/Equinox")
plt.show()





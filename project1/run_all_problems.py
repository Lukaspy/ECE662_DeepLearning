# %%
import jax
import jax.numpy as jnp
import numpy as np
import optax
import equinox as eqx
from mlp_prob1 import MLP as MLP1
from mlp_prob2 import MLP as MLP2


#φ[n] = [φ_1 φ_2 φ_3z φ_3x] //one row
#s[n] = (θ, ̇θ_dot, h)  
with open("data.npy", 'rb') as f:
    s = jnp.load(f)
    phi = jnp.load(f) # phi for prob 1
    phi_2 = jnp.load(f) #phi for prob 2
    phi_3 = jnp.load(f) #phi for prob 3

def shuffle_and_split(x, y, train_pcnt = 0.8, seed = 42):
    
    #shuffle the dataset and split into test/train
    n = len(x)
    indexes = np.arange(n)
    np.random.seed(seed)
    np.random.shuffle(indexes)
    
    split = int (train_pcnt * n)
    
    train_idx = indexes[0:split]
    test_idx = indexes[split:]
    
    x_train, x_test = x[train_idx], x[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    return x_train, y_train, x_test, y_test


#MSE Loss fxn
def compute_loss(model, x, y):
    pred_y = jax.vmap(model)(x)  
    return jnp.mean((pred_y - y) ** 2)

#RMSE accuracy metric
def compute_accuracy(model, x, y):

    def eval_single(model, x, y):
        pred_y = model(x)
        residual_sq = jnp.mean((y - pred_y)**2)
        return residual_sq
    res_all = jax.vmap(eval_single, in_axes=(None, 0, 0))(model, x, y)
    return jnp.sqrt(jnp.sum(res_all) / x.shape[0])
    

#Single training step (JIT compiled)
@eqx.filter_jit
def train_step(model, x, y, opt_state, optimizer):
    def loss_fn(model):
        return compute_loss(model, x, y)
    loss, grads = eqx.filter_value_and_grad(loss_fn)(model)
    updates, opt_state = optimizer.update(grads, opt_state, eqx.filter(model, eqx.is_array))
    model = eqx.apply_updates(model, updates)
    return model, opt_state


def train(x_train, y_train, x_test, y_test, model, optimizer, num_epochs=500, batch_size = 64):
    num_train = len(x_train)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    for epoch in range(num_epochs):

        for batch_start in range(0, num_train, batch_size):
            batch_end = batch_start + batch_size
            x_batch = x_train[batch_start:batch_end,:] #train on theta, theta_dot, and h
            y_batch = y_train[batch_start:batch_end,:] #model ouput should be f_3x and f_3z only
            model, opt_state = train_step(model, x_batch, y_batch, opt_state, optimizer)

        if (epoch + 1) % 50 == 0 or epoch == 0:
            loss = compute_loss(model, x_batch, y_batch)
            acc = compute_accuracy(model, x_batch, y_batch)
            test_loss = compute_loss(model, x_test, y_test)
            test_acc = compute_accuracy(model, x_test, y_test)
            print(f"Epoch {epoch+1}: Loss={loss:.4f}, RMSE={acc:.4f}  Test set: Loss={test_loss:.4f}, RMSE={test_acc:.4f}")


# initialize random key and optimizer
key = jax.random.PRNGKey(0)
optimizer = optax.adamw(learning_rate=1e-3, weight_decay=1e-4) 

"""Problem 1"""
print("\n\n-----Now training network for problem 1-----")
model = MLP1(key)
s_train, phi_train, s_test, phi_test = shuffle_and_split(s, phi)
x_train = s_train[:,[0,2]]
y_train = phi_train[:,[3,2]]
x_test = s_test[:,[0,2]]
y_test = phi_test[:,[3,2]]
train(x_train, y_train, x_test, y_test, model, optimizer)

"""Problem 2"""
print("\n\n-----Now training network for problem 2-----")
model = MLP2(key)
s_train, phi_train, s_test, phi_test = shuffle_and_split(s, phi_2)
x_train = s_train[:,:3]
y_train = phi_train[:,[3,2]]
x_test = s_test[:,:3]
y_test = phi_test[:,[3,2]]
train(x_train, y_train, x_test, y_test, model, optimizer)

"""Problem 3"""
print("\n\n-----Now training network for problem 3-----")
model = MLP2(key)
s_train, phi_train, s_test, phi_test = shuffle_and_split(s, phi_3)
x_train = s_train[:,:3]
y_train = phi_train[:,[3,2]]
x_test = s_test[:,:3]
y_test = phi_test[:,[3,2]]
train(x_train, y_train, x_test, y_test, model, optimizer)

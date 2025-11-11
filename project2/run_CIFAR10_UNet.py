from UNetClassifier import UNetClassifier 
import jax
from jax import random as jr
from jax import numpy as jnp
import optax
import equinox as eqx
from loader import load_data, load_data_onehot


def cross_entropy_loss(model, x, y):
    pred_y = jax.vmap(model)(x)
    return optax.softmax_cross_entropy(logits=pred_y, labels=y).mean()


def compute_accuracy(model, x, y):
    def eval_single(model, x, y):
        pred_y = model(x)
        return jnp.argmax(pred_y).astype(int) == jnp.argmax(y).astype(int)

    correct = jax.vmap(eval_single, in_axes=(None,0,0))(model, x, y)
    return jnp.sum(correct) / x.shape[0] * 100


def train(model, optimizer, X, Y, X_test, Y_test, batch_size=32, epochs=10):
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    @eqx.filter_jit
    def train_step(model, opt_state, optimizer, x, y):
        def loss_fn(model):
            return cross_entropy_loss(model, x, y)
        loss_value, grads = eqx.filter_value_and_grad(loss_fn)(model)
        updates, opt_state = optimizer.update(grads, opt_state, eqx.filter(model, eqx.is_array))
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss_value
    
    def batch_evaluate(model, X, Y, eval_batch_size=256):
        def eval_single(model, x, y):
            pred = model(x)
            return jnp.argmax(pred).astype(int) == jnp.argmax(y).astype(int)
        
        num_correct = 0
        total = 0
        for i in range(0, X.shape[0], eval_batch_size):
            X_b = X[i:i+eval_batch_size]
            Y_b = Y[i:i+eval_batch_size]
            correct = jax.vmap(eval_single, in_axes=(None, 0, 0))(model, X_b, Y_b)
            num_correct += correct.sum()
            total += correct.shape[0]

        return (num_correct / total) * 100


    for epoch in range(epochs):
        for i in range(0, int(X.shape[0]), batch_size):
            X_batch = X[i:i+batch_size]
            Y_batch = Y[i:i+batch_size]
            model, opt_state, train_loss = train_step(model, opt_state, optimizer, X_batch, Y_batch)
        train_acc = batch_evaluate(model, X, Y)
        test_acc = batch_evaluate(model, X_test, Y_test)
        print(f"Epoch {epoch+1}, Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Test Acc: {test_acc:.2f}%")
    return model




if __name__ == "__main__":
    key = jax.random.PRNGKey(0)
    model = UNetClassifier(num_classes=10, key=key)
    optimizer = optax.adamw(learning_rate=1e-4, weight_decay=5e-4)

    train_iter, test_iter = load_data_onehot(flatten=False)
    X_train, Y_train = next(train_iter)
    X_test, Y_test = next(test_iter)

    print("X_train shape:", X_train.shape)
    print("X_test shape:", X_test.shape)
    print("Y_train shape:", Y_train.shape)
    print("Y_test shape:", Y_test.shape)

    #print("Train == Test images? ", jnp.all(X_train == X_test))
    #print("Train == Test labels? ", jnp.all(Y_train == Y_test))


    train(model, optimizer, X_train, Y_train, X_test, Y_test, batch_size=32, epochs=100)




            







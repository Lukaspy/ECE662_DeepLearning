"""
Lukas Crockett ECE 662 HW2
Adapted from equinox docs
"""
import jaxNN
import jax, jax.numpy as jnp, equinox as eqx, optax

def train(model: jaxNN, optim:  optax.GradientTransformation, X, Y, X_test, Y_test, batch_size=32, epochs=3):
    opt_state = optim.init(eqx.filter(model, eqx.is_array))

    @eqx.filter_jit
    def train_step(model: jaxNN, opt_state, x, y):
        loss_value, grads = eqx.filter_value_and_grad(mse_loss)(model, x, y)
        updates, opt_state = optim.update(grads, opt_state, eqx.filter(model, eqx.is_array))
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss_value
    
    def evaluate(model: jaxNN, x, y):
        pred = model(x)
        return jnp.argmax(pred).astype(int) == jnp.argmax(y).astype(int)
    
    def batch_evaluate(model: jaxNN, X_batch, Y_batch):
        correct = jax.vmap(evaluate, in_axes=(None, 0, 0))(model, X_batch, Y_batch)
        return jnp.sum(correct) / X_batch.shape[0] * 100

    for epoch in range(epochs):
        for i in range(0, int(X.shape[0]), batch_size):
            X_batch = X[i:i+batch_size]
            Y_batch = Y[i:i+batch_size]
            model, opt_state, train_loss = train_step(model, opt_state, X_batch, Y_batch)
        train_acc = batch_evaluate(model, X, Y)
        test_acc = batch_evaluate(model, X_test, Y_test)
        print(f"Epoch {epoch+1}, Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Test Acc: {test_acc:.2f}%")
    return model
                

def mse_loss(model: jaxNN, x, y):
    pred_y = jax.vmap(model)(x)
    return jnp.mean((y - pred_y) **2)

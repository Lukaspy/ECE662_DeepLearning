from UNetClassifier import UNetClassifier

import jax
from jax import numpy as jnp
import optax
import equinox as eqx

from loader import load_data_onehot


def cross_entropy_loss(model, x, y):
    logits = jax.vmap(model)(x)  # (B, num_classes)
    return optax.softmax_cross_entropy(logits=logits, labels=y).mean()


def train(model, optimizer, train_loader, test_loader, epochs: int = 20):
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    @eqx.filter_jit
    def train_step(model, opt_state, x, y):
        def loss_fn(m, xb, yb):
            return cross_entropy_loss(m, xb, yb)

        loss_value, grads = eqx.filter_value_and_grad(loss_fn)(model, x, y)
        updates, opt_state = optimizer.update(grads, opt_state, eqx.filter(model, eqx.is_array))
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss_value

    def eval_loader(model, data_loader):
        """Compute accuracy (%) over a JAXBatchLoader."""
        def eval_single(m, x, y):
            pred = m(x)  
            return jnp.argmax(pred).astype(int) == jnp.argmax(y).astype(int)

        total_correct = 0
        total = 0
        for X_batch, Y_batch in data_loader:
            correct = jax.vmap(eval_single, in_axes=(None, 0, 0))(model, X_batch, Y_batch)
            total_correct += correct.sum()
            total += correct.shape[0]
        return float(total_correct) / float(total) * 100.0

    best_test = 0.0

    for epoch in range(epochs):
        # Training loop over (augmented) CIFAR-10 batches
        epoch_loss = 0.0
        n_seen = 0

        for X_batch, Y_batch in train_loader:
            model, opt_state, batch_loss = train_step(model, opt_state, X_batch, Y_batch)
            bsz = X_batch.shape[0]
            epoch_loss += float(batch_loss) * bsz
            n_seen += bsz

        avg_loss = epoch_loss / n_seen

        train_acc = eval_loader(model, train_loader)
        test_acc = eval_loader(model, test_loader)

        if test_acc > best_test:
            best_test = test_acc

        print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}, Train Acc: {train_acc:.2f}%, Test Acc: {test_acc:.2f}%")

    print(f"Best test accuracy: {best_test:.2f}%")
    return model


if __name__ == "__main__":
    key = jax.random.PRNGKey(0)

    model = UNetClassifier(num_classes=10, key=key)

    optimizer = optax.adamw(
        learning_rate=1e-3,
        weight_decay=5e-4,
    )

    train_loader, test_loader = load_data_onehot(
        batch_size=(128, 256),
        shuffle=True,
        num_classes=10,
        flatten=False,
    )

    model = train(model, optimizer, train_loader, test_loader, epochs=100)
    eqx.tree_serialise_leaves("unet_cifar10_aug.eqx", model)
    print("Saved trained model to unet_cifar10_aug.eqx")

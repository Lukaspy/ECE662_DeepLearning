import numpy as np
import jax
from jax import numpy as jnp
import equinox as eqx
import matplotlib.pyplot as plt

from UNetClassifier import UNetClassifier
from loader_nonaug import load_data_onehot   


def load_trained_model(path: str = "unet_cifar10_aug.eqx", num_classes: int = 10):
    key = jax.random.PRNGKey(0)

    model_init = UNetClassifier(
        num_classes=num_classes,
        key=key,
        in_channels=3,
        num_spatial_dims=2,
        hidden_channels=32,
        num_levels=3,
        feature_channels=64,
    )

    model = eqx.tree_deserialise_leaves(path, model_init)
    return model


def get_test_data(num_classes: int = 10):
    _, test_loader = load_data_onehot(
        batch_size=[50_000, 10_000],
        shuffle=False,
        num_classes=num_classes,
        flatten=False,
    )
    X_test, Y_test = next(test_loader)   
    return X_test, Y_test


def compute_confusion_matrix(model, X_test, Y_test, num_classes: int = 10):
    logits = jax.vmap(model)(X_test)          # (N, num_classes)
    preds = jnp.argmax(logits, axis=-1)       # (N,)
    true = jnp.argmax(Y_test, axis=-1)        # (N,)

    preds_np = np.array(preds)
    true_np = np.array(true)

    cm = np.zeros((num_classes, num_classes), dtype=int)
    # cm[true, pred] += 1 for each sample
    np.add.at(cm, (true_np, preds_np), 1)
    return cm, true_np, preds_np


def pretty_print_confusion_matrix(cm, class_names):
    print("Confusion matrix (rows = true class, cols = predicted class):\n")
    header = "      " + " ".join(f"{name[:3]:>4}" for name in class_names)
    print(header)
    for i, name in enumerate(class_names):
        row = " ".join(f"{cm[i, j]:4d}" for j in range(cm.shape[1]))
        print(f"{name[:3]:>3}  {row}")


def plot_confusion_matrix_heatmap(cm, class_names, save_path: str | None = "confusion_matrix.png"):
    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    data = cm.astype(float) / row_sums
    fmt = ".2f"
    title = "CIFAR-10 Confusion Matrix (Normalized)"

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(data, interpolation="nearest", cmap="Blues")
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("Proportion", rotation=-90, va="bottom")

    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="True label",
        xlabel="Predicted label",
        title=title,
    )

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    thresh = data.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(data[i, j], fmt),
                ha="center",
                va="center",
                color="white" if data[i, j] > thresh else "black",
                fontsize=8,
            )

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=300)
        print(f"Saved confusion matrix heatmap to {save_path}")

    plt.show()


def main():
    num_classes = 10
    class_names = [
        "airplane", "automobile", "bird", "cat", "deer",
        "dog", "frog", "horse", "ship", "truck",
    ]

    print("Loading trained model...")
    model = load_trained_model("unet_cifar10_aug.eqx", num_classes=num_classes)

    print("Loading CIFAR-10 test data...")
    X_test, Y_test = get_test_data(num_classes=num_classes)

    print("Computing confusion matrix...")
    cm, true_np, preds_np = compute_confusion_matrix(model, X_test, Y_test, num_classes=num_classes)

    acc = (true_np == preds_np).mean() * 100.0
    print(f"\nTest accuracy (sanity check): {acc:.2f}%\n")

    # Text version
    pretty_print_confusion_matrix(cm, class_names)

    # Heatmap version
    plot_confusion_matrix_heatmap(cm, class_names, normalize=True, save_path="confusion_matrix_norm.png")


if __name__ == "__main__":
    main()

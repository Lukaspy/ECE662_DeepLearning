from jax import numpy as jnp
import numpy as np

def generate_spiral_data(points_per_class=100, num_classes=3, noise=0.2, random_seed=42):
    np.random.seed(random_seed)
    X = np.zeros((points_per_class * num_classes, 2))
    y = np.zeros(points_per_class * num_classes, dtype=int)
    angle_span = 4*np.pi/3  # 240 degrees in radians
    for class_number in range(num_classes):
        ix = range(points_per_class * class_number, points_per_class * (class_number + 1))
        r = np.linspace(0.0, 1, points_per_class)  # radius
        t = np.linspace(class_number * angle_span, (class_number + 1) * angle_span, points_per_class) + np.random.randn(points_per_class) * noise  # theta
        X[ix] = np.c_[r * np.cos(t), r * np.sin(t)]
        y[ix] = class_number
    return jnp.array(X), jnp.array(y)

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    X, y = generate_spiral_data(points_per_class=100, num_classes=3, noise=0.2)
    plt.figure(figsize=(6, 6))
    for class_number in range(3):
        plt.scatter(X[y == class_number, 0], X[y == class_number, 1], label=f"Class {class_number}")
    plt.legend()
    plt.title("Spiral Data (3 Classes)")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.axis("equal")
    plt.show()

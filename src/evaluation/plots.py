import matplotlib.pyplot as plt


def plot_confusion_matrix(
    matrix,
    labels=("0", "1"),
    title="Confusion Matrix",
):
    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(matrix)
    ax.set_xticks(range(len(labels)), labels=labels)
    ax.set_yticks(range(len(labels)), labels=labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    return fig, ax

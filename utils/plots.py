import os
import json
import seaborn as sns
import matplotlib.pyplot as plt

plt.rc("font", **{"family": "cmr10", "size": 15})
plt.rc("axes", **{"formatter.use_mathtext": True})
plt.rc("text", usetex=True)


def plot_learning_curve(
    train_accuracies: list,
    val_accuracies: list,
    train_losses: list,
    val_losses: list,
    save_path: str,
):
    # Create the figure
    plt.figure(figsize=(12, 4))

    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(train_accuracies, label="Train Accuracy", color="b", marker="o")
    plt.plot(val_accuracies, label="Validation Accuracy", color="g", marker="x")
    plt.title("Model Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)

    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(train_losses, label="Train Loss", color="r", marker="o")
    plt.plot(val_losses, label="Validation Loss", color="m", marker="x")
    plt.title("Model Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    plt.savefig(
        os.path.join(save_path, "learning_curve.pdf"),
        format="pdf",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def plot_confusion_matrix(
    conf_matrix: list, target_names: list, save_path: str, prefix: str = None
):
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        conf_matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=target_names,
        yticklabels=target_names,
    )
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title("Confusion Matrix")
    plt.savefig(
        os.path.join(save_path, f"{prefix}confusion_matrix.pdf"),
        format="pdf",
        dpi=300,
        bbox_inches="tight",
    )


def plot_pareto_from_json(
    json_all_path: str,
    json_pareto_path: str,
    save_path: str,
    x_key="objective1",
    y_key="objective2",
    x_label="Val Acc",
    y_label="Energy (mJ)",
):
    with open(json_all_path, "r") as f:
        all_data = json.load(f)
    with open(json_pareto_path, "r") as f:
        pareto_data = json.load(f)

    all_points = {
        (round(t[x_key], 6), round(t[y_key], 6))
        for t in all_data
        if t[x_key] is not None and t[y_key] is not None
    }
    pareto_points = {
        (round(t[x_key], 6), round(t[y_key], 6))
        for t in pareto_data
        if t[x_key] is not None and t[y_key] is not None
    }

    non_pareto_points = all_points - pareto_points

    non_pareto_losses, non_pareto_hw = (
        zip(*non_pareto_points) if non_pareto_points else ([], [])
    )
    pareto_losses, pareto_hw = zip(*pareto_points) if pareto_points else ([], [])

    plt.figure(figsize=(4, 3))
    if non_pareto_points:
        plt.scatter(
            non_pareto_losses,
            non_pareto_hw,
            c="blue",
            label="Non-Pareto Front",
            alpha=0.6,
        )
    if pareto_points:
        plt.scatter(pareto_losses, pareto_hw, c="red", label="Pareto Front", alpha=0.8)

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight", format="pdf")
    plt.close()

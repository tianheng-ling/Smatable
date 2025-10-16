import os
import torch
import wandb
import torch.nn as nn
import torch.optim as optim
from tabulate import tabulate
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader

from config import DEVICE
from utils.EarlyStopping import EarlyStopping
from utils.plots import plot_learning_curve
from utils.exp_utils import setup_logger
from models.build_model import build_model


def train_val(
    model_config: dict, train_dataset: Dataset, val_dataset: Dataset, exp_config: dict
) -> None:

    # set parameters
    batch_size = exp_config["batch_size"]
    lr = exp_config["lr"]
    num_epochs = exp_config["num_epochs"]
    model_save_dir = exp_config["model_save_dir"]
    fig_save_dir = exp_config["fig_save_dir"]
    log_save_dir = exp_config["log_save_dir"]

    # create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # build model
    model = build_model(model_config, exp_config["enable_qat"]).to(DEVICE)
    wandb.log(model_config)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    early_stopping = EarlyStopping(
        patience=15, verbose=True, path=model_save_dir, monitor="val_acc", mode="max"
    )

    # Set up logging
    train_logger = setup_logger(
        "train_logger", os.path.join(log_save_dir, "train_logfile.log")
    )

    train_stats = defaultdict(list)
    val_stats = defaultdict(list)

    for epoch in range(num_epochs):
        # === Training ===
        model.train()
        train_loss, correct, total = 0.0, 0, 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        epoch_train_loss = train_loss / total
        epoch_train_acc = correct / total

        train_stats["loss"].append(epoch_train_loss)
        train_stats["acc"].append(epoch_train_acc)

        # Validation
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        epoch_val_loss = val_loss / val_total
        epoch_val_acc = val_correct / val_total

        val_stats["loss"].append(epoch_val_loss)
        val_stats["acc"].append(epoch_val_acc)

        # Early stopping check
        early_stopping(epoch_val_acc, model)
        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break

        # Log learning rate and epoch results
        headers = ["Epoch", "train_loss", "train_acc", "val_loss", "val_acc"]
        row = [
            [
                epoch + 1,
                f"{epoch_train_loss:.3f}",
                f"{epoch_train_acc:.3f}",
                f"{epoch_val_loss:.3f}",
                f"{epoch_val_acc:.3f}",
            ]
        ]
        train_logger.info(tabulate(row, headers=headers, tablefmt="pretty"))

        wandb.log(
            {
                "train_loss": epoch_train_loss,
                "train_acc": epoch_train_acc,
                "val_loss": epoch_val_loss,
                "val_acc": epoch_val_acc,
                "epoch": epoch + 1,
            }
        )

    # Plot accuracies and losses
    plot_learning_curve(
        train_accuracies=train_stats["acc"],
        val_accuracies=val_stats["acc"],
        train_losses=train_stats["loss"],
        val_losses=val_stats["loss"],
        save_path=fig_save_dir,
    )
    val_metrics = {
        "best_val_loss": min(val_stats["loss"]),
        "best_val_acc": max(val_stats["acc"]),
    }
    wandb.log(val_metrics)
    return val_metrics

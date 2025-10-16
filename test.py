import os
import torch
import wandb
from tabulate import tabulate
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

from config import DEVICE
from utils.exp_utils import setup_logger
from models.build_model import build_model
from utils.plots import plot_confusion_matrix
from utils.eval_metrics import get_model_complexity


def test(
    behaviors_list: list,
    test_dataset: Dataset,
    model_config: dict,
    exp_config: dict,
):

    batch_size = exp_config["batch_size"]
    exp_mode = exp_config["exp_mode"]
    model_save_dir = exp_config["model_save_dir"]
    fig_save_dir = exp_config["fig_save_dir"]
    log_save_dir = exp_config["log_save_dir"]

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # build model and load weights
    model = build_model(model_config, exp_config["enable_qat"]).to(DEVICE)
    print("Loading model weights from:", model_save_dir)
    checkpoint = torch.load(model_save_dir / "best_model.pth", weights_only=True)
    model.load_state_dict(checkpoint, strict=False)
    # print(model)

    # Set up logging
    test_logger = setup_logger(
        "test_logger", os.path.join(log_save_dir, "test_logfile.log")
    )

    # Print results
    prefix = (
        "int_"
        if exp_config["enable_qat"] == True
        and model_config["enable_int_forward"] == True
        else ""
    )

    # execute test
    model.eval()
    correct_preds = 0
    total_preds = 0
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            predicted = predicted.to(DEVICE)
            correct_preds += (predicted == labels).sum().item()
            total_preds += labels.size(0)

            # Collect labels and predictions for test_metrics
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    # Calculate test_metrics
    accuracy = correct_preds / total_preds
    precision = precision_score(
        all_labels, all_predictions, average="weighted", zero_division=0
    )
    recall = recall_score(all_labels, all_predictions, average="weighted")
    f1 = f1_score(all_labels, all_predictions, average="weighted")
    conf_matrix = confusion_matrix(all_labels, all_predictions)

    test_metrics = {
        f"{prefix}test_acc": accuracy,
        f"{prefix}test_precision": precision,
        f"{prefix}test_recall": recall,
        f"{prefix}test_f1": f1,
    }

    # calculate model complexity
    model_complexity = get_model_complexity(
        model_config, model, exp_config, prefix, inputs
    )

    test_logger.info(f"------------- {prefix}test_results -------------")
    headers = list(test_metrics.keys()) + list(model_complexity.keys())
    row = [
        [f"{test_metrics[k]:.3f}" for k in list(test_metrics.keys())]
        + [
            (
                model_complexity[k]
                if isinstance(model_complexity[k], (int, float))
                else str(model_complexity[k])
            )
            for k in model_complexity.keys()
        ]
    ]
    test_logger.info(tabulate(row, headers=headers, tablefmt="pretty"))
    if exp_mode == "train":
        wandb.log({**test_metrics, **model_complexity})

    # Visualize confusion matrix
    plot_confusion_matrix(
        conf_matrix, target_names=behaviors_list, save_path=fig_save_dir, prefix=prefix
    )

    return test_metrics

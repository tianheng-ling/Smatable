import os
import wandb
import logging
from pathlib import Path
from datetime import datetime


def setup_logger(logger_name, log_file, level=logging.INFO):
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)

    # Avoid adding handlers multiple times
    if not logger.hasHandlers():
        # File handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_formatter = logging.Formatter("[%(levelname)s] %(asctime)s - %(message)s")
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

        # Stream handler
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(level)
        stream_handler.setFormatter(file_formatter)
        logger.addHandler(stream_handler)

    return logger


def set_base_paths(exp_base_dir: str, given_timestamp: str):

    if given_timestamp is None:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    else:
        timestamp = given_timestamp

    model_save_dir = Path(exp_base_dir) / timestamp
    fig_save_dir, log_save_dir = model_save_dir / "figs", model_save_dir / "logs"
    for path in [model_save_dir, fig_save_dir, log_save_dir]:
        os.makedirs(path, exist_ok=True)
    return model_save_dir, fig_save_dir, log_save_dir


def get_paths_list(source_path: str):
    with open(source_path, "r") as f:
        paths_list = [line.strip() for line in f.read().splitlines()]
    return paths_list


def safe_print(label, value, unit=""):
    print(f"{label}: {value if value is not None else 'N/A'}{unit}")


def safe_wandb_log(hw_metrics: dict):
    wandb_log_dict = {}

    for k, v in hw_metrics.items():
        if v is None:
            continue
        if isinstance(v, dict):
            for sub_k, sub_v in v.items():
                if sub_v is not None:
                    wandb_log_dict[sub_k] = sub_v
        else:
            wandb_log_dict[k] = v

    if wandb_log_dict:
        wandb.log(wandb_log_dict)

import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_config = {"data_dir": "data/wav/"}

# ------------------ OPTUNA CONFIG ------------------
search_space = {
    "quant_bits": {"low": 4, "high": 8, "step": 2},
    "batch_size": {"low": 32, "high": 64, "step": 8},
    "lr": {"low": 1e-5, "high": 1e-3, "log": True},
    "num_blocks": {"low": 1, "high": 6, "step": 1},
}

ps_acc_thresholds = {
    8: 0.80,
    6: 0.75,
    4: 0.70,
}

loso_acc_thresholds = {
    8: 0.60,
    6: 0.55,
    4: 0.50,
}

aos_acc_thresholds = {
    8: 0.75,
    6: 0.70,
    4: 0.65,
}

STOP_TIME = "100ms"  # for better HCI
LATENCY_THRESHOLD = 100  # ms
POWER_THRESHOLD = 500  # mW
ENERGY_THRESHOLD = 100  # muJ

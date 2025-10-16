import torch
import wandb
import optuna

from data import (
    get_data,
    preprocess_data_waveform,
    preprocess_data_stft,
)
from data.CustomDataset import AudioDataset, AugumentedAudioDataset
from utils.exp_utils import set_base_paths, safe_print, safe_wandb_log
from train_val import train_val
from test import test
from hw_converter import convert2hw, run_hw_simulation
from utils import safe_print, set_base_paths


def run_one_experiment(
    data_config: dict,
    model_config: dict,
    exp_config: dict,
    wandb_config: dict,
    quant_config: dict = None,
    hw_config: dict = None,
    use_optuna: bool = False,
    optuna_config: dict = None,
):

    # Load preprocessed data
    behaviors_list, train_subjects_list, test_subjects_list = get_data(
        data_flag=data_config["data_flag"],
        data_splitting_method=data_config["data_splitting_method"],
        target_subject=data_config["target_subject"],
    )
    if (
        model_config["model_type"] == "2dcnn"
    ):  # for reproduce the results in reference paper
        x_train, y_train, x_val, y_val, x_test, y_test, le_classes = (
            preprocess_data_stft(
                behaviors_list=behaviors_list,
                train_subjects_list=train_subjects_list,
                test_subjects_list=test_subjects_list,
                data_path=data_config["data_dir"] + data_config["data_flag"],
            )
        )
        x_val, y_val = x_test, y_test
    else:
        x_train, y_train, x_val, y_val, x_test, y_test, le_classes = (
            preprocess_data_waveform(
                behaviors_list=behaviors_list,
                train_subjects_list=train_subjects_list,
                test_subjects_list=test_subjects_list,
                data_path=data_config["data_dir"] + data_config["data_flag"],
                normalization_type=data_config["normalization_type"],
            )
        )
        x_val, y_val = x_test, y_test

    # Convert to Torch tensors
    tensor_map = lambda x, dtype: torch.from_numpy(x).to(dtype)
    x_train, y_train = tensor_map(x_train, torch.float32), tensor_map(
        y_train, torch.long
    )

    x_val, y_val = tensor_map(x_val, torch.float32), tensor_map(y_val, torch.long)
    x_test, y_test = tensor_map(x_test, torch.float32), tensor_map(y_test, torch.long)

    # Define custom Dataset
    train_dataset = AugumentedAudioDataset(
        x_train, y_train, data_config["downsampling_rate"]
    )
    val_dataset = AugumentedAudioDataset(x_val, y_val, data_config["downsampling_rate"])
    test_dataset = AugumentedAudioDataset(
        x_test, y_test, data_config["downsampling_rate"]
    )

    print(
        f"Augumented Train data shape: {len(train_dataset)}, {train_dataset[0][0].shape}"
    )
    print(
        f"Augumented Validation data shape: {len(val_dataset)}, {val_dataset[0][0].shape}"
    )
    print(
        f"Augumented Test data shape: {len(test_dataset)}, {test_dataset[0][0].shape}"
    )
    print(f"Classes: {le_classes}")

    # Update model config
    model_config.update(
        {
            "in_channels": (
                int(x_train.shape[-1]) if model_config["model_type"] != "2dcnn" else 1
            ),
            "le_classes": len(le_classes),
        }
    )
    if model_config["model_type"] == "2dcnn":
        model_config.update(
            {
                "input_height": int(x_train.shape[1]),
                "input_width": int(x_train.shape[2]),
            }
        )
    if exp_config["enable_qat"]:
        model_config.update(
            {
                "seq_len": (int(x_train.shape[1] / data_config["downsampling_rate"])),
            }
        )

    # set exp_save_path
    model_save_dir, fig_save_dir, log_save_dir = set_base_paths(
        exp_config["exp_base_dir"], exp_config["given_timestamp"]
    )
    exp_config["model_save_dir"] = model_save_dir
    exp_config["fig_save_dir"] = fig_save_dir
    exp_config["log_save_dir"] = log_save_dir
    timestamp = str(exp_config["model_save_dir"]).split("/")[-1]

    if exp_config["exp_mode"] == "train":
        # set up wandb
        wandb.init(
            project=wandb_config["name"],
            mode=wandb_config["mode"],
            config=wandb_config["config"],
        )
        val_metrics = train_val(
            model_config=model_config,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            exp_config=exp_config,
        )

    test_metrics = test(
        behaviors_list=behaviors_list,
        test_dataset=test_dataset,
        model_config=model_config,
        exp_config=exp_config,
    )
    wandb.log({"timestamp": timestamp})

    int_test_metrics, hw_metrics = {}, {}
    hw_metrics["did_hw_simulation"] = False

    if quant_config is not None and exp_config["enable_qat"]:
        model_config["enable_int_forward"] = True
        int_test_metrics = test(
            behaviors_list=behaviors_list,
            test_dataset=test_dataset,
            model_config=model_config,
            exp_config=exp_config,
        )
        if use_optuna:
            if data_config["data_splitting_method"] == "PS":
                optuna_config["acc_thresholds"] = optuna_config["ps_acc_thresholds"]
            elif data_config["data_splitting_method"] == "LOSO":
                optuna_config["acc_thresholds"] = optuna_config["loso_acc_thresholds"]
            elif data_config["data_splitting_method"] == "AOS":
                optuna_config["acc_thresholds"] = optuna_config["aos_acc_thresholds"]
            else:
                raise ValueError(
                    f"Unsupported data splitting method: {data_config['data_splitting_method']}"
                )
            threshold = optuna_config["acc_thresholds"][quant_config["quant_bits"]]

            if val_metrics["best_val_acc"] < threshold:

                print(
                    f"[PRUNE] Accuracy {val_metrics['best_val_acc']:.3f} < threshold {threshold:.3f}"
                )
                hw_metrics["failure_type"] = "accuracy_failure"
                wandb.log(hw_metrics)
                return (
                    timestamp,
                    val_metrics,
                    test_metrics,
                    int_test_metrics,
                    hw_metrics,
                )

        if hw_config is not None and exp_config["enable_hw_simulation"]:

            # generate necessary files for hardware simulation
            convert2hw(
                test_dataset=test_dataset,
                subset_size=hw_config["subset_size"],
                model_config=model_config,
                model_save_dir=exp_config["model_save_dir"],
                target_hw=hw_config["target_hw"],
            )

            # run hardware simulation
            hw_metrics = run_hw_simulation(
                model_save_dir=exp_config["model_save_dir"],
                hw_config=hw_config,
            )

            hw_metrics["did_hw_simulation"] = True
            safe_print("Resource Utilization", hw_metrics["res_info"])
            safe_print("Time", hw_metrics["time(ms)"], " (ms)")
            safe_print("Power Consumption", hw_metrics["power_info"])
            safe_print("Energy Consumption", hw_metrics["energy(mJ)"])

            # log hw metrics to wandb
            if use_optuna == False:
                safe_wandb_log(hw_metrics)

    return (timestamp, val_metrics, test_metrics, int_test_metrics, hw_metrics)

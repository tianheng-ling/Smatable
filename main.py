import wandb
import argparse

from config import data_config
from run_one_experiment import run_one_experiment


def main(args):

    # set data config
    data_config.update(
        {
            "data_flag": args.data_flag,
            "normalization_type": args.normalization_type,
            "data_splitting_method": args.data_splitting_method,
            "target_subject": args.target_subject,
            "downsampling_rate": args.downsampling_rate,
        }
    )

    # set exp_config
    exp_config = {
        "exp_mode": args.exp_mode,
        "exp_base_dir": args.exp_base_dir,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "num_epochs": args.num_epochs,
        "enable_qat": args.enable_qat,
        "given_timestamp": None,
        "enable_hw_simulation": args.enable_hw_simulation,
    }

    # set model config
    model_config = {
        "model_type": args.model_type,
        "num_blocks": args.num_blocks,
        "p": args.p,
    }

    wandb_config = {
        "name": args.wandb_project_name,
        "mode": args.wandb_mode,
        "config": {**data_config, **model_config, **exp_config},
    }

    quant_config = None
    if args.enable_qat:
        quant_config = {
            "model_name": "network",
            "quant_bits": args.quant_bits,
        }
        model_config.update(
            {
                "name": quant_config["model_name"],
                "quant_bits": quant_config["quant_bits"],
                "enable_int_forward": False,
            }
        )
        wandb_config["config"].update({**model_config, **quant_config})

    hw_config = None
    if exp_config["enable_qat"] and exp_config["enable_hw_simulation"]:
        hw_config = {
            "top_module": quant_config["model_name"],
            "subset_size": args.subset_size,
            "target_hw": args.target_hw,
            "fpga_type": args.fpga_type,
        }
        wandb_config["config"].update(hw_config)

    run_one_experiment(
        data_config=data_config,
        model_config=model_config,
        exp_config=exp_config,
        wandb_config=wandb_config,
        quant_config=quant_config,
        hw_config=hw_config,
    )
    wandb.finish()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # wandb settings
    parser.add_argument("--wandb_project_name", type=str)
    parser.add_argument(
        "--wandb_mode",
        type=str,
        choices=["online", "offline", "disabled"],
    )

    # model settings
    parser.add_argument(
        "--model_type",
        type=str,
        choices=[
            "2dcnn",
            "1dcnn",
            "1dsepcnn",
            "1dsepcnnfused",
        ],
    )
    parser.add_argument("--p", type=float, help="Dropout probability")
    parser.add_argument("--num_blocks", type=int, help="Number of residual blocks")

    # data settings
    parser.add_argument(
        "--data_flag", type=str, choices=["DatabyPerson", "DatabyTable"]
    )
    parser.add_argument(
        "--data_splitting_method",
        type=str,
        choices=["PS", "LOSO", "AOS"],
    )
    parser.add_argument(
        "--target_subject", type=str, choices=["A", "B", "C"], help="Person or Table"
    )
    parser.add_argument(
        "--normalization_type",
        type=str,
        choices=["standard", "minmax", "maxabs"],
        default="standard",
    )
    parser.add_argument(
        "--downsampling_rate",
        type=int,
        default=None,
        help="Only for waveform data",
    )

    # experiment settings
    parser.add_argument("--exp_mode", type=str, choices=["train", "test"])
    parser.add_argument("--exp_base_dir", type=str)
    parser.add_argument("--batch_size", type=int, help="Batch size for training")
    parser.add_argument("--lr", type=float, help="Learning rate for training")
    parser.add_argument("--num_epochs", type=int, help="Number of epochs for training")
    parser.add_argument("--enable_qat", action="store_true", help="Quantize the model")
    parser.add_argument("--enable_hw_simulation", action="store_true")

    # quantization settings
    parser.add_argument("--quant_bits", type=int)

    # hw simulation config
    parser.add_argument(
        "--subset_size",
        type=int,
    )
    parser.add_argument(
        "--target_hw",
        type=str,
        choices=["amd"],
    )
    parser.add_argument(
        "--fpga_type",
        type=str,
        choices=["xc7s50ftgb196-2", "xc7s25ftgb196-2", "xc7s15ftgb196-2"],
        default="xc7s15ftgb196-2",
    )
    args = parser.parse_args()

    main(args)

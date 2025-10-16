import os
import wandb
import optuna
import argparse
from functools import partial
import optuna.visualization as vis
from optuna.storages import RDBStorage
from optuna.samplers import NSGAIISampler

from config import (
    data_config,
    search_space,
    ps_acc_thresholds,
    loso_acc_thresholds,
    aos_acc_thresholds,
    LATENCY_THRESHOLD,
    POWER_THRESHOLD,
    ENERGY_THRESHOLD,
)
from run_one_experiment import run_one_experiment
from utils import save_trials_records, plot_pareto_from_json, safe_wandb_log


def objective(trial, args):

    try:
        # optuna search space
        quant_bits = (
            trial.suggest_int("quant_bits", **search_space["quant_bits"])
            if args.enable_qat
            else None
        )
        batch_size = trial.suggest_int("batch_size", **search_space["batch_size"])
        lr = trial.suggest_float("lr", **search_space["lr"])
        num_blocks = trial.suggest_int("num_blocks", **search_space["num_blocks"])

        # set optuna config
        optuna_config = {
            "optuna_hw_target": args.optuna_hw_target,
            "ps_acc_thresholds": ps_acc_thresholds,
            "loso_acc_thresholds": loso_acc_thresholds,
            "aos_acc_thresholds": aos_acc_thresholds,
        }
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
            "batch_size": batch_size,
            "lr": lr,
            "num_epochs": args.num_epochs,
            "enable_qat": args.enable_qat,
            "given_timestamp": None,
            "enable_hw_simulation": args.enable_hw_simulation,
        }

        # set model config
        model_config = {
            "model_type": args.model_type,
            "num_blocks": num_blocks,
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
                "quant_bits": quant_bits,
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

        timestamp, val_metrics, test_metrics, int_test_metrics, hw_metrics = (
            run_one_experiment(
                data_config=data_config,
                model_config=model_config,
                exp_config=exp_config,
                wandb_config=wandb_config,
                quant_config=quant_config,
                hw_config=hw_config,
                use_optuna=True,
                optuna_config=optuna_config,
            )
        )
        acc_target = val_metrics["best_val_acc"]
        user_attrs = {
            "timestamp": timestamp,
            **val_metrics,
            **test_metrics,
        }

        if args.enable_qat:
            user_attrs.update(**int_test_metrics)  # of quantized model

            if args.enable_hw_simulation == False:
                return (
                    acc_target  # if no hardware simulation is needed, single objective
                )
            else:

                for k, v in hw_metrics.items():
                    if isinstance(v, dict):
                        for kk, vv in v.items():
                            user_attrs[f"{k}/{kk}"] = vv
                    else:
                        user_attrs[k] = v

                if hw_metrics["failure_type"] == "accuracy_failure":
                    raise optuna.exceptions.TrialPruned()
                else:
                    # === Add Deployability Condition ===
                    if not hw_metrics["res_info"].get("is_deployable"):
                        print(
                            f"[PRUNE] Resource Utilization {hw_metrics['res_info']} is not deployable"
                        )
                        hw_metrics["failure_type"] = "deployability_failure"
                        safe_wandb_log(hw_metrics)
                        trial.set_user_attr(user_attrs)
                        raise optuna.exceptions.TrialPruned()
                    else:
                        # === Add Latency Condition ===
                        if hw_metrics["time(ms)"] is None or float(
                            hw_metrics["time(ms)"]
                        ) > float(LATENCY_THRESHOLD):
                            print(
                                f"[PRUNE] Latency {hw_metrics['time(ms)']} > LATENCY_THRESHOLD {LATENCY_THRESHOLD}"
                            )
                            hw_metrics["failure_type"] = "latency_failure"
                            safe_wandb_log(hw_metrics)
                            trial.set_user_attr(user_attrs)
                            raise optuna.exceptions.TrialPruned()
                        else:
                            # === Add Power Condition ===
                            if (
                                hw_metrics["power_info"].get("total_power(mW)") is None
                                or float(
                                    hw_metrics["power_info"].get("total_power(mW)")
                                )
                                > POWER_THRESHOLD
                            ):
                                print("[PRUNE] Power info is not available")
                                hw_metrics["failure_type"] = "power_failure"
                                safe_wandb_log(hw_metrics)
                                trial.set_user_attr(user_attrs)
                                raise optuna.exceptions.TrialPruned()
                            else:
                                # === Add Energy Condition ===
                                if (
                                    hw_metrics["energy(mJ)"] is None
                                    or float(hw_metrics["energy(muJ)"])
                                    > ENERGY_THRESHOLD
                                ):
                                    print("[PRUNE] Energy info is not available")
                                    hw_metrics["failure_type"] = "energy_failure"
                                    safe_wandb_log(hw_metrics)
                                    trial.set_user_attr(user_attrs)
                                    raise optuna.exceptions.TrialPruned()
                                else:
                                    # if all constraints are satisfied, return the target hw metric
                                    safe_wandb_log(hw_metrics)
                                    if args.optuna_hw_target == "power":
                                        return acc_target, hw_metrics["power_info"].get(
                                            "total_power(mW)"
                                        )
                                    elif args.optuna_hw_target == "latency":
                                        return acc_target, hw_metrics["time(ms)"]
                                    elif args.optuna_hw_target == "energy":
                                        return acc_target, hw_metrics["energy(mJ)"]
                                    else:
                                        raise ValueError(
                                            f"Unsupported optuna_hw_target: {args.optuna_hw_target}"
                                        )

        return acc_target

    except Exception as e:
        print(f"Trial {trial.number} failed with error: {e}")
        # raise optuna.exceptions.TrialPruned()
    finally:
        wandb.finish()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # wandb config
    parser.add_argument("--wandb_project_name", type=str)
    parser.add_argument("--wandb_mode", type=str)

    # model config
    parser.add_argument(
        "--model_type",
        type=str,
        choices=[
            "2dcnn",
            "1dcnn",
            "1dsepcnnfused",
        ],
    )
    parser.add_argument(
        "--p",
        type=float,
        help="Dropout probability",
    )

    # data config
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

    # experiment config
    parser.add_argument("--exp_mode", type=str, choices=["train", "test"])
    parser.add_argument("--exp_base_dir", type=str)
    parser.add_argument("--num_epochs", type=int)
    parser.add_argument("--enable_qat", action="store_true")
    parser.add_argument("--enable_hw_simulation", action="store_true")

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

    # optuna configs
    parser.add_argument("--n_trials", type=int, default=1)
    parser.add_argument(
        "--optuna_hw_target",
        type=str,
        choices=["power", "latency", "energy", None],
        default=None,
    )
    args = parser.parse_args()

    # ------------------------- setup optuna db -------------------------
    os.makedirs(args.exp_base_dir, exist_ok=True)
    study_name = f"{args.data_splitting_method}_{args.enable_qat}"
    db_path = os.path.join(args.exp_base_dir, f"{study_name}.db")
    storage = RDBStorage(f"sqlite:///{db_path}")

    if args.enable_qat and args.enable_hw_simulation:
        directions = ["maximize", "minimize"]  # [val_acc, hw_metric]
    else:
        directions = ["maximize"]  # [val_acc]

    study = optuna.create_study(
        directions=directions,
        sampler=NSGAIISampler(),
        storage=storage,
        load_if_exists=True,
        study_name=study_name,
    )
    study.optimize(
        partial(objective, args=args), n_trials=args.n_trials, catch=(Exception,)
    )

    json_all_path = f"{args.exp_base_dir}/all_trials.json"
    json_pareto_path = f"{args.exp_base_dir}/pareto_trials.json"
    save_trials_records(json_path=json_all_path, study=study, only_best=False)
    save_trials_records(json_path=json_pareto_path, study=study, only_best=True)

    if len(directions) == 2:
        plot_pareto_from_json(
            json_all_path,
            json_pareto_path,
            save_path=f"{args.exp_base_dir}/pareto_plot.pdf",
        )
    elif len(directions) == 1:
        fig1 = vis.plot_optimization_history(study)
        fig2 = vis.plot_param_importances(study)
        fig1.write_html(f"{args.exp_base_dir}/optimization_history.html")
        fig2.write_html(f"{args.exp_base_dir}/param_importances.html")
    else:
        raise ValueError("Invalid number of objectives. Must be 1 or 2.")

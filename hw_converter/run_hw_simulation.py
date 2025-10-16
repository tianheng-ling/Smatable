import os

from hw_converter import vivado_runner


def run_hw_simulation(
    model_save_dir: str,
    hw_config: dict,
):
    target_hw = hw_config["target_hw"]
    runner_kwargs = {
        "top_module": hw_config["top_module"],
        "base_dir": os.path.abspath(os.path.join(model_save_dir, "hw", target_hw)),
        "fpga_type": hw_config["fpga_type"],
    }
    return vivado_runner(**runner_kwargs)

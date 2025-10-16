import os
import shutil
import subprocess

from hw_converter.run_ghdl_simulation import run_ghdl_simulation
from hw_converter.analyze_ghdl_report import (
    get_ghdl_simulation_time,
)
from hw_converter.analyze_vivado_report import (
    analyze_resource_utilization,
    analyze_power_consumption,
)
from hw_converter.utils import copy_to_dir


def run_resource_estimation(
    tmp_dir: str,
    tcl_path: str,
    report_dir,
    top_module: str,
    source_dir: str,
    data_dir: str,
    const_dir: str,
    firmware_dir: str,
    fpga_type: str,
):
    try:
        # copy necessary files to tmp_dir
        copy_to_dir(src=tcl_path, dest=tmp_dir)
        copy_to_dir(src=source_dir, dest=tmp_dir)
        copy_to_dir(src=data_dir, dest=tmp_dir)
        copy_to_dir(src=const_dir, dest=tmp_dir)

        # replacing the testbed data path
        absolute_data_path = os.path.join(tmp_dir, "data")
        firmware_dest = os.path.join(tmp_dir, "source")
        subprocess.run(f"cp -r {firmware_dir}/* {firmware_dest}", shell=True)
        tb_path = os.path.join(tmp_dir, "source", top_module, f"{top_module}_tb.vhd")
        subprocess.run(
            ["sed", "-i", f"s|./data|{absolute_data_path}|g", tb_path], check=True
        )

        # execute resource estimation
        vivado_cmd = (
            f"bash -c 'source /tools/Xilinx/Vivado/2019.2/settings64.sh && "
            f"vivado -mode batch -nolog -nojournal -source resource_estimation.tcl -tclargs {report_dir} {fpga_type}'"
        )
        subprocess.run(vivado_cmd, shell=True, cwd=tmp_dir, check=True, timeout=7200)
    except Exception as e:
        print(f"[Resource Estimation] Failed for top module '{top_module}': {e}")


def run_power_estimation(
    tmp_dir: str,
    power_tcl_path: str,
    report_dir: str,
    top_module: str,
    time_fs: float,
    fpga_type: str,
):
    try:
        saif_path = os.path.join(tmp_dir, "sim_wave.saif")
        copy_to_dir(src=power_tcl_path, dest=tmp_dir)
        vivado_cmd = (
            f"bash -c 'source /tools/Xilinx/Vivado/2019.2/settings64.sh && "
            f"vivado -mode batch -nolog -nojournal -source power_estimation.tcl -tclargs {saif_path} {report_dir} {time_fs} {top_module} {fpga_type}'"
        )
        subprocess.run(vivado_cmd, shell=True, cwd=tmp_dir, check=True, timeout=7200)
    except Exception as e:
        print(f"[Power Estimation] Failed for top module '{top_module}': {e}")


def vivado_runner(base_dir: str, top_module: str, fpga_type: str):

    # Setup dirs
    data_dir = os.path.join(base_dir, "data")
    source_dir = os.path.join(base_dir, "source")
    makefile_path = os.path.join(base_dir, "makefile")
    const_dir = os.path.join(base_dir, "constraints")
    firmware_dir = os.path.join(base_dir, "firmware")
    vivado_tcl_dir = "hw_converter/tcl_files/amd/"
    resource_tcl_path = f"{vivado_tcl_dir}resource_estimation.tcl"
    power_tcl_path = f"{vivado_tcl_dir}power_estimation.tcl"

    # Initialize hw_metrics
    hw_metrics = {
        "res_info": {},
        "time(ms)": None,
        "power_info": {},
        "energy(mJ)": None,
    }

    # [1] Run GHDL simulation and get simulation time
    run_ghdl_simulation(
        base_dir=base_dir,
        source_dir=source_dir,
        data_dir=data_dir,
        makefile_path=makefile_path,
    )
    # TODO: time breakdown of each module
    time_fs = get_ghdl_simulation_time(
        ghdl_report_dir=os.path.join(base_dir, "ghdl_report"),
        target_module=top_module,
    )
    if time_fs is None:
        return hw_metrics
    else:
        time_ms = time_fs / 1e12  # fs → ms
        hw_metrics["time(ms)"] = time_ms

        # [2] Estimate resource utilization and parse resource utilization report
        tmp_vivado_dir = os.path.join(base_dir, "tmp_vivado_proj")
        vivado_report_dir = os.path.abspath(os.path.join(base_dir, "vivado_report"))
        for path in [tmp_vivado_dir, vivado_report_dir]:
            shutil.rmtree(path, ignore_errors=True)
            os.makedirs(path, exist_ok=True)
        run_resource_estimation(
            tmp_dir=tmp_vivado_dir,
            tcl_path=resource_tcl_path,
            report_dir=vivado_report_dir,
            top_module=top_module,
            source_dir=source_dir,
            data_dir=data_dir,
            const_dir=const_dir,
            firmware_dir=firmware_dir,
            fpga_type=fpga_type,
        )
        hw_metrics["res_info"] = analyze_resource_utilization(
            report_dir=vivado_report_dir
        )
        if hw_metrics["res_info"].get("is_deployable") == False:
            return hw_metrics
        else:
            # [3] Power estimation
            run_power_estimation(
                tmp_dir=tmp_vivado_dir,
                power_tcl_path=power_tcl_path,
                report_dir=vivado_report_dir,
                top_module=top_module,
                time_fs=time_fs,
                fpga_type=fpga_type,
            )
            power_info = analyze_power_consumption(report_dir=vivado_report_dir)
            if power_info is None:
                return hw_metrics
            else:
                hw_metrics["power_info"] = power_info

                # [4] Calculate energy consumption
                energy_used = round(power_info["total_power(mW)"] * time_ms / 1000, 3)
                hw_metrics["energy(mJ)"] = energy_used
                return hw_metrics

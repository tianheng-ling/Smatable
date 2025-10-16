import os


def get_ghdl_simulation_time(ghdl_report_dir: str, target_module: str):
    report_path = os.path.join(ghdl_report_dir, f"ghdl_{target_module}_output.txt")

    if not os.path.exists(report_path):
        raise FileNotFoundError(f"Report file not found: {report_path}")

    with open(report_path, "r") as f:
        lines = f.readlines()

    # Step 1: check difference
    for line in lines:
        if "Differece" in line:
            parts = line.split("Differece =")
            if len(parts) > 1:
                try:
                    diff_val = int(parts[1].strip())
                    if diff_val != 0:
                        print(f"Mismatch in Differece = {diff_val} in {target_module}")
                        return None
                except ValueError:
                    print(f"Unable to parse difference value: {line.strip()}")
                    return None

    # Step 2: extract time
    for line in lines:
        if "Time taken for processing" in line:
            parts = line.split("=")
            value_str = parts[1].split("fs")[0].strip()
            return float(value_str)
        if (
            "simulation stopped by --stop-time" in line
        ):  # soft contraint to limit simulation time
            print(
                f"Simulation stopped by --stop-time in {target_module}. Check the simulation settings."
            )
            return None

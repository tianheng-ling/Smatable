import os
import shutil


def copy_to_dir(src: str, dest: str):
    if os.path.isdir(src):
        shutil.copytree(
            src, os.path.join(dest, os.path.basename(src)), dirs_exist_ok=True
        )
    else:
        shutil.copy(src, dest)


def deployability_check(res_info: dict) -> bool:
    for key, val in res_info.items():
        if key.endswith("_used_util") and val > 100.0:
            return False
    return True

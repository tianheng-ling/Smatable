import torch
from thop import profile

from config import DEVICE


def get_model_complexity(
    model_config: dict,
    model: torch.nn.Module,
    exp_config: dict,
    prefix: str,
    inputs: int,
):
    # determine the data type bit widths
    unit_bits = (
        model_config["quant_bits"]
        if exp_config["enable_qat"] == True
        and model_config["enable_int_forward"] == True
        else 32
    )
    unit_bytes = unit_bits / 8  # default FP32 (4 bytes)

    # calculate the parameters
    param_size = 0
    param_amount = 0
    param_tensors = 0
    for param in model.parameters():
        param_size += param.numel() * unit_bytes  # Calculate parameter size
        param_amount += param.numel()  # Calculate parameter amount
        param_tensors += 1  # Calculate parameter tensors
    param_size /= 1e3  # Convert to kilobytes(KB)

    # calculate the buffers
    buffer_size = 0
    buffer_amount = 0
    buffer_tensors = 0
    for buffer in model.buffers():
        buffer_size += buffer.numel() * unit_bytes  # Calculate buffer size
        buffer_amount += buffer.numel()  # Calculate buffer amount
        buffer_tensors += 1  # Calculate buffer tensors
    buffer_size /= 1e3  # Convert to kilobytes(KB)

    # calculate the total model size
    model_size = param_size + buffer_size  # Calculate total model size
    if exp_config["enable_qat"] == False or (
        exp_config["enable_qat"] == True and model_config["enable_int_forward"] == False
    ):
        memory_size = torch.cuda.max_memory_allocated() / 1e6  # Convert to MB
        dummy_inputs = torch.randn(1, *inputs.shape[1:]).to(DEVICE)
        flops, _ = profile(model, inputs=(dummy_inputs,), verbose=False)
        gflops = flops / 1e9  # Convert to gigaFLOPS(GFLOPS)
    else:
        memory_size = None
        flops = None
        gflops = None

    return {
        f"{prefix}param_size (KB)": f"{param_size:.3f}",
        f"{prefix}param_amount": int(param_amount),
        f"{prefix}param_tensors": param_tensors,
        f"{prefix}buffer_size (KB)": f"{buffer_size:.3f}",
        f"{prefix}buffer_amount": int(buffer_amount),
        f"{prefix}buffer_tensors": buffer_tensors,
        f"{prefix}model_size (KB)": f"{model_size:.3f}",
        f"{prefix}memory_size (MB)": f"{memory_size:.3f}" if memory_size else None,
        f"{prefix}FLOPs": f"{flops:.3f}" if flops else None,
        f"{prefix}GFLOPs": f"{gflops:.3f}" if gflops else None,
    }

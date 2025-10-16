from .fp32 import (
    Float2DCNN,
    Float1DCNN,
    Float1DSepCNN,
)
from .quant import (
    Quant1DCNN,
    Quant1DSepCNN,
    Quant1DSepCNNFused,
)


def build_model(model_config: dict, enable_qat) -> object:

    model_type = model_config.get("model_type")

    if enable_qat:
        model_map = {
            "1dcnn": Quant1DCNN,
            "1dsepcnn": Quant1DSepCNN,
            "1dsepcnnfused": Quant1DSepCNNFused,
        }
    else:
        model_map = {
            "2dcnn": Float2DCNN,
            "1dcnn": Float1DCNN,
            "1dsepcnn": Float1DSepCNN,
        }
    return model_map[model_type](**model_config)

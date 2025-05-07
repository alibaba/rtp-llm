from typing import Any
from maga_transformer.utils.model_weight import W
from maga_transformer.model_loader.weight_module import WeightModule, AtomicWeight
from maga_transformer.model_loader.ffn_weight import FfnAtomicWeight, MoeAtomicWeight
from maga_transformer.model_loader.attn_weight import AttnAtomicWeight, MlaAttnAtomicWeight

class W8A8Int8AtomicWeight(AtomicWeight):
    gpt_style_tp_strategy = W.gemm_int8_gpt_style_tp_strategy()
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)

    def _get_split_func(self):
        return self.gpt_style_tp_strategy[self.name]

class W8A8Int8AttnAtomicWeight(AttnAtomicWeight, W8A8Int8AtomicWeight):
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)

class W8A8Int8MlaAttnAtomicWeight(MlaAttnAtomicWeight, W8A8Int8AtomicWeight):
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)

class W8A8Int8FfnAtomicWeight(FfnAtomicWeight, W8A8Int8AtomicWeight):
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)

class W8A8Int8MoeAtomicWeight(MoeAtomicWeight, W8A8Int8AtomicWeight):
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)


def create_w8a8_int8_weight(src_weight_info: WeightModule, *args: Any, **kwargs: Any) -> W8A8Int8AtomicWeight :
    if isinstance(src_weight_info, MlaAttnAtomicWeight):
        return W8A8Int8MlaAttnAtomicWeight(*args, **kwargs)
    if isinstance(src_weight_info, AttnAtomicWeight):
        return W8A8Int8AttnAtomicWeight(*args, **kwargs)
    if isinstance(src_weight_info, MoeAtomicWeight):
        return W8A8Int8MoeAtomicWeight(*args, **kwargs)
    if isinstance(src_weight_info, FfnAtomicWeight):
        return W8A8Int8FfnAtomicWeight(*args, **kwargs)
    if isinstance(src_weight_info, AtomicWeight):
        return W8A8Int8AtomicWeight(*args, **kwargs)
    raise NotImplementedError(f"Unsupported weight type: {src_weight_info}")

class W8A8Fp8AtomicWeight(AtomicWeight):
    gpt_style_tp_strategy = W.gemm_fp8_per_tensor_gpt_style_tp_strategy()
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)

    def _get_split_func(self):
        return self.gpt_style_tp_strategy[self.name]


class W8A8Fp8AttnAtomicWeight(AttnAtomicWeight, W8A8Fp8AtomicWeight):
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)

class W8A8Fp8MlaAttnAtomicWeight(MlaAttnAtomicWeight, W8A8Fp8AtomicWeight):
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)

class W8A8Fp8FfnAtomicWeight(FfnAtomicWeight, W8A8Fp8AtomicWeight):
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)

class W8A8Fp8MoeAtomicWeight(MoeAtomicWeight, W8A8Fp8AtomicWeight):
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)

def create_w8a8_fp8_weight(src_weight_info: WeightModule, *args: Any, **kwargs: Any) -> W8A8Fp8AtomicWeight :
    if isinstance(src_weight_info, MlaAttnAtomicWeight):
        return W8A8Fp8MlaAttnAtomicWeight(*args, **kwargs)
    if isinstance(src_weight_info, AttnAtomicWeight):
        return W8A8Fp8AttnAtomicWeight(*args, **kwargs)
    if isinstance(src_weight_info, MoeAtomicWeight):
        return W8A8Fp8MoeAtomicWeight(*args, **kwargs)
    if isinstance(src_weight_info, FfnAtomicWeight):
        return W8A8Fp8FfnAtomicWeight(*args, **kwargs)
    if isinstance(src_weight_info, AtomicWeight):
        return W8A8Fp8AtomicWeight(*args, **kwargs)
    raise NotImplementedError(f"Unsupported weight type: {src_weight_info}")

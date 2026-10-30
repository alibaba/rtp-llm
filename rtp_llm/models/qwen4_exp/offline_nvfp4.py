"""Load ModelOpt's split NVFP4 experts without dequantizing/requantizing them."""

import torch

from rtp_llm.config.quant_config import QWEN4_NVFP4_RECIPE, ModelOptFp4Config
from rtp_llm.model_loader.ffn_weight import MoeWeight
from rtp_llm.model_loader.model_weight_info import ModelWeightInfo
from rtp_llm.model_loader.per_group_fp4_quant_weight import PerGroupFp4Weight
from rtp_llm.utils.model_weight import W


def _scalar_vector(tensors):
    if not tensors or any(t.numel() != 1 for t in tensors):
        raise ValueError("Qwen4 NVFP4 requires one scalar per expert projection")
    values = torch.stack([t.reshape(()) for t in tensors])
    if not bool(torch.isfinite(values).all()) or not bool((values > 0).all()):
        raise ValueError("Qwen4 NVFP4 scales must be finite and positive")
    return values


def _equal_up_gate_scalars(tensors):
    values = _scalar_vector(tensors)
    if values.numel() % 2:
        raise ValueError("Qwen4 NVFP4 requires paired up/gate scales")
    up, gate = values.chunk(2)
    # Taking max would change the effective weights. ModelOpt's fused export
    # shares these scales; reject other exports rather than silently requantize.
    if not torch.equal(up, gate):
        raise ValueError("Qwen4 NVFP4 requires equal up/gate scales")
    return up.contiguous()


def _validate_quant_config(quant_config):
    if (
        not isinstance(quant_config, ModelOptFp4Config)
        or quant_config.model_recipe != QWEN4_NVFP4_RECIPE
        or not quant_config.is_quanted()
        or quant_config.bits != 4
        or quant_config.group_size() != 16
    ):
        raise ValueError("Qwen4 offline NVFP4 requires the QWEN4_NVFP4 recipe")


class Qwen4Nvfp4MoeWeight(PerGroupFp4Weight):
    """Retain packed bytes, pair gate/up scales and normalize scalar shapes.

    The existing descriptor owns expert selection, block-aligned TP splitting,
    device placement and the one-time CUTLASS block-scale swizzle.
    """

    @classmethod
    def support(cls, quant_config, src_weight_info):
        # Model-local selection; never compete with the generic FP4 registry.
        return False

    def __init__(self, source, quant_config):
        _validate_quant_config(quant_config)
        if source.name not in (W.moe_w1, W.moe_w2) or source.stacked_ckpt_keys:
            raise ValueError("Qwen4 offline NVFP4 requires split expert keys")
        super().__init__(source, quant_config, name=source.name)

    @staticmethod
    def _scalar_companion(source, process_fun):
        return type(source)(
            source.name,
            source.weights,
            process_fun=process_fun,
            data_type=torch.float32,
            config=source.config,
        )

    def _get_moe_w1_quant_weight(self, source):
        kernel, scale, scale_2, input_scale = super()._get_moe_w1_quant_weight(source)
        return (
            kernel,
            scale,
            self._scalar_companion(scale_2, _equal_up_gate_scalars),
            self._scalar_companion(input_scale, _equal_up_gate_scalars),
        )

    def _get_moe_w2_quant_weight(self, source):
        kernel, scale, scale_2, input_scale = super()._get_moe_w2_quant_weight(source)
        return (
            kernel,
            scale,
            self._scalar_companion(scale_2, _scalar_vector),
            self._scalar_companion(input_scale, _scalar_vector),
        )

    def _load_raw_tensor(self, tensor_source, layer_id, device, load_config):
        count = self.kernel.config.expert_num
        if count % load_config.ep_size:
            raise ValueError("Qwen4 NVFP4 expert count must divide EP size")
        local_count = count // load_config.ep_size
        expected = list(
            range(
                local_count * load_config.ep_rank,
                local_count * (load_config.ep_rank + 1),
            )
        )
        if (
            not expected
            or list(load_config.get_selected_experts(layer_id, count)) != expected
        ):
            raise ValueError("Qwen4 NVFP4 requires contiguous logical expert ownership")
        result = super()._load_raw_tensor(tensor_source, layer_id, device, load_config)
        scale = result[self.input_scale.name]
        if scale.shape != (local_count,) or not bool((scale == scale[0]).all()):
            raise ValueError(
                "Qwen4 NVFP4 requires uniform activation scales across experts"
            )
        return result


class Qwen4Nvfp4WeightInfo(ModelWeightInfo):
    """Quantize only routed descriptors; all other checkpoint weights stay raw."""

    def to_quant_weight_info(self, quant_config):
        _validate_quant_config(quant_config)
        layers = []
        for weights in self.layer_weights:
            layer = []
            for weight in weights:
                if isinstance(weight, MoeWeight):
                    weight = MoeWeight(
                        sub_weights=[
                            (
                                Qwen4Nvfp4MoeWeight(child, quant_config)
                                if child.name in (W.moe_w1, W.moe_w2)
                                else child
                            )
                            for child in weight.sub_weights.values()
                        ],
                        config=weight.config,
                    )
                layer.append(weight)
            layers.append(layer)
        return ModelWeightInfo(self.weights, layers)

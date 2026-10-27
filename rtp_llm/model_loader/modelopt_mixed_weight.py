"""Load mixed FP8/NVFP4 without interpreting packed codes as BF16 weights."""

import copy

import torch

from rtp_llm.config.modelopt_mixed import ModelOptMixedPrecisionConfig
from rtp_llm.config.quant_config import ModelOptFp4Config
from rtp_llm.model_loader.ffn_weight import MoeAtomicWeight
from rtp_llm.model_loader.per_group_fp4_quant_weight import PerGroupFp4Weight
from rtp_llm.model_loader.weight_module import (
    CompositeWeight,
    CustomAtomicWeight,
    QuantWeight,
)
from rtp_llm.utils.model_weight import CkptWeightInfo, W, identity, sp_0_pad8, sp_id

FP8_SCALES = {
    W.attn_qkv_w: W.attn_qkv_s,
    W.attn_gate_w: W.attn_gate_s,
    W.attn_o_w: W.attn_o_s,
    W.linear_attn_qkvz_w: W.linear_attn_qkvz_s,
    W.linear_attn_out_w: W.linear_attn_out_s,
}


def _scalar(source, name, device):
    value = identity(source.load_tensor(name, torch.float32)).to(device)
    if (
        value.numel() != 1
        or not bool(torch.isfinite(value).all())
        or not bool((value > 0).all())
    ):
        raise ValueError(f"Expected a finite positive scalar for {name}")
    return value.reshape(())


class ModelOptMixedWeight(CompositeWeight, QuantWeight):
    @classmethod
    def support(cls, quant_config, src_weight_info):
        return isinstance(quant_config, ModelOptMixedPrecisionConfig) and any(
            quant_config.recipe_for(w.name) is not None for w in src_weight_info.weights
        )

    def __init__(self, src_weight_info, quant_config, **kwargs):
        self.source = src_weight_info
        recipes = [quant_config.recipe_for(w.name) for w in src_weight_info.weights]
        if (
            not recipes
            or any(r is None for r in recipes)
            or any(r != recipes[0] for r in recipes)
        ):
            raise ValueError(
                f"Incompatible fused ModelOpt recipes for {src_weight_info.name}"
            )
        self.algo = recipes[0]["quant_algo"]
        self.delegate = None
        if self.algo == "NVFP4" and src_weight_info.name != W.lm_head:
            if src_weight_info.name not in PerGroupFp4Weight.w4a4_weight_list:
                raise ValueError(
                    f"Unsupported mixed NVFP4 weight {src_weight_info.name}"
                )
            self.delegate = PerGroupFp4Weight(
                src_weight_info,
                ModelOptFp4Config(bits=4, group_size=16, is_quanted=True),
                name=src_weight_info.name,
            )
            sub_weights = self.delegate.sub_weights
        elif self.algo == "NVFP4":
            head_prefix = src_weight_info.weights[0].name.removesuffix(".weight")
            sub_weights = {
                W.lm_head: CustomAtomicWeight(
                    W.lm_head,
                    src_weight_info.weights,
                    data_type=torch.uint8,
                    split_func=sp_0_pad8,
                ),
                W.lm_head_s: CustomAtomicWeight(
                    W.lm_head_s,
                    [CkptWeightInfo(head_prefix + ".weight_scale", identity)],
                    data_type=torch.float8_e4m3fn,
                    split_func=sp_0_pad8,
                ),
                W.lm_head_s2: CustomAtomicWeight(
                    W.lm_head_s2,
                    [CkptWeightInfo(head_prefix + ".weight_scale_2", identity)],
                    data_type=torch.float32,
                    split_func=sp_id,
                ),
                W.lm_head_i_s: CustomAtomicWeight(
                    W.lm_head_i_s,
                    [CkptWeightInfo(head_prefix + ".input_scale", identity)],
                    data_type=torch.float32,
                    split_func=sp_id,
                ),
            }
        else:
            if src_weight_info.name not in FP8_SCALES:
                raise ValueError(f"Unsupported mixed FP8 weight {src_weight_info.name}")
            kernel = copy.copy(src_weight_info)
            kernel.data_type = torch.float8_e4m3fn
            # Companion descriptors also expose every scale to checkpoint discovery.
            sub_weights = {kernel.name: kernel}
            for suffix in (".weight_scale", ".input_scale"):
                name = kernel.name + suffix
                sub_weights[name] = CustomAtomicWeight(
                    name,
                    [
                        CkptWeightInfo(
                            w.name.removesuffix(".weight") + suffix, identity
                        )
                        for w in src_weight_info.weights
                    ],
                    data_type=torch.float32,
                )
        super().__init__(
            sub_weights, name=src_weight_info.name, quant_config=quant_config
        )
        self.kernel = next(iter(sub_weights.values()))

    def _load_raw_tensor(self, tensor_source, layer_id, device, load_config):
        if self.algo == "NVFP4":
            # Taking max(global_scale) for two independently scaled projections
            # changes their decoded weights. Only fuse exactly equal scales.
            if isinstance(self.source, MoeAtomicWeight):
                name_groups = [
                    [
                        self.source._ckpt_name(w, layer_id, expert_id)
                        for w in self.source.weights
                    ]
                    for expert_id in load_config.get_selected_experts(
                        layer_id, self.source.config.expert_num
                    )
                ]
            else:
                name_groups = [[w.tensor_name(layer_id) for w in self.source.weights]]
            for names in name_groups:
                for suffix in (".weight_scale_2", ".input_scale"):
                    values = [
                        _scalar(
                            tensor_source, name.removesuffix(".weight") + suffix, device
                        )
                        for name in names
                    ]
                    if any(not torch.equal(v, values[0]) for v in values[1:]):
                        raise ValueError(f"Cannot fuse unequal NVFP4 {suffix}: {names}")
            return super()._load_raw_tensor(
                tensor_source, layer_id, device, load_config
            )

        matrices, weight_scales, input_scales = [], [], []
        for w in self.source.weights:
            name = w.tensor_name(layer_id)
            values = tensor_source.load_tensor(name, None)
            if any(v.dtype != torch.float8_e4m3fn for v in values):
                raise ValueError(f"Expected FP8 checkpoint tensor {name}")
            matrix = w.merge_fun([v.to(device) for v in values])
            matrices.append(matrix)
            prefix = name.removesuffix(".weight")
            # Run the original Q/gate split and GDN ordering on scale columns too.
            for dest, suffix in (
                (weight_scales, ".weight_scale"),
                (input_scales, ".input_scale"),
            ):
                scale = _scalar(tensor_source, prefix + suffix, device)
                dest.append(w.merge_fun([scale.expand(v.shape[0], 1) for v in values]))
        process = self.source.process_fun
        return {
            self.name: process(matrices).contiguous(),
            FP8_SCALES[self.name]: torch.cat(
                [process(weight_scales), process(input_scales)], dim=0
            ).contiguous(),
        }

    def _split(self, tensor, load_config):
        if self.algo == "NVFP4":
            if self.source.name == W.lm_head:
                head_config = load_config.model_copy(
                    update={
                        "tp_size": load_config.lm_head_tp_size,
                        "tp_rank": load_config.lm_head_tp_rank,
                    }
                )
                return super()._split(tensor, head_config)
            return super()._split(tensor, load_config)
        kernel = self.source._split(tensor[self.name], load_config)[self.name]
        scales = tensor[FP8_SCALES[self.name]]
        if self.name not in (W.attn_o_w, W.linear_attn_out_w):
            scales = self.source._split(scales, load_config)[self.name]
        return {self.name: kernel, FP8_SCALES[self.name]: scales}

    def _postprocess(self, tensor, device, load_config):
        if self.delegate is not None:
            return self.delegate._postprocess(tensor, device, load_config)
        if self.algo == "NVFP4":
            tensor[W.lm_head], tensor[W.lm_head_s] = (
                load_config.exported_device.convert_fp4_gemm_weight_params(
                    tensor[W.lm_head], tensor[W.lm_head_s]
                )
            )
            return tensor
        # Original descriptors produce KN; FP8 _scaled_mm consumes row-major NK.
        tensor[self.name] = tensor[self.name].T.contiguous()
        return tensor

    def _merge_lora(self, tensor, database, layer_id, load_config, lora_name=None):
        if load_config.merge_lora:
            raise ValueError(
                "Mixed ModelOpt checkpoints do not support load-time LoRA merging"
            )
        return tensor

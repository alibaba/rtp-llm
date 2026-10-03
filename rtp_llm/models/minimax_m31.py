"""MiniMax-M3.1 text model and checkpoint-specific weight handling."""

import json
import os
from typing import Any, Dict, List

import torch

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.model_factory_register import register_model
from rtp_llm.model_loader.ffn_weight import MoeAtomicWeight, MoeConfig, MoeWeight
from rtp_llm.model_loader.weight_module import CustomAtomicWeight
from rtp_llm.models.minimax_m3 import MiniMaxM3, MiniMaxM3Weight, _router_dtype
from rtp_llm.utils.model_weight import (
    CkptWeightInfo,
    W,
    identity,
    stack_,
    stack_moe_w1,
    transpose,
)

M31_RAW_ATTENTION_NORMS = {
    "minimax_m31.raw_q_norm": "q_norm",
    "minimax_m31.raw_k_norm": "k_norm",
    "minimax_m31.raw_index_q_norm": "index_q_norm",
    "minimax_m31.raw_index_k_norm": "index_k_norm",
}


def stack_nvfp4_w13_global_scales(ts: List[torch.Tensor]) -> torch.Tensor:
    """Stack per-expert up/gate NVFP4 inverse GSFs as ``[E, 2]``.

    The normal RTP MoE W1 contract is ``[up | gate]``.  Keep the global
    scales in the same order here; ``MegaMoeNvfp4Wrapper`` restacks both the
    packed weights and their GSFs to DeepGEMM's ``[gate | up]`` contract.
    """
    if len(ts) % 2 != 0:
        raise ValueError(f"NVFP4 W13 global scales require pairs, got {len(ts)}")
    half = len(ts) // 2
    up = torch.stack([t.reshape(()) for t in ts[:half]], dim=0)
    gate = torch.stack([t.reshape(()) for t in ts[half:]], dim=0)
    return torch.stack([up, gate], dim=1).contiguous()


class MiniMaxM31Weight(MiniMaxM3Weight):
    """M3.1 loader boundary for the released per-expert NVFP4 checkpoint."""

    def __init__(self, *args: Any, **kwargs: Any):
        self._prepacked_nvfp4_routed = False
        super().__init__(*args, **kwargs)

    def _process_meta(self, meta_dict, weight_keys):
        super()._process_meta(meta_dict, weight_keys)
        self._prepacked_nvfp4_routed = self._contains(
            weight_keys, ".block_sparse_moe.experts.0.w1.weight_packed"
        )
        expected = set(range(self._num_layers))
        q_layers = self._sparse_layer_set or set()
        k_layers = set()
        for key in weight_keys:
            if ".mtp." in key or ".self_attn.index_k_proj.weight" not in key:
                continue
            try:
                k_layers.add(int(key.split(".layers.")[1].split(".")[0]))
            except (IndexError, ValueError):
                continue
        if q_layers != expected or k_layers != expected:
            raise ValueError(
                "MiniMax-M3.1 checkpoint must contain index_q_proj and "
                "index_k_proj for every transformer layer; "
                f"missing_q={sorted(expected - q_layers)}, "
                f"missing_k={sorted(expected - k_layers)}, "
                f"unexpected_q={sorted(q_layers - expected)}, "
                f"unexpected_k={sorted(k_layers - expected)}"
            )

    def _should_load_msa_index(self, layer_id: int) -> bool:
        """M3.1 is sparse in every transformer layer by checkpoint contract."""
        sparse_set = self._sparse_layer_set or set()
        return layer_id in sparse_set

    def _get_hf_layer_weight_info(self, layer_id: int):
        layer_weights = super()._get_hf_layer_weight_info(layer_id)
        # Preserve legacy effective-gamma keys until attention is migrated.
        # The fused M3.1 producer needs original BF16 values: subtracting one
        # from a rounded gamma cannot recover the checkpoint weight.
        for key, checkpoint_name in M31_RAW_ATTENTION_NORMS.items():
            if checkpoint_name.startswith("index_"):
                if not self._should_load_msa_index(layer_id):
                    continue
            elif not self._use_qk_norm:
                continue
            layer_weights.append(
                CustomAtomicWeight(
                    key,
                    [
                        CkptWeightInfo(
                            self.prefix
                            + "model.layers.{i}.self_attn."
                            + checkpoint_name
                            + ".weight",
                            identity,
                        )
                    ],
                    identity,
                    data_type=torch.bfloat16,
                    disable_quantization=True,
                )
            )
        return layer_weights

    def _get_hf_ffn_layer_weight_info(self, layer_id: int):
        layer_weights = super()._get_hf_ffn_layer_weight_info(layer_id)
        if not self._prepacked_nvfp4_routed or layer_id not in self.moe_layer_index_:
            return layer_weights

        moe_config = MoeConfig(
            align_size=self._align_size,
            expert_num=self.expert_num_,
        )
        moe_root = self.prefix + "model.layers.{i}.block_sparse_moe."
        # Preserve the inherited shared expert and routing bias. Only replace
        # the routed MoE module, keeping packed values and both scale levels
        # intact until MegaMoeNvfp4Wrapper prepares the DeepGEMM layout.
        routed_weights = [
            MoeAtomicWeight(
                W.moe_gate,
                [CkptWeightInfo(moe_root + "gate.weight", identity)],
                transpose,
                data_type=_router_dtype(),
                config=moe_config,
            ),
            MoeAtomicWeight(
                W.moe_w2,
                [
                    CkptWeightInfo(
                        moe_root + "experts.{expert_id}.w2.weight_packed",
                        identity,
                    )
                ],
                stack_,
                data_type=torch.int8,
                config=moe_config,
                disable_quantization=True,
            ),
            MoeAtomicWeight(
                W.moe_s2,
                [
                    CkptWeightInfo(
                        moe_root + "experts.{expert_id}.w2.weight_scale",
                        identity,
                    )
                ],
                stack_,
                data_type=torch.float8_e4m3fn,
                config=moe_config,
                disable_quantization=True,
            ),
            MoeAtomicWeight(
                W.moe_w2_s2,
                [
                    CkptWeightInfo(
                        moe_root + "experts.{expert_id}.w2.weight_global_scale",
                        identity,
                    )
                ],
                stack_,
                data_type=torch.float32,
                config=moe_config,
                disable_quantization=True,
            ),
            # Standard RTP W1 order is [up(w3) | gate(w1)].
            MoeAtomicWeight(
                W.moe_w1,
                [
                    CkptWeightInfo(
                        moe_root + "experts.{expert_id}.w3.weight_packed",
                        identity,
                    ),
                    CkptWeightInfo(
                        moe_root + "experts.{expert_id}.w1.weight_packed",
                        identity,
                    ),
                ],
                stack_moe_w1,
                data_type=torch.int8,
                config=moe_config,
                disable_quantization=True,
            ),
            MoeAtomicWeight(
                W.moe_s1,
                [
                    CkptWeightInfo(
                        moe_root + "experts.{expert_id}.w3.weight_scale",
                        identity,
                    ),
                    CkptWeightInfo(
                        moe_root + "experts.{expert_id}.w1.weight_scale",
                        identity,
                    ),
                ],
                stack_moe_w1,
                data_type=torch.float8_e4m3fn,
                config=moe_config,
                disable_quantization=True,
            ),
            MoeAtomicWeight(
                W.moe_w1_s2,
                [
                    CkptWeightInfo(
                        moe_root + "experts.{expert_id}.w3.weight_global_scale",
                        identity,
                    ),
                    CkptWeightInfo(
                        moe_root + "experts.{expert_id}.w1.weight_global_scale",
                        identity,
                    ),
                ],
                stack_nvfp4_w13_global_scales,
                data_type=torch.float32,
                config=moe_config,
                disable_quantization=True,
            ),
        ]
        for index, module in enumerate(layer_weights):
            if isinstance(module, MoeWeight):
                layer_weights[index : index + 1] = routed_weights
                break
        return layer_weights


class MiniMaxM31(MiniMaxM3):
    """MiniMax-M3.1 text backbone, isolated from the legacy M3 runtime."""

    @classmethod
    def _create_config(cls, ckpt_path: str) -> ModelConfig:
        config = super()._create_config(ckpt_path)
        config.model_type = "minimax_m31"
        return config

    @classmethod
    def _from_hf(cls, config: ModelConfig, ckpt_path: str):
        super()._from_hf(config, ckpt_path)
        config_path = os.path.join(ckpt_path, "config.json")
        if not os.path.exists(config_path):
            return config
        with open(config_path) as reader:
            config_json = json.load(reader)
        cls._parse_nvfp4_config(config, config_json)
        return config

    @staticmethod
    def _parse_nvfp4_config(config: ModelConfig, config_json: Dict[str, Any]) -> None:
        quant_cfg = config_json.get("quantization_config", {})
        packed_nvfp4 = (
            str(quant_cfg.get("moe_quant_algo", "")).upper() == "NVFP4"
            and str(quant_cfg.get("moe_quant_format", "")).lower()
            == "nvfp4-pack-quantized"
        )
        config.prepacked_nvfp4_moe = bool(packed_nvfp4)

    def _create_python_model(self):
        from rtp_llm.models_py.model_desc.minimax_m31 import MiniMaxM31Model

        self.py_model = MiniMaxM31Model(
            self.model_config,
            self.parallelism_config,
            self.weight,
            self.moe_config,
            max_generate_batch_size=self.max_generate_batch_size,
            fmha_config=self.fmha_config,
            py_hw_kernel_config=self.hw_kernel_config,
            device_resource_config=self.device_resource_config,
        )
        return self.py_model

    @staticmethod
    def get_weight_cls():
        return MiniMaxM31Weight


register_model("minimax_m31", MiniMaxM31)


__all__ = ["MiniMaxM31", "MiniMaxM31Weight"]

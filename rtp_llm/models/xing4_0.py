import json
import logging
import os
from typing import List

import torch

from rtp_llm.model_factory_register import register_model
from rtp_llm.model_loader.weight_module import AtomicWeight, WeightModule
from rtp_llm.models.deepseek_v2 import (
    DeepSeekV2,
    DeepSeekV2Weight,
    DeepSeekV3Mtp,
    DeepSeekV3MtpWeight,
)
from rtp_llm.utils.model_weight import CkptWeightInfo, W, identity


class Xing4Weight(DeepSeekV2Weight):
    """Xing4.0 weight info: DeepSeekV3-family MLA/MoE/FP8 layout plus per-layer
    mHC hyper-connection tensors (attn_hc/ffn_hc hc_base/hc_fn/hc_scale).

    The mHC tensors reuse the v4_hc_* W slots: they are per-layer replicated
    (sp_id TP strategy) and absent from the FP8 per-block whitelist, so they
    stay unquantized exactly like the checkpoint's modules_to_not_convert.
    """

    def _get_hc_layer_weights(self) -> List[WeightModule]:
        weights: List[WeightModule] = []
        for tag in ("attn", "ffn"):
            for sub, w_name in (
                ("base", getattr(W, f"v4_hc_{tag}_base")),
                ("fn", getattr(W, f"v4_hc_{tag}_fn")),
                ("scale", getattr(W, f"v4_hc_{tag}_scale")),
            ):
                weights.append(
                    AtomicWeight(
                        w_name,
                        [CkptWeightInfo(f"model.layers.{{i}}.{tag}_hc.hc_{sub}", identity)],
                        identity,
                        data_type=torch.float32,
                    )
                )
        return weights

    def _get_hf_layer_weight_info(self, layer_id: int):
        layer_weights = super()._get_hf_layer_weight_info(layer_id)
        layer_weights.extend(self._get_hc_layer_weights())
        return layer_weights


class Xing4(DeepSeekV2):
    """XingChen-AGI Xing4.0-29B-A4B-FP8 (Xing4_0ForCausalLM).

    DeepSeek-V3-family backbone (MLA + noaux_tc MoE + blockwise FP8
    weight_scale_inv + yarn rope with interleaved layout) with DeepSeek-V4
    style matrix hyper-connections: 4 residual streams mixed per sublayer by
    attn_hc/ffn_hc, reduced by mean(dim=hc) before the final norm.
    """

    @classmethod
    def _create_config(cls, ckpt_path: str):
        config = super()._create_config(ckpt_path)
        config_path = os.path.join(ckpt_path, "config.json")
        if not os.path.exists(config_path):
            return config
        with open(config_path) as reader:
            config_json = json.loads(reader.read())
        # mHC fields (C++ ModelConfig.h already exposes hc_mult /
        # hc_sinkhorn_iters / hc_eps; the h_res clamp is python-only).
        config.hc_mult = int(config_json.get("hc_mult", 1))
        config.hc_sinkhorn_iters = int(config_json.get("hc_sinkhorn_iters", 0))
        config.hc_eps = float(config_json.get("hc_eps", 1e-6))
        config.mhc_h_res_clamp_min = float(
            config_json.get("mhc_h_res_clamp_min", -30.0)
        )
        config.mhc_h_res_clamp_max = float(
            config_json.get("mhc_h_res_clamp_max", 30.0)
        )
        # Xing4.0 checkpoints use "dtype" instead of "torch_dtype".
        if config.config_dtype is None:
            config.config_dtype = config_json.get("dtype")
        return config

    def support_cuda_graph(self) -> bool:
        return True

    @staticmethod
    def get_weight_cls():
        return Xing4Weight

    def _create_python_model(self):
        from rtp_llm.models_py.model_desc.xing4_0_model import Xing4Model

        model_config = self.model_config
        self.py_model = Xing4Model(
            model_config,
            self.parallelism_config,
            self.weight,
            self.moe_config,
            max_generate_batch_size=self.max_generate_batch_size,
            fmha_config=self.fmha_config,
            py_hw_kernel_config=self.hw_kernel_config,
            device_resource_config=self.device_resource_config,
        )


class Xing4MtpWeight(DeepSeekV3MtpWeight):
    """MTP (nextn) weights for Xing4.0 live inside the main checkpoint as
    ``model.layers.40.*`` while the V3-style MTP loader expects
    ``model.layers.0.*``; every CkptWeightInfo template is rewritten to the
    single nextn layer before the loader resolves keys."""

    mtp_ckpt_layer = 40

    def _remap_to_nextn_layer(self, module: WeightModule) -> None:
        target = f"model.layers.{self.mtp_ckpt_layer}."

        def _rewrite(info: CkptWeightInfo) -> None:
            # Layer weights use "model.layers.{i}." templates; the V3 MTP
            # global weights (embed_tokens / shared_head) use literal
            # "model.layers.0." names. The draft holds exactly one layer, so
            # every layer reference is the nextn layer.
            if "model.layers.{i}." in info.name:
                info.name = info.name.replace("model.layers.{i}.", target)
            elif info.name.startswith("model.layers.0."):
                info.name = target + info.name[len("model.layers.0.") :]

        stack: List[WeightModule] = [module]
        while stack:
            current = stack.pop()
            if current is None:
                continue
            for info in getattr(current, "weights", None) or []:
                _rewrite(info)
            subs = getattr(current, "sub_weights", None)
            if subs:
                # CompositeWeight children (FfnWeight/MoeWeight) keep
                # sub_weights as a dict keyed by W names.
                stack.extend(subs.values() if isinstance(subs, dict) else subs)
            lora_a = getattr(current, "lora_a", None)
            lora_b = getattr(current, "lora_b", None)
            if lora_a is not None:
                stack.append(lora_a)
            if lora_b is not None:
                stack.append(lora_b)

    def _get_weight_info(self):
        info = super()._get_weight_info()
        for layer_modules in info.layer_weights:
            for module in layer_modules:
                self._remap_to_nextn_layer(module)
        for module in info.weights:
            self._remap_to_nextn_layer(module)
        return info


class Xing4Mtp(DeepSeekV3Mtp):
    """Xing4.0 MTP draft: the single nextn layer (layer 40) of the shared
    checkpoint. Standard V3 nextn residual (no mHC); enorm/hnorm/eh_proj and
    shared_head weights are consumed by the C++ spec layer."""

    @classmethod
    def _create_config(cls, ckpt_path: str):
        config = super()._create_config(ckpt_path)
        # The checkpoint config.json declares the 40 backbone layers; the
        # draft model holds exactly the one nextn layer.
        config.num_layers = 1
        config.moe_layer_index = [0]
        # Xing4.0's eh_proj consumes [enorm(embed), hnorm(hidden)] (e-first).
        # The V3Mtp default reverse_e_h_norm=True swaps the concat and was
        # empirically rejected: offline draft-order diagnosis on the real
        # weights gives 7/16 next-token matches + correct first-generated-
        # token prediction with e-first and 0/16 with the reversed order
        # (RUN scripts/mtp_order_diag.py, reference/short_ref.pt).
        config.reverse_e_h_norm = False
        # The MTP block has no hyper-connections: keep hc fields at defaults so
        # C++/cuda-graph plumbing does not widen draft hidden states.
        config.hc_mult = 1
        config.hc_sinkhorn_iters = 0
        return config

    def _create_python_model(self):
        from rtp_llm.models_py.model_desc.xing4_0_model import Xing4MtpModel

        self.py_model = Xing4MtpModel(
            self.model_config,
            self.parallelism_config,
            self.weight,
            self.max_generate_batch_size,
            moe_config=self.moe_config,
            quant_config=self.model_config.quant_config,
            fmha_config=self.fmha_config,
            py_hw_kernel_config=self.hw_kernel_config,
            device_resource_config=self.device_resource_config,
        )

    @staticmethod
    def get_weight_cls():
        return Xing4MtpWeight


register_model("xing4_0", Xing4, ["Xing4_0ForCausalLM"])
register_model("xing4_0_mtp", Xing4Mtp, ["Xing4_0ForCausalLMNextN"])

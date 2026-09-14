"""V4.1 three-stage DSpark descriptors for the existing speculative engine."""

import os

import torch

from rtp_llm.config.dsv41_weights import build_v41_manifest
from rtp_llm.model_factory_register import register_model
from rtp_llm.model_loader.model_weight_info import ModelWeightInfo
from rtp_llm.model_loader.weight_module import AtomicWeight
from rtp_llm.models.deepseek_v41 import (
    DeepSeekV41,
    DeepSeekV41PrefillDraftWeight,
    V41AtomicWeight,
    _CHECKPOINT_DTYPES,
    _wo_a_bf16,
)
from rtp_llm.utils.model_weight import CkptWeightInfo, W, identity


class DeepSeekV41DSparkWeight(DeepSeekV41PrefillDraftWeight):
    @property
    def prefill_commit_only(self):
        return (
            str(getattr(self, "role_type", "")).upper().rsplit(".", 1)[-1] == "PREFILL"
        )

    def _get_weight_info(self):
        shared = super()._get_weight_info()
        if self.prefill_commit_only:
            return shared
        config = self.model_config.dsv41_config
        config.validate_parallelism(tp_size=self.tp_size, ep_size=self.ep_size)
        if self.tp_size != 1:
            raise ValueError("decode DSpark must use attention TP1")
        specs = build_v41_manifest(config)
        local_experts = config.text["dspark_n_routed_experts"] // self.ep_size
        layers = [[], [], []]
        globals_ = list(shared.weights)
        for name, spec in specs.items():
            if not name.startswith("mtp."):
                continue
            parts = name.split(".")
            stage, local = int(parts[1]), ".".join(parts[2:])
            if (
                local.startswith(("main_", "markov_head.", "confidence_head."))
                or local == "norm.weight"
            ):
                continue
            if ".ffn.experts." in name:
                expert = int(parts[4])
                if (
                    not self.ep_rank * local_experts
                    <= expert
                    < (self.ep_rank + 1) * local_experts
                ):
                    continue
            if spec.conversion == "dequantize_bf16" and name.endswith(".scale"):
                continue
            source, process, dtype = [spec], identity, _CHECKPOINT_DTYPES[spec.dtype]
            if spec.conversion == "dequantize_bf16":
                source.append(specs[name.removesuffix(".weight") + ".scale"])
                process, dtype = _wo_a_bf16, torch.bfloat16
            layers[stage].append(
                V41AtomicWeight("v41." + local, source, process, dtype)
            )
        for key, name in (
            (W.final_ln_gamma, "mtp.2.norm.weight"),
            (W.v4_dspark_markov_w1, "mtp.2.markov_head.embed.weight"),
            (W.v4_dspark_markov_w2, "mtp.2.markov_head.head.weight"),
        ):
            globals_.append(
                AtomicWeight(
                    key,
                    [CkptWeightInfo(name, identity)],
                    identity,
                    data_type=torch.bfloat16,
                )
            )
        return ModelWeightInfo(weights=globals_, layer_weights=layers)


class DeepSeekV41DSpark(DeepSeekV41):
    @classmethod
    def _create_config(cls, ckpt_path):
        config = super()._create_config(ckpt_path)
        config.model_type = "deepseek_v41_dspark"
        config.num_layers = 3
        config.is_mtp = True
        config.attn_config.layer_compress_ratios = [0, 0, 0]
        config.moe_layer_index = [0, 1, 2]
        config.expert_num = config.dsv41_config.text["dspark_n_routed_experts"]
        config.moe_k = config.dsv41_config.text["dspark_num_experts_per_tok"]
        return config

    @classmethod
    def speculative_weight_alias_names(cls, target_model, draft_model_config):
        if not isinstance(target_model, DeepSeekV41) or isinstance(
            target_model, DeepSeekV41DSpark
        ):
            raise TypeError("V4.1 DSpark requires a V4.1 target owner")
        target = target_model.model_config
        for name in (
            "vocab_size",
            "hidden_size",
            "data_type",
            "enable_fp32_lm_head",
            "dsv41_model_revision",
        ):
            if getattr(target, name) != getattr(draft_model_config, name):
                raise ValueError(f"V4.1 target/draft weight aliases disagree on {name}")
        return W.embedding, W.lm_head

    def init_multimodal(self, mm_model_config, vit_config, device):
        self.mm_part = None

    def load_mm_weight(self, model_config, ctype, tp_size, tp_rank, device):
        self.mm_part = None

    def support_cuda_graph(self):
        return True

    def _create_python_model(self):
        from rtp_llm.models_py.model_desc.deepseek_v41_dspark_model import (
            DeepSeekV41DSparkModel,
        )

        self.py_model = DeepSeekV41DSparkModel(
            self.model_config,
            self.parallelism_config,
            self.weight,
            kv_cache_config=self.kv_cache_config,
            max_tokens_per_rank=int(os.environ["DSV41_MAX_TOKENS_PER_RANK"]),
            max_generate_batch_size=self.max_generate_batch_size,
            fmha_config=self.fmha_config,
            py_hw_kernel_config=self.hw_kernel_config,
            device_resource_config=self.device_resource_config,
        )

    @staticmethod
    def get_weight_cls():
        return DeepSeekV41DSparkWeight


register_model(
    "deepseek_v41_dspark", DeepSeekV41DSpark, ["DeepseekV41DSparkDraftModel"]
)

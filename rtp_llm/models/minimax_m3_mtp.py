"""Native recurrent MTP draft model for MiniMax-M3."""

import json
import logging
import os
import re
from typing import Any, List

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.model_factory_register import register_model
from rtp_llm.model_loader.loader import ModelLoader
from rtp_llm.model_loader.model_weight_info import ModelWeightInfo
from rtp_llm.model_loader.weight_module import AtomicWeight, WeightModule
from rtp_llm.models.minimax_m3 import MiniMaxM3, MiniMaxM3Weight, add_unit_offset
from rtp_llm.ops import KvCacheDataType, SpeculativeType
from rtp_llm.utils.model_weight import CkptWeightInfo, W, identity, transpose


class MiniMaxM3MTPWeight(MiniMaxM3Weight):
    """Load one physical MTP module from a bundled MiniMax-M3 checkpoint."""

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self._mtp_root = "language_model.model.mtp.layers.0."

    def _process_meta(self, meta_dict, weight_keys):
        del meta_dict
        self.prefix = (
            "language_model."
            if any(name.startswith("language_model.") for name in weight_keys)
            else ""
        )
        mtp_modules = set()
        pattern = re.compile(r"(?:^|\.)model\.mtp\.layers\.(\d+)\.")
        for name in weight_keys:
            match = pattern.search(name)
            if match:
                mtp_modules.add(int(match.group(1)))
        if mtp_modules == {0}:
            self._mtp_root = self.prefix + "model.mtp.layers.0."
        else:
            raise ValueError(
                "MiniMax-M3 MTP requires exactly module 0 in the checkpoint; "
                f"found modules {sorted(mtp_modules)}"
            )
        eh_proj_prefix = self._mtp_root + "eh_proj"
        self._mtp_eh_proj_is_quantized = (
            eh_proj_prefix + ".weight_scale_inv" in weight_keys
        )
        self._sparse_layer_set = {0}
        self.has_e_score_correction_bias = self._contains(
            weight_keys,
            self._mtp_root
            + "transformer_layer.block_sparse_moe.e_score_correction_bias",
        )
        self._native_mxfp4_routed = self._contains(
            weight_keys,
            self._mtp_root
            + "transformer_layer.block_sparse_moe.experts.w13_weight",
        ) and self._contains(
            weight_keys,
            self._mtp_root
            + "transformer_layer.block_sparse_moe.experts.w2_weight",
        )

    def _should_load_msa_index(self, layer_id: int) -> bool:
        # The trained MTP block is always sparse and must never silently fall
        # back to dense attention.
        return layer_id == 0

    def _get_hf_layer_weight_info(self, layer_id: int):
        if layer_id != 0:
            raise ValueError(
                f"MiniMax-M3 MTP only has physical module 0, got {layer_id}"
            )
        layer_weights = super()._get_hf_layer_weight_info(layer_id)
        source_prefix = self.prefix + "model.layers.{i}."
        target_prefix = self._mtp_root + "transformer_layer."
        self._remap_checkpoint_prefixes(layer_weights, source_prefix, target_prefix)
        return layer_weights

    @staticmethod
    def _remap_checkpoint_prefixes(
        layer_weights: List[WeightModule], source_prefix: str, target_prefix: str
    ) -> None:
        for weight_module in layer_weights:
            for component in weight_module.get_components():
                checkpoint_weights = getattr(component, "weights", None)
                if not checkpoint_weights:
                    continue
                for checkpoint_weight in checkpoint_weights:
                    name = getattr(checkpoint_weight, "name", None)
                    if name and source_prefix in name:
                        checkpoint_weight.name = name.replace(
                            source_prefix, target_prefix
                        )

    def _get_weight_info(self):
        if self._num_layers != 1:
            raise ValueError(
                "MiniMax-M3 MTP runtime config must expose exactly one layer, "
                f"got {self._num_layers}"
            )
        weights = [
            AtomicWeight(
                W.embedding,
                [CkptWeightInfo(self.prefix + "model.embed_tokens.weight", identity)],
                identity,
            ),
            AtomicWeight(
                W.lm_head,
                [CkptWeightInfo(self.prefix + "lm_head.weight", identity)],
                identity,
            ),
            AtomicWeight(
                W.multi_tokens_predict_enorm,
                [CkptWeightInfo(self._mtp_root + "enorm.weight", identity)],
                add_unit_offset,
            ),
            AtomicWeight(
                W.multi_tokens_predict_hnorm,
                [CkptWeightInfo(self._mtp_root + "hnorm.weight", identity)],
                add_unit_offset,
            ),
            AtomicWeight(
                W.multi_tokens_predict_eh_proj,
                [CkptWeightInfo(self._mtp_root + "eh_proj.weight", identity)],
                transpose,
                # Released MiniMax-M3 MTP checkpoints store eh_proj in BF16
                # even though the transformer block is MXFP8. Do not let the
                # global quant config synthesize a required scale tensor when
                # the checkpoint deliberately has none. Debug checkpoints that
                # carry weight_scale_inv retain the MXFP8 path.
                disable_quantization=not self._mtp_eh_proj_is_quantized,
            ),
            AtomicWeight(
                W.multi_tokens_predict_final_ln_gamma,
                [CkptWeightInfo(self._mtp_root + "final_layernorm.weight", identity)],
                add_unit_offset,
            ),
        ]
        return ModelWeightInfo(
            layer_weights=[self._get_hf_layer_weight_info(0)], weights=weights
        )


class MiniMaxM3MTP(MiniMaxM3):
    """A single MiniMax-M3 MTP module reused for every proposal step."""

    def _reuse_target_weight(
        self, device: str, name: str, target_weight, draft_weight
    ) -> None:
        if (
            target_weight.shape != draft_weight.shape
            or target_weight.dtype != draft_weight.dtype
            or target_weight.device != draft_weight.device
        ):
            raise RuntimeError(
                f"MiniMax-M3 MTP target/draft {name} must have matching shape, "
                "dtype and device; "
                f"target={tuple(target_weight.shape)}/{target_weight.dtype}/"
                f"{target_weight.device}, "
                f"draft={tuple(draft_weight.shape)}/{draft_weight.dtype}/"
                f"{draft_weight.device}"
            )
        self.weight.set_global_weight(name, target_weight)
        logging.info(
            "MiniMax-M3 MTP reuses target %s on %s "
            "(target_data_ptr=%d, discarded_draft_data_ptr=%d)",
            name,
            device,
            target_weight.data_ptr(),
            draft_weight.data_ptr(),
        )

    def _bind_colocated_weights(self, device: str):
        from rtp_llm.models.minimax_m3 import _get_target_embedding, _get_target_lm_head

        target_embedding = _get_target_embedding(device)
        target_lm_head = _get_target_lm_head(device)
        draft_embedding = self.weight.get_global_weight(W.embedding)
        draft_lm_head = self.weight.get_global_weight(W.lm_head)
        self._reuse_target_weight(
            device, W.embedding, target_embedding, draft_embedding
        )
        self._reuse_target_weight(device, W.lm_head, target_lm_head, draft_lm_head)

        # Both replacements must succeed before releasing the placeholders.
        # Emptying the allocator cache here makes the reclaimed memory visible
        # to the C++ KV/cache allocations that run after model loading. This is
        # a one-time startup operation and never enters the inference path.
        del draft_embedding, draft_lm_head
        ModelLoader.force_clean_cuda_memory()

    @classmethod
    def _create_config(cls, ckpt_path: str) -> ModelConfig:
        config = super()._create_config(ckpt_path)
        config_path = os.path.join(ckpt_path, "config.json")
        with open(config_path) as reader:
            config_json = json.load(reader)
        text_config = config_json.get("text_config", config_json)
        num_mtp_modules = int(text_config.get("num_mtp_modules", 1))
        if num_mtp_modules != 1:
            raise ValueError(
                "MiniMax-M3 MTP currently requires num_mtp_modules=1, "
                f"got {num_mtp_modules}"
            )
        if config.msa_sparse_config is None:
            raise ValueError("MiniMax-M3 MTP requires sparse_attention_config")

        config.num_layers = 1
        config.moe_layer_index = [0]
        config.msa_sparse_config["sparse_layer_ids"] = [0]
        config.msa_sparse_config["disable_value_layer_ids"] = [0]
        config.attn_config.indexer_head_dim = int(
            config.msa_sparse_config["idx_head_dim"]
        )
        config.is_mtp = True
        config.physical_mtp_module_num = 1
        config.index_share_for_mtp_iteration = False
        config.model_type = "minimax_m3_mtp"
        config.enable_fp32_lm_head = False
        config.use_opaque_kv_cache_store = True
        return config

    @classmethod
    def configure_speculative_model(
        cls, sp_config, target_config: ModelConfig, draft_config: ModelConfig
    ) -> None:
        if sp_config.type != SpeculativeType.MTP:
            raise ValueError(
                "MiniMax-M3 native MTP requires SP_TYPE=mtp, "
                f"got {sp_config.type.name.lower()}"
            )
        draft_tokens = int(sp_config.gen_num_per_cycle)
        if draft_tokens < 1 or draft_tokens > 7:
            raise ValueError(
                "MiniMax-M3 native MTP supports 1..7 draft tokens per cycle, "
                f"got {draft_tokens}"
            )
        if target_config.hidden_size != draft_config.hidden_size:
            raise ValueError(
                "MiniMax-M3 target/draft hidden sizes must match: "
                f"{target_config.hidden_size} != {draft_config.hidden_size}"
            )
        if target_config.vocab_size != draft_config.vocab_size:
            raise ValueError(
                "MiniMax-M3 target/draft vocabulary sizes must match: "
                f"{target_config.vocab_size} != {draft_config.vocab_size}"
            )

        # Native MTP consumes the target's final pre-norm hidden state. Mark
        # that producer explicitly so CP prefill can keep it in rank-local
        # zigzag order instead of restoring and broadcasting a full-sequence
        # tensor before the draft pass.
        target_config._minimax_m3_target_hidden_state_layer_ids = (
            target_config.num_layers,
        )
        target_config.hc_mult = 1

    @staticmethod
    def _validate_kv_cache_dtype(model_config: ModelConfig) -> None:
        kv_cache_dtype = model_config.attn_config.kv_cache_dtype
        if kv_cache_dtype not in (KvCacheDataType.BASE, KvCacheDataType.FP8):
            raise ValueError(
                "MiniMax-M3 native MTP supports BF16 or FP8 draft KV cache; "
                f"got {kv_cache_dtype}. Set SP_FP8_KV_CACHE to 0 or 1."
            )
        logging.info(
            "MiniMax-M3 native MTP draft KV cache dtype: %s",
            kv_cache_dtype,
        )

    def _create_python_model(self):
        from rtp_llm.models_py.model_desc.minimax_m3_mtp import MiniMaxM3VLMTPModel

        self._validate_kv_cache_dtype(self.model_config)
        self.py_model = MiniMaxM3VLMTPModel(
            self.model_config,
            self.parallelism_config,
            self.weight,
            self.moe_config,
            max_generate_batch_size=self.max_generate_batch_size,
            fmha_config=self.fmha_config,
            py_hw_kernel_config=self.hw_kernel_config,
            device_resource_config=self.device_resource_config,
        )

    @staticmethod
    def get_weight_cls() -> type[MiniMaxM3MTPWeight]:
        return MiniMaxM3MTPWeight


register_model("minimax_m3_mtp", MiniMaxM3MTP, ["MiniMaxM3MTP"])


class MiniMaxM3VLMTP(MiniMaxM3MTP):
    """MiniMax-M3 VL MTP draft model without a local ViT instance."""

    @classmethod
    def _from_hf(cls, config: ModelConfig, ckpt_path: str):
        config_path = os.path.join(ckpt_path, "config.json")
        if not os.path.exists(config_path):
            return
        with open(config_path) as reader:
            config_json = json.load(reader)

        from rtp_llm.models.minimax_m3_vl import _apply_minimax_m3_vl_config

        _apply_minimax_m3_vl_config(config, config_json, ckpt_path)
        return config

    @classmethod
    def _create_config(cls, ckpt_path: str) -> ModelConfig:
        config = super()._create_config(ckpt_path)
        config.model_type = "minimax_m3_vl_mtp"
        return config

    def _create_python_model(self):
        from rtp_llm.models_py.model_desc.minimax_m3_mtp import MiniMaxM3MTPModel

        self._validate_kv_cache_dtype(self.model_config)
        self.py_model = MiniMaxM3MTPModel(
            self.model_config,
            self.parallelism_config,
            self.weight,
            self.moe_config,
            max_generate_batch_size=self.max_generate_batch_size,
            fmha_config=self.fmha_config,
            py_hw_kernel_config=self.hw_kernel_config,
            device_resource_config=self.device_resource_config,
        )


register_model("minimax_m3_vl_mtp", MiniMaxM3VLMTP, ["MiniMaxM3VLMTP"])

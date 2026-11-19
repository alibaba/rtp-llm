"""Qwen3 DSpark draft-model registration."""

from typing import Any, List

import torch

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.device.device_type import is_ppu
from rtp_llm.model_factory_register import register_model
from rtp_llm.model_loader.weight_module import AtomicWeight
from rtp_llm.models.qwen_v3 import QwenV3, QWenV3Weight
from rtp_llm.ops import KvCacheDataType, RoleType
from rtp_llm.utils.model_weight import (
    CkptWeightInfo,
    W,
    WeightStyle,
    identity,
    transpose,
)
from rtp_llm.utils.util import get_config_from_path


def dspark_offset_d2t_to_absolute(ts: List[torch.Tensor]) -> torch.Tensor:
    """Normalize Speculators' offset d2t table to absolute target token ids."""
    offsets = identity(ts).to(torch.int64)
    return offsets + torch.arange(
        offsets.numel(), dtype=offsets.dtype, device=offsets.device
    )


def _pack_commit_kv(tensors: List[torch.Tensor]) -> torch.Tensor:
    # [K0.T | V0.T | K1.T | V1.T | ...], the layout consumed by the fused linear.
    return torch.cat([weight.T for weight in tensors], dim=1)


def _native_prefill(config, role_type) -> bool:
    return config.dspark_layer_windows is not None and role_type == RoleType.PREFILL


class Qwen3DSparkWeight(QWenV3Weight):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._d2t_path = None
        self._hidden_norm_path = "model.hidden_norm.weight"

    def _process_meta(self, meta_dicts: Any, weight_keys: List[str]):
        super()._process_meta(meta_dicts, weight_keys)
        if "hidden_norm.weight" in weight_keys:
            self._hidden_norm_path = "hidden_norm.weight"
        if "d2t" in weight_keys:
            self._d2t_path = "d2t"
        elif "model.d2t" in weight_keys:
            self._d2t_path = "model.d2t"

    def get_weight_info(self, database=None):
        info = super().get_weight_info(database)
        if _native_prefill(self.model_config, self.role_type):
            # Dedicated prefill seeds context KV; it never runs proposal
            # attention, the FFN, the output norm or the Markov sampler.
            layers = {W.attn_qkv_w, W.k_ln_gamma}
            globals_ = {
                W.embedding,
                W.lm_head,
                W.dspark_fc_w,
                W.dspark_hidden_norm_gamma,
            }
            if self.tp_size == 1 and self.weight_style == WeightStyle.NONE:
                # Replicated attention in native CP prefill. Read just K/V from
                # the HF checkpoint, directly into the one cross-layer projection.
                # Split attention TP and converted checkpoints keep the loader's
                # own packed-QKV layout and slicing below in the model.
                layers.remove(W.attn_qkv_w)
                tables = [
                    CkptWeightInfo(
                        f"{self.transformer_prefix}layers.{i}.self_attn.{kv}_proj.weight",
                        identity,
                    )
                    for i in range(len(info.layer_weights))
                    for kv in ("k", "v")
                ]
                info.weights.append(
                    AtomicWeight(W.dspark_context_kv_w, tables, _pack_commit_kv)
                )
                globals_.add(W.dspark_context_kv_w)
            info.layer_weights = [
                [weight for weight in layer if weight.name in layers]
                for layer in info.layer_weights
            ]
            info.weights = [
                weight for weight in info.weights if weight.name in globals_
            ]
        return info

    def _get_weight_info(self):
        info = super()._get_weight_info()
        extras = (
            (W.dspark_fc_w, "fc.weight", transpose),
            (W.dspark_hidden_norm_gamma, self._hidden_norm_path, identity),
            (W.dspark_markov_w1, "markov_head.markov_w1.weight", identity),
            (W.dspark_markov_w2, "markov_head.markov_w2.weight", identity),
        )
        info.weights.extend(
            AtomicWeight(name, [CkptWeightInfo(path, identity)], transform)
            for name, path, transform in extras
        )
        if self._d2t_path is not None:
            transform = (
                dspark_offset_d2t_to_absolute if self._d2t_path == "d2t" else identity
            )
            info.weights.append(
                AtomicWeight(
                    W.multi_tokens_predict_d2t_map,
                    [CkptWeightInfo(self._d2t_path, identity)],
                    transform,
                    data_type=torch.int64,
                )
            )
        return info


class Qwen3DSpark(QwenV3):
    @classmethod
    def _apply_kv_cache_config(cls, model_config, kv_cache_config):
        super()._apply_kv_cache_config(model_config, kv_cache_config)
        if is_ppu() and model_config.dspark_layer_windows is not None:
            # GLM target KV can be FP8. The BF16 draft's paged FA3 uses the
            # projected BF16 keys/values directly for both commit and proposal.
            model_config.attn_config.kv_cache_dtype = KvCacheDataType.BASE

    @classmethod
    def speculative_weight_alias_names(cls, target_model, draft_model_config):
        if draft_model_config.dspark_layer_windows is None:
            return ()
        # The native Qwen3 DSpark contract embeds and projects through the
        # target, including when an exporter redundantly saved either matrix.
        # Speculators checkpoints keep their own weight descriptions above.
        target = target_model.model_config
        if target.vocab_size != draft_model_config.input_vocab_size:
            raise ValueError("cannot share DSpark embedding: input vocabulary differs")
        if (target.vocab_size, target.enable_fp32_lm_head) != (
            draft_model_config.vocab_size,
            draft_model_config.enable_fp32_lm_head,
        ):
            raise ValueError(
                "cannot share DSpark LM head: vocabulary or precision differs"
            )
        if (target.hidden_size, target.data_type) != (
            draft_model_config.hidden_size,
            draft_model_config.data_type,
        ):
            raise ValueError(
                "cannot share DSpark weights: hidden size or dtype differs"
            )
        return (W.embedding, W.lm_head)

    @classmethod
    def _create_config(cls, ckpt_path: str) -> ModelConfig:
        dspark = get_config_from_path(ckpt_path)
        assert dspark is not None, f"config.json missing under {ckpt_path}"
        backbone = dspark.get("transformer_layer_config", dspark)
        if not isinstance(backbone, dict):
            raise TypeError("transformer_layer_config must be an object")
        config = cls._create_config_from_json(ckpt_path, backbone)
        config.attn_config.is_causal = False
        config.input_vocab_size = int(backbone["vocab_size"])
        config.vocab_size = int(dspark.get("draft_vocab_size", backbone["vocab_size"]))
        native = "Qwen3DSparkModel" in dspark.get("architectures", ())
        if native:
            dflash = dspark["dflash_config"]
            markov = dspark.get("dspark_config", dflash)
            if markov.get("markov_head_type", "vanilla") != "vanilla":
                raise ValueError("only the vanilla DSpark Markov head is supported")
            target_layer_ids = [int(x) for x in dflash["target_layer_ids"]]
            types = dspark.get("layer_types", ["full_attention"] * config.num_layers)
            if len(types) != config.num_layers or set(types) - {
                "full_attention",
                "sliding_attention",
            }:
                raise ValueError(f"invalid native DSpark layer_types: {types}")
            window = int(dspark.get("sliding_window") or 0)
            if "sliding_attention" in types and window <= 0:
                raise ValueError(
                    "DSpark sliding attention requires a positive sliding_window"
                )
            # This native backend implements the GLM layer mask contract. Do not
            # silently run a different mask for a checkpoint with an override.
            is_causal = dspark.get("is_causal")
            if is_causal is not None and any(
                is_causal != (kind == "sliding_attention") for kind in types
            ):
                raise NotImplementedError(
                    "DSpark explicit is_causal conflicts with the supported per-layer masks"
                )
            config.dspark_layer_windows = tuple(
                window if t == "sliding_attention" else None for t in types
            )
        else:
            markov = dspark
            target_layer_ids = [int(x) for x in dspark["aux_hidden_state_layer_ids"]]
        config.dspark_noise_token_id = int(markov["mask_token_id"])
        if dspark.get("speculators_model_type") == "dspark":
            target_layer_ids = [x - 1 for x in target_layer_ids]
            if any(x < 0 for x in target_layer_ids):
                raise ValueError(
                    "Speculators DSpark aux_hidden_state_layer_ids must be positive"
                )
        config.dspark_target_layer_ids = target_layer_ids
        config.dspark_markov_rank = int(markov.get("markov_rank", 0) or 0)
        if config.dspark_markov_rank <= 0:
            raise ValueError("Qwen3 DSpark requires markov_rank > 0")
        config.dspark_sample_from_anchor = bool(
            dspark.get("sample_from_anchor", native)
        )
        return config

    @staticmethod
    def get_weight_cls():
        return Qwen3DSparkWeight

    def _create_python_model(self):
        from rtp_llm.models_py.model_desc.qwen3_dspark_model import (
            Qwen3DSparkModel,
        )

        self.py_model = Qwen3DSparkModel(
            self.model_config,
            self.parallelism_config,
            self.weight,
            max_generate_batch_size=self.max_generate_batch_size,
            quant_config=self.model_config.quant_config,
            fmha_config=self.fmha_config,
            py_hw_kernel_config=self.hw_kernel_config,
            commit_only=_native_prefill(
                self.model_config, self.parallelism_config.role_type
            ),
        )
        return self.py_model


register_model(
    "qwen_3_dspark", Qwen3DSpark, ["Qwen3DSparkForCausalLM", "Qwen3DSparkModel"]
)

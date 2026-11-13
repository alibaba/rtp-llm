import hashlib
import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Collection, Dict, List, Optional, Union

import torch

from rtp_llm.config.qwen4_exp_config import uses_fp8_ple_table
from rtp_llm.model_loader.ffn_weight import MoeAtomicWeight
from rtp_llm.model_loader.load_config import LoadConfig
from rtp_llm.model_loader.tensor_source import TensorSource
from rtp_llm.model_loader.weight_module import AtomicWeight, WeightModule
from rtp_llm.models.qwen3_next.qwen3_next_weight import Qwen35MoeWeight
from rtp_llm.ops import HybridAttentionType
from rtp_llm.utils.model_weight import CkptWeightInfo, W, identity, stack_, stack_moe_w1

# Verified against Qwen/Qwen3.8-Flash-Next model.safetensors.index.json; the
# bring-up status and revisit list live in
# docs/design/qwen3.8_flash_next_tp8_bringup_handoff.md.
_PREFIX_ANCHOR = "layers.0.attn_hyper_connection.hc_norm.weight"

_HC_UNIT_TAGS = {
    "attn_hyper_connection": (
        W.qwen4_hc_attn_norm,
        W.qwen4_hc_attn_mix_down,
        W.qwen4_hc_attn_mix_up,
        W.qwen4_hc_attn_inject,
    ),
    "mlp_hyper_connection": (
        W.qwen4_hc_mlp_norm,
        W.qwen4_hc_mlp_mix_down,
        W.qwen4_hc_mlp_mix_up,
        W.qwen4_hc_mlp_inject,
    ),
}

_INDEXER_WEIGHTS = (
    (W.qwen4_indexer_qk_proj_w, "indexer.index_qk_proj.weight"),
    (W.qwen4_indexer_q_ln_gamma, "indexer.q_layernorm.weight"),
    (W.qwen4_indexer_k_ln_gamma, "indexer.k_layernorm.weight"),
)

_PLE_WEIGHTS = (
    (W.qwen4_ple_conv_w, "conv1d.weight", None),
    (W.qwen4_ple_key_proj_w, "key_proj.weight", None),
    (W.qwen4_ple_value_proj_w, "value_proj.weight", None),
    (W.qwen4_ple_norm_conv_gamma, "norm_conv.weight", None),
    (W.qwen4_ple_norm_key_gamma, "norm_key.weight", None),
    (W.qwen4_ple_norm_query_gamma, "norm_query.weight", None),
    # These participate in integer modulo/XOR hashing. Letting AtomicWeight
    # default to compute_dtype would both destroy the 45-bit multipliers and
    # make torch.bitwise_xor reject them at runtime.
    (W.qwen4_ple_multipliers, "ple_embedding.layer_multipliers", torch.int64),
    (W.qwen4_ple_ngram_offsets, "ple_embedding.ngram_heads_offsets", torch.int64),
    (
        W.qwen4_ple_ngram_vocab_sizes,
        "ple_embedding.ngram_heads_vocab_sizes",
        torch.int64,
    ),
)

_PLE_NGRAM_SHARD_KEY = "ple_embedding.ngram_embedding.shard_{s}.weight"


@lru_cache(maxsize=1)
def _shared_ple_table(
    manifest_path: str, checkpoint_path: str
) -> tuple[torch.Tensor, dict]:
    manifest = json.loads(Path(manifest_path).read_text())
    root = Path(checkpoint_path)
    for field, name in (
        ("checkpoint_index_sha256", "model.safetensors.index.json"),
        ("checkpoint_config_sha256", "config.json"),
    ):
        if (
            manifest.get(field)
            != hashlib.sha256((root / name).read_bytes()).hexdigest()
        ):
            raise ValueError("Qwen4 shared PLE manifest does not match checkpoint")
    path = manifest["path"]
    table = torch.from_file(
        path, shared=True, size=os.stat(path).st_size, dtype=torch.uint8
    )
    table.untyped_storage()._qwen4_ple_shared = True
    return table, manifest["entries"]


class Qwen4ExpPleNgramWeight(AtomicWeight):
    """Load only this TP rank's PLE table shards, without concatenating them.

    The released table consists of 128 equal, row-contiguous checkpoint tensors.
    A plain :class:`AtomicWeight` would first materialize all 128 tensors and then
    call its process/split functions, which cannot fit in memory. This descriptor
    instead selects checkpoint *names* before loading. Each local shard remains a
    separate runtime tensor named ``<name>.<global shard index>`` so loading does
    not create a second table-sized stack/concat allocation.

    Shards always remain on the host. Projections, convolution weights and hash
    metadata retain their ordinary execution-device placement. The PLE consumer
    routes rank-local lookups and combines results across the TP group.
    """

    supports_fastsafetensors_iteration = False

    def __init__(
        self,
        name: str,
        weights: List[CkptWeightInfo],
        *,
        scale_weight: Optional[CkptWeightInfo] = None,
        preserve_fp8: bool = False,
        shared_ple_checkpoint: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        if "cpu_offload" in kwargs:
            raise TypeError("Qwen4 PLE table residency is fixed to CPU")
        super().__init__(name, weights, **kwargs)
        self.resident_device = "cpu"
        self.scale_weight = scale_weight
        self.preserve_fp8 = preserve_fp8
        self.shared_ple_checkpoint = shared_ple_checkpoint
        self.resident_output_devices = (
            {W.qwen4_ple_ngram_scale: "cpu"} if preserve_fp8 else {}
        )

    def _local_shard_range(self, load_config: LoadConfig) -> range:
        shard_count = len(self.weights)
        tp_size = load_config.tp_size
        tp_rank = load_config.tp_rank
        if tp_size < 1:
            raise ValueError(f"qwen4_exp PLE requires positive TP, got {tp_size}")
        if tp_rank < 0 or tp_rank >= tp_size:
            raise ValueError(
                f"qwen4_exp PLE tp_rank must be in [0, {tp_size}), got {tp_rank}"
            )
        if shard_count % tp_size != 0:
            raise ValueError(
                f"qwen4_exp PLE has {shard_count} checkpoint shards, which is not "
                f"divisible by tp_size={tp_size}"
            )
        shards_per_rank = shard_count // tp_size
        start = tp_rank * shards_per_rank
        return range(start, start + shards_per_rank)

    def get_tensor_names(
        self, layer_id: Optional[int], load_config: LoadConfig
    ) -> set[str]:
        names = {
            self.weights[index].tensor_name(layer_id)
            for index in self._local_shard_range(load_config)
        }
        if self.scale_weight is not None:
            names.add(self.scale_weight.tensor_name(layer_id))
        return names

    def _load_raw_tensor(
        self,
        tensor_source: TensorSource,
        layer_id: Optional[int],
        device: str,
        load_config: LoadConfig,
    ) -> Dict[str, torch.Tensor]:
        if layer_id is None:
            raise ValueError("qwen4_exp PLE n-gram table is a per-layer weight")
        convert_type = (
            torch.float8_e4m3fn
            if self.preserve_fp8
            else self.data_type or load_config.compute_dtype
        )
        scale = None
        if self.scale_weight is not None:
            scale = self.scale_weight.merge_fun(
                tensor_source.load_tensor(
                    self.scale_weight.tensor_name(layer_id), torch.float32
                )
            )
            if (
                scale.numel() != 1
                or not bool(torch.isfinite(scale).all())
                or not bool((scale > 0).all())
            ):
                raise ValueError(
                    "Qwen4 FP8 PLE requires one finite positive global scale"
                )
            if not self.preserve_fp8:
                scale = scale.to(device=device, dtype=convert_type)
        result: Dict[str, torch.Tensor] = {}
        if self.preserve_fp8:
            if scale is None:
                raise ValueError("FP8 PLE storage requires its checkpoint global scale")
            result[W.qwen4_ple_ngram_scale] = scale.to(device="cpu")
        shared = None
        if self.shared_ple_checkpoint is not None:
            shared = _shared_ple_table(
                os.environ["QWEN4_PLE_SHARED_TABLE"], self.shared_ple_checkpoint
            )
        for shard_index in self._local_shard_range(load_config):
            ckpt_weight = self.weights[shard_index]
            tensor_name = ckpt_weight.tensor_name(layer_id)
            if shared is None:
                shard = ckpt_weight.merge_fun(
                    tensor_source.load_tensor(tensor_name, convert_type)
                )
            else:
                table, entries = shared
                offset, length, shape = entries[tensor_name]
                shard = table.narrow(0, offset, length).view(convert_type).view(shape)
            if not isinstance(shard, torch.Tensor):
                raise TypeError(
                    f"qwen4_exp PLE shard {tensor_name!r} did not load as a tensor"
                )
            shard = shard.to(device=device, dtype=convert_type)
            # Match vLLM's lookup contract: cast FP8 values and the scalar to
            # output dtype, multiply, then retain the BF16 resident shard.
            # This preserves existing PLE gather and graph replay consumers.
            if scale is not None and not self.preserve_fp8:
                shard = shard * scale
            result[f"{self.name}.{shard_index}"] = shard
        return result

    def _split(
        self,
        tensor: Union[torch.Tensor, Dict[str, torch.Tensor]],
        load_config: LoadConfig,
    ) -> Dict[str, torch.Tensor]:
        if not isinstance(tensor, dict):
            raise TypeError("qwen4_exp PLE shard loader expected a tensor dictionary")
        return tensor

    def _postprocess(
        self,
        tensor: Union[torch.Tensor, Dict[str, torch.Tensor]],
        device: str,
        load_config: LoadConfig,
    ) -> Dict[str, torch.Tensor]:
        if not isinstance(tensor, dict):
            raise TypeError("qwen4_exp PLE shard loader expected a tensor dictionary")
        return tensor


class _PackedExpertWeight(MoeAtomicWeight):
    """Preserve checkpoint bytes and let the FP8/FP4 executor set its layout."""

    supports_fastsafetensors_iteration = False

    def _postprocess(self, tensor, device, load_config):
        raw = tensor[self.name] if isinstance(tensor, dict) else tensor
        dtype = (
            torch.float8_e8m0fnu if self.name in (W.moe_s1, W.moe_s2) else torch.int8
        )
        return {self.name: raw.view(dtype)}


class Qwen4ExpWeight(Qwen35MoeWeight):
    """Qwen3.8-Flash-Next weight loading.

    Reuses the Qwen3.5 mappings for GDN (split ``in_proj_qkv`` / ``in_proj_z`` /
    ``in_proj_b`` / ``in_proj_a``), stacked MoE experts, the shared expert and the
    full-attention projections -- ``q_proj`` likewise carries the output gate in
    its second half.

    The residual stream differs: there is no ``input_layernorm``,
    ``post_attention_layernorm`` or final ``norm.weight``. Each layer instead has
    ``attn_hyper_connection`` and ``mlp_hyper_connection`` units, and the model has
    a global ``hyper_connection_mixer`` whose ``hc_norm`` doubles as the final
    norm. All of those operate on an ``hc_mult * hidden_size`` wide stream.

    **No plain RMSNorm here takes ``plus_one``** -- not the ``hc_norm``s and not
    ``self_attn.{q,k}_norm``, which the Qwen3.5 base does fold. Upstream applies
    ``1.0 + weight`` in fp32, and folding it into the bf16 checkpoint tensor rounds
    the gain by up to ~4e-3 relative. The consuming modules add the one instead
    (``grouped_rms_norm`` for the wide stream, ``ZeroCenteredFusedQKRMSNorm`` for q/k),
    which keeps us bit-exact with upstream.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._moe_only_fp4 = bool(
            getattr(self.model_config, "_qwen4_full_fp8_nvfp4", False)
        )
        if self._moe_only_fp4 or getattr(
            self.model_config, "_qwen4_nvfp4_source", False
        ):
            self._align_size = 0

    def _create_moe_expert_weights(self, moe_config):
        if not getattr(self.model_config, "_qwen4_mxfp4_source", False):
            return super()._create_moe_expert_weights(moe_config)
        prefix = self.prefix + "layers.{i}.mlp.experts.{expert_id}."
        specs = (
            (W.moe_w1, ("gate", "up"), "weight", stack_moe_w1),
            (W.moe_s1, ("gate", "up"), "weight_scale", stack_moe_w1),
            (W.moe_w2, ("down",), "weight", stack_),
            (W.moe_s2, ("down",), "weight_scale", stack_),
        )
        return [
            _PackedExpertWeight(
                name,
                [
                    CkptWeightInfo(prefix + f"{projection}_proj.{suffix}")
                    for projection in projections
                ],
                process_fun=stack,
                config=moe_config,
                data_type=torch.uint8,
            )
            for name, projections, suffix, stack in specs
        ]

    def _process_meta(self, meta_dict: Any, weight_keys: Collection[str]):
        for key in weight_keys:
            if key.endswith(_PREFIX_ANCHOR) and "mtp." not in key:
                self.prefix = key[: -len(_PREFIX_ANCHOR)]
                break
        else:
            raise ValueError(
                f"Qwen4ExpWeight: cannot determine prefix, no non-mtp key ending "
                f"with {_PREFIX_ANCHOR!r} in {len(weight_keys)} ckpt keys"
            )
        if self._contains(weight_keys, "layers.0.mlp.experts.gate_up_proj"):
            self._has_stacked_ckpt = True

    def _create_mqa_weight(self) -> List[WeightModule]:
        """Reuse the Qwen3.5 attention mappings, minus the q/k norm ``plus_one``.

        Only those two entries are rebuilt so the qkv/o projection logic (output
        gate split, quantization variants) keeps coming from the base class.
        """
        raw_norms = {
            W.q_ln_gamma: "q_norm.weight",
            W.k_ln_gamma: "k_norm.weight",
        }
        weights = []
        for weight in super()._create_mqa_weight():
            source = raw_norms.pop(weight.name, None)
            if source is None:
                weights.append(weight)
            else:
                weights.append(
                    AtomicWeight(
                        weight.name,
                        [
                            CkptWeightInfo(
                                self.prefix + "layers.{i}.self_attn." + source
                            )
                        ],
                    )
                )
        if raw_norms:
            raise ValueError(
                f"Qwen4ExpWeight: base _create_mqa_weight no longer emits "
                f"{sorted(raw_norms)}; the plus_one strip is now a no-op"
            )
        return weights

    def _create_layer_norm_weight(self) -> List[WeightModule]:
        """The layer norms live inside the two gated residual units."""
        weights: List[WeightModule] = []
        for module, (norm, down, up, inject) in _HC_UNIT_TAGS.items():
            prefix = self.prefix + "layers.{i}." + module + "."
            weights.extend(
                [
                    AtomicWeight(
                        norm,
                        [CkptWeightInfo(prefix + "hc_norm.weight")],
                    ),
                    AtomicWeight(
                        down,
                        [CkptWeightInfo(prefix + "input_mix_weight_down.weight")],
                    ),
                    AtomicWeight(
                        up,
                        [CkptWeightInfo(prefix + "input_mix_weight_up.weight")],
                    ),
                    AtomicWeight(
                        inject,
                        [CkptWeightInfo(prefix + "block_inject_weight.weight")],
                    ),
                ]
            )
        return weights

    def _get_weight_info(self):
        """Base per-layer weights plus the qwen4-only indexer / PLE units.

        Both units exist only on *some* layers (indexer: full-attention layers;
        PLE: ``ple_layer_ids``), and the base builder hands every layer the same
        method set -- so the extras are appended per layer here.
        """
        info = super()._get_weight_info()
        self._append_indexer_weights(info.layer_weights)
        self._append_ple_weights(info.layer_weights)
        if getattr(self.model_config, "_qwen4_nvfp4_source", False):
            from rtp_llm.models.qwen4_exp.moe_nvfp4 import Qwen4Nvfp4WeightInfo

            return Qwen4Nvfp4WeightInfo(info.weights, info.layer_weights)
        if self._moe_only_fp4:
            from rtp_llm.models.qwen4_exp.moe_fp4 import Qwen4MoeFp4WeightInfo

            return Qwen4MoeFp4WeightInfo(
                info.weights,
                info.layer_weights,
                dense_storage=getattr(self.model_config, "_qwen4_dense_storage", None),
                vocab_storage=getattr(self.model_config, "_qwen4_vocab_storage", None),
            )
        return info

    def _append_indexer_weights(self, layer_weights: List[List[WeightModule]]) -> None:
        """Full-attention layers own a QSA indexer (fused qk proj + two norms)."""
        if not getattr(self.model_config, "enable_qwen4_qsa", False):
            return
        if getattr(self.model_config, "_qwen4_indexer_head_dim", 0) <= 0:
            return
        types = self.model_config.hybrid_attention_config.hybrid_attention_types
        for idx, weights in enumerate(layer_weights):
            if types[idx] == HybridAttentionType.LINEAR:
                continue
            weights.extend(
                AtomicWeight(
                    name,
                    [CkptWeightInfo(self.prefix + "layers.{i}.self_attn." + src)],
                )
                for name, src in _INDEXER_WEIGHTS
            )

    def _append_ple_weights(self, layer_weights: List[List[WeightModule]]) -> None:
        """Map checkpoint PLE layers through the rank-local host-table loader.

        A plain descriptor would materialize the 102.4 GB table per rank before
        TP splitting. The specialized loader selects source shards first.
        """
        ple_ids = getattr(self.model_config, "_qwen4_ple_layer_ids", [])
        if not ple_ids or not getattr(self.model_config, "enable_qwen4_ple", False):
            return
        shard_count = getattr(self.model_config, "_qwen4_split_ngram_parts", 0)
        if shard_count <= 0:
            raise ValueError(
                "qwen4_exp: PLE layers present but split_ngram_parts is not "
                "configured; cannot map the n-gram table shards"
            )
        preserve_fp8 = uses_fp8_ple_table(self.model_config)
        for layer_id in ple_ids:
            idx = layer_id - 1  # upstream ids are 1-based
            weights = layer_weights[idx]
            weights.extend(
                AtomicWeight(
                    name,
                    [CkptWeightInfo(self.prefix + "layers.{i}.ple." + src)],
                    data_type=data_type,
                )
                for name, src, data_type in _PLE_WEIGHTS
            )
            weights.append(
                Qwen4ExpPleNgramWeight(
                    W.qwen4_ple_ngram_shards,
                    [
                        CkptWeightInfo(
                            self.prefix
                            + "layers.{i}.ple."
                            + f"ple_embedding.ngram_embedding.shard_{shard}.weight"
                        )
                        for shard in range(shard_count)
                    ],
                    scale_weight=(
                        CkptWeightInfo(
                            self.prefix
                            + "layers.{i}.ple.ple_embedding.ngram_embedding.weight_scale"
                        )
                        if getattr(self.model_config, "_qwen4_fp8_source", False)
                        or preserve_fp8
                        else None
                    ),
                    preserve_fp8=preserve_fp8,
                    shared_ple_checkpoint=(
                        self.model_config.ckpt_path
                        if getattr(self.model_config, "_qwen4_mxfp4_source", False)
                        else None
                    ),
                )
            )

    def _create_global_weights(self) -> List[WeightModule]:
        mixer = self.prefix + "hyper_connection_mixer."
        return [
            AtomicWeight(
                W.embedding,
                [CkptWeightInfo(self.prefix + "embed_tokens.weight", identity)],
            ),
            AtomicWeight(W.lm_head, [CkptWeightInfo("lm_head.weight", identity)]),
            # The mixer's own norm is the model's final norm; there is no
            # separate `norm.weight` in this checkpoint. Stored raw, see the
            # class docstring on why this one skips `plus_one`.
            AtomicWeight(
                W.qwen4_hc_mixer_norm,
                [CkptWeightInfo(mixer + "hc_norm.weight")],
            ),
            AtomicWeight(
                W.qwen4_hc_mixer_mix_down,
                [CkptWeightInfo(mixer + "input_mix_weight_down.weight")],
            ),
            AtomicWeight(
                W.qwen4_hc_mixer_mix_up,
                [CkptWeightInfo(mixer + "input_mix_weight_up.weight")],
            ),
        ]

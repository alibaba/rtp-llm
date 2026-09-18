from typing import Any, Collection, Dict, List, Optional, Union

import torch

from rtp_llm.model_loader.load_config import LoadConfig
from rtp_llm.model_loader.tensor_source import TensorSource
from rtp_llm.model_loader.weight_module import AtomicWeight, WeightModule
from rtp_llm.models.qwen3_next.qwen3_next_weight import Qwen35MoeWeight
from rtp_llm.ops import HybridAttentionType
from rtp_llm.utils.model_weight import CkptWeightInfo, W, identity

# Verified against Qwen/Qwen3.8-Flash-Next model.safetensors.index.json; see
# docs/design/qwen3.8_flash_next_support_design.md appendix C for the full manifest.
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


class Qwen4ExpPleNgramWeight(AtomicWeight):
    """Load only this TP rank's PLE table shards, without concatenating them.

    The released table consists of 128 equal, row-contiguous checkpoint tensors.
    A plain :class:`AtomicWeight` would first materialize all 128 tensors and then
    call its process/split functions, which cannot fit in memory. This descriptor
    instead selects checkpoint *names* before loading. Each local shard remains a
    separate runtime tensor named ``<name>.<global shard index>`` so loading does
    not create a second table-sized stack/concat allocation.

    Hashed ids can target any rank, so the PLE consumer routes lookups and combines
    results across the TP group. The descriptor remains behind
    ``ModelConfig.enable_qwen4_ple`` (default false) because that correctness path
    deliberately rejects unsupported production modes.
    """

    supports_fastsafetensors_iteration = False

    def _local_shard_range(self, load_config: LoadConfig) -> range:
        shard_count = len(self.weights)
        tp_size = load_config.tp_size
        tp_rank = load_config.tp_rank
        if tp_size <= 1:
            raise ValueError(
                "qwen4_exp PLE requires TP > 1: refusing to load all "
                f"{shard_count} n-gram shards on one rank"
            )
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

    def local_shard_indices(self, load_config: LoadConfig) -> tuple[int, ...]:
        """Global checkpoint shard indices owned by the current TP rank."""
        return tuple(self._local_shard_range(load_config))

    def get_tensor_names(
        self, layer_id: Optional[int], load_config: LoadConfig
    ) -> set[str]:
        return {
            self.weights[index].tensor_name(layer_id)
            for index in self._local_shard_range(load_config)
        }

    def _load_raw_tensor(
        self,
        tensor_source: TensorSource,
        layer_id: Optional[int],
        device: str,
        load_config: LoadConfig,
    ) -> Dict[str, torch.Tensor]:
        if layer_id is None:
            raise ValueError("qwen4_exp PLE n-gram table is a per-layer weight")
        convert_type = self.data_type or load_config.compute_dtype
        result: Dict[str, torch.Tensor] = {}
        for shard_index in self._local_shard_range(load_config):
            ckpt_weight = self.weights[shard_index]
            tensor_name = ckpt_weight.tensor_name(layer_id)
            shard = ckpt_weight.merge_fun(
                tensor_source.load_tensor(tensor_name, convert_type)
            )
            if not isinstance(shard, torch.Tensor):
                raise TypeError(
                    f"qwen4_exp PLE shard {tensor_name!r} did not load as a tensor"
                )
            result[f"{self.name}.{shard_index}"] = shard.to(
                device=device, dtype=convert_type
            )
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
    (``grouped_rms_norm`` for the wide stream, ``Qwen4ExpFusedQKRMSNorm`` for q/k),
    which keeps us bit-exact with upstream.
    """

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
        """Add PLE weights only when its restricted correctness path is enabled.

        Keeping this opt-in is a memory-safety requirement, not just a feature
        switch: requesting the table with a plain descriptor materializes 102.4GB
        per rank before TP splitting. ``Qwen4ExpPleNgramWeight`` performs source
        selection first, and its consumer performs distributed rank-local lookups.
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

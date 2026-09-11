"""Full target and explicit prefill stages over bound V4.1 components.

Decode always uses the forty-layer path. The prefill executor owns the L20
tail and attention context; distributed scheduling remains caller-owned.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from rtp_llm.models_py.modules.dsv41.block import V41Block
from rtp_llm.models_py.modules.dsv41.engram import (
    Engram,
    EngramHash,
    build_compressed_token_map,
)
from rtp_llm.models_py.modules.dsv41.inputs import V41ModelRows
from rtp_llm.models_py.modules.dsv41.linear import V41Block32Linear
from rtp_llm.models_py.modules.dsv41.math import hc_pre, identity_pre_mix, rms_norm


@dataclass(frozen=True)
class V41ImageFeatures:
    row_indices: torch.Tensor
    token_types: torch.Tensor
    values: torch.Tensor

    def for_extend(self, start: int, end: int):
        """Rebase already computed global feature rows to a local LM extend."""
        if type(start) is not int or type(end) is not int or not 0 <= start <= end:
            raise ValueError("image feature extend requires ordered canonical bounds")
        selected = (self.row_indices >= start) & (self.row_indices < end)
        return V41ImageFeatures(
            (self.row_indices[selected] - start).contiguous(),
            self.token_types[selected].contiguous(),
            self.values[selected].contiguous(),
        )

    @classmethod
    def from_prepared(cls, vision, prepared):
        prepared.validate()
        positions, types, features = [], [], []
        previous_end = 0
        for image in prepared.images:
            end = image.start + image.length
            if image.start < previous_end or end > len(prepared.token_ids):
                raise ValueError(
                    "V4.1 image spans overlap or exceed the canonical rows"
                )
            if (
                any(
                    token != vision.processor_config.image_token_id
                    for token in prepared.token_ids[image.start : end]
                )
                or tuple(image.types.tolist())
                != prepared.token_types[image.start : end]
            ):
                raise ValueError("V4.1 prepared image IDs or delimiter types changed")
            positions.append(
                torch.arange(image.start, end, device=vision._device, dtype=torch.int64)
            )
            types.append(image.types.to(device=vision._device, dtype=torch.int32))
            features.append(vision.encode_image(image))
            previous_end = end
        if not positions:
            return cls(
                torch.empty(0, device=vision._device, dtype=torch.int64),
                torch.empty(0, device=vision._device, dtype=torch.int32),
                vision.image_start.new_empty((0, vision.image_start.numel())),
            )
        return cls(torch.cat(positions), torch.cat(types), torch.cat(features))

    def validate(self, rows: V41ModelRows, hidden_size: int):
        indices = self.row_indices
        if (
            indices.ndim != 1
            or indices.dtype != torch.int64
            or indices.device != rows.token_ids.device
            or not indices.is_contiguous()
            or self.token_types.shape != indices.shape
            or self.token_types.dtype != torch.int32
            or self.token_types.device != indices.device
            or self.values.shape != (indices.numel(), hidden_size)
            or self.values.dtype != torch.bfloat16
            or self.values.device != indices.device
        ):
            raise ValueError("V4.1 image features need matching row/type/BF16 metadata")
        torch._assert_async(
            ((indices >= 0) & (indices < rows.token_ids.numel())).all(),
            "image feature row outside the current target input",
        )
        torch._assert_async(
            (indices[1:] > indices[:-1]).all(),
            "image feature rows must be unique and in canonical order",
        )
        torch._assert_async(
            rows.image_mask.sum() == indices.numel(),
            "every image patch and delimiter must have exactly one feature row",
        )
        torch._assert_async(
            rows.image_mask[indices].all()
            & (rows.token_types[indices] == self.token_types).all(),
            "image features disagree with canonical token types",
        )


@dataclass(frozen=True)
class V41TargetOutput:
    hidden_states: torch.Tensor
    final_pre_mix: torch.Tensor
    aux_hidden_states: torch.Tensor | None
    aux_row_indices: torch.Tensor | None
    aux_layer_ids: tuple[int, ...]


@dataclass(frozen=True)
class V41L20Output:
    rows: V41ModelRows
    hidden_states: torch.Tensor
    pre_mix: torch.Tensor


class V41TargetModel(nn.Module):
    def __init__(
        self,
        config,
        embedding: torch.Tensor,
        norm: torch.Tensor,
        head: torch.Tensor,
        blocks: Sequence[V41Block],
        engrams: Mapping[int, Engram],
        token_hasher: nn.Module,
        *,
        vision=None,
        head_tp_size: int = 1,
        head_tp_rank: int = 0,
    ):
        super().__init__()
        self.config = config
        t = config.text
        self.hidden_size = t["hidden_size"]
        self.hc_mult = t["hc_mult"]
        if (
            type(head_tp_size) is not int
            or head_tp_size not in (1, 8)
            or type(head_tp_rank) is not int
            or not 0 <= head_tp_rank < head_tp_size
            or t["vocab_size"] % head_tp_size
        ):
            raise ValueError("V4.1 head partition must be explicit TP1 or CP8 metadata")
        self.head_tp_size = head_tp_size
        self.head_tp_rank = head_tp_rank
        self.head_vocab_size = t["vocab_size"] // head_tp_size
        self.head_vocab_start = head_tp_rank * self.head_vocab_size
        self.head_vocab_end = self.head_vocab_start + self.head_vocab_size
        if t["num_hidden_layers"] != 40 or len(blocks) != 40 or self.hc_mult != 4:
            raise ValueError(
                "V4.1 target execution requires all forty four-stream blocks"
            )
        if set(engrams) != {1, 14}:
            raise ValueError(
                "V4.1 target execution requires both L1/L14 Engram modules"
            )
        if any(module.layer_id != layer for layer, module in engrams.items()):
            raise ValueError("V4.1 Engram bindings use the wrong owner layers")
        if not all(isinstance(block, V41Block) for block in blocks):
            raise TypeError("V4.1 target execution requires its shifted-pre blocks")
        for name, tensor, shape, dtype in (
            (
                "embedding",
                embedding,
                (t["vocab_size"], self.hidden_size),
                torch.bfloat16,
            ),
            ("norm", norm, (self.hidden_size,), torch.bfloat16),
            ("head", head, (self.head_vocab_size, self.hidden_size), torch.float32),
        ):
            if (
                tensor.shape != shape
                or tensor.dtype != dtype
                or tensor.device != embedding.device
            ):
                raise ValueError(
                    f"V4.1 {name} shape/dtype/device differs from loaded weights"
                )
        self.register_buffer("embedding", embedding)
        self.register_buffer("norm", norm)
        self.register_buffer("head", head)
        self.blocks = nn.ModuleList(blocks)
        self.engrams = nn.ModuleDict(
            {str(layer): module for layer, module in engrams.items()}
        )
        self.token_hasher = token_hasher
        self.vision = vision
        self.aux_layer_ids = tuple(t["dspark_target_layer_ids"])
        if self.aux_layer_ids != (37, 38, 39):
            raise ValueError("V4.1 target auxiliary inputs must come from L37/L38/L39")

    @classmethod
    def from_model_weights(
        cls,
        config,
        weights,
        *,
        attention_factory,
        moe_factory,
        shared_lookup,
        tokenizer=None,
        token_hasher=None,
        vision=None,
        projection_factory=V41Block32Linear,
        head_tp_size: int = 1,
        head_tp_rank: int = 0,
    ):
        """Bind installed ModelWeights without reloading, converting or cloning them.

        Both factories receive (layer_id, checkpoint-local layer weights).
        The mapping removes only the loader's ``v41.`` prefix. MoE must
        provide the compatible quantized expert implementation explicitly;
        no legacy block128 or dequantized-expert implementation is selected.
        The caller owns the shared lookup's registration/Graph/close lifecycle.
        CP8 callers pass their actual head partition. The engine retains its
        existing last-hidden gather and TP logits gather after this model.
        """
        from rtp_llm.utils.model_weight import W

        if len(weights.weights) != 40:
            raise ValueError("installed V4.1 target weights must contain forty layers")
        embedding = weights.global_weights[W.embedding]
        if token_hasher is None:
            if tokenizer is None:
                raise ValueError(
                    "V4.1 target needs its canonical tokenizer or token hasher"
                )
            token_hasher = EngramHash(config, build_compressed_token_map(tokenizer))
        token_hasher = token_hasher.to(device=embedding.device)
        blocks, engrams = [], {}
        block_names = (
            "attn_norm.weight",
            "ffn_norm.weight",
            "hc_attn_fn",
            "hc_attn_base",
            "hc_attn_scale",
            "hc_ffn_fn",
            "hc_ffn_base",
            "hc_ffn_scale",
        )
        for layer, installed in enumerate(weights.weights):
            if any(not key.startswith("v41.") for key in installed):
                raise ValueError(
                    "V4.1 component initialization received another weight layout"
                )
            local = {
                key.removeprefix("v41."): value for key, value in installed.items()
            }
            if any(name.startswith("engram.embed.") for name in local):
                raise ValueError("Engram tables must remain in the host-shared loader")
            attention = attention_factory(layer, local)
            moe = moe_factory(layer, local)
            blocks.append(
                V41Block(attention, moe, {name: local[name] for name in block_names})
            )
            if layer in (1, 14):
                engrams[layer] = Engram.from_weights(
                    layer,
                    local,
                    shared_lookup,
                    projection_factory=projection_factory,
                )
        if vision is None and any(
            name.startswith("v41.vision.") for name in weights.global_weights
        ):
            from rtp_llm.models.multimodal.deepseek_v41_vision import (
                DeepSeekV41VisionEmbedding,
            )

            vision = DeepSeekV41VisionEmbedding.from_model_weights(
                config, weights.global_weights
            )
        return cls(
            config,
            embedding,
            weights.global_weights[W.final_ln_gamma],
            weights.global_weights[W.lm_head],
            blocks,
            engrams,
            token_hasher,
            vision=vision,
            head_tp_size=head_tp_size,
            head_tp_rank=head_tp_rank,
        )

    def prepare_images(self, prepared) -> V41ImageFeatures:
        if self.vision is None:
            raise RuntimeError("V4.1 vision weights have not been bound")
        return V41ImageFeatures.from_prepared(self.vision, prepared)

    def _embed_rows(self, rows, image_features):
        rows.validate()
        if rows.token_ids.device != self.embedding.device:
            raise ValueError(
                "V4.1 target rows and installed weights must share a device"
            )
        ids = rows.token_ids.masked_fill(~rows.valid, self.config.pad_token_id)
        embedded = F.embedding(ids, self.embedding)
        if image_features is None:
            torch._assert_async(
                ~rows.image_mask.any(), "V4.1 image rows require vision features"
            )
        else:
            image_features.validate(rows, self.hidden_size)
            embedded.index_copy_(0, image_features.row_indices, image_features.values)
        embedded.masked_fill_(~rows.valid[:, None], 0)
        hidden = embedded.unsqueeze(1).repeat(1, self.hc_mult, 1)
        return hidden, identity_pre_mix(hidden)

    def _run_blocks(
        self,
        rows,
        hidden,
        pre_mix,
        context,
        first,
        last,
        aux_row_indices=None,
        lookup_outputs=None,
    ):
        hashes = rows.engram_hashes(self.token_hasher) if first < 15 else None
        if hashes is not None and (
            hashes.shape != (rows.token_ids.numel(), 2, 24)
            or hashes.dtype != torch.int64
        ):
            raise ValueError(
                "V4.1 target hasher must provide both complete Engram head sets"
            )
        aux = []
        for layer in range(first, last):
            if layer in (1, 14):
                hidden = self.engrams[str(layer)](
                    hidden,
                    hashes[:, (0 if layer == 1 else 1), :].contiguous(),
                    rows.text_mask,
                    lookup_output=(
                        None if lookup_outputs is None else lookup_outputs[layer]
                    ),
                )
            if aux_row_indices is not None and layer in self.aux_layer_ids:
                # DSpark captures HC means at the block input, after Engram.
                aux.append(hidden.index_select(0, aux_row_indices).mean(dim=1))
            hidden, pre_mix = self.blocks[layer](
                hidden, pre_mix, context, rows.image_mask
            )
            hidden = hidden.masked_fill(~rows.valid[:, None, None], 0)
        return hidden, pre_mix, aux

    def _finish(self, hidden, pre_mix, aux, aux_row_indices):
        return V41TargetOutput(
            rms_norm(hc_pre(hidden, pre_mix), self.norm),
            pre_mix,
            torch.cat(aux, dim=-1) if aux_row_indices is not None else None,
            aux_row_indices.clone() if aux_row_indices is not None else None,
            self.aux_layer_ids if aux_row_indices is not None else (),
        )

    @torch.inference_mode()
    def prefill_encoder(
        self, rows, context, *, image_features=None, lookup_outputs=None
    ):
        """Materialize all new L0-L20 rows before selecting decoder consumers."""
        hidden, pre_mix = self._embed_rows(rows, image_features)
        hidden, pre_mix, _ = self._run_blocks(
            rows, hidden, pre_mix, context, 0, 21, lookup_outputs=lookup_outputs
        )
        return V41L20Output(rows, hidden, pre_mix)

    @torch.inference_mode()
    def prefill_decoder(self, l20: V41L20Output, context):
        """Execute only retained late rows; every returned aux row is computed."""
        l20.rows.validate()
        count = l20.rows.token_ids.numel()
        if (
            l20.hidden_states.shape != (count, self.hc_mult, self.hidden_size)
            or l20.hidden_states.dtype != torch.bfloat16
            or l20.hidden_states.device != self.embedding.device
            or l20.pre_mix.shape != (count, self.hc_mult)
            or l20.pre_mix.dtype != torch.float32
            or l20.pre_mix.device != self.embedding.device
        ):
            raise ValueError(
                "retained L20 HC/pre_mix has a different geometry or device"
            )
        torch._assert_async(
            l20.rows.valid.all(), "late prefill cannot consume padding aux"
        )
        selected = torch.arange(count, dtype=torch.int64, device=self.embedding.device)
        hidden, pre_mix, aux = self._run_blocks(
            l20.rows, l20.hidden_states, l20.pre_mix, context, 21, 40, selected
        )
        return self._finish(hidden, pre_mix, aux, selected)

    @torch.inference_mode()
    def forward(
        self,
        rows: V41ModelRows,
        context,
        *,
        execution_mode: str,
        image_features: V41ImageFeatures | None = None,
        aux_row_indices: torch.Tensor | None = None,
        lookup_outputs: Mapping[int, torch.Tensor] | None = None,
    ) -> V41TargetOutput:
        if execution_mode != "full":
            raise ValueError(
                "this target component only implements explicit full execution"
            )
        hidden, pre_mix = self._embed_rows(rows, image_features)
        if aux_row_indices is not None:
            indices = aux_row_indices
            if (
                indices.ndim != 1
                or indices.dtype != torch.int64
                or indices.device != rows.token_ids.device
                or not indices.is_contiguous()
            ):
                raise ValueError(
                    "V4.1 aux consumers need explicit contiguous int64 row indices"
                )
            torch._assert_async(
                ((indices >= 0) & (indices < rows.token_ids.numel())).all()
                & (indices[1:] > indices[:-1]).all(),
                "V4.1 aux rows must be unique, ordered and inside the current input",
            )
            torch._assert_async(
                rows.valid[indices].all(), "V4.1 aux rows cannot include padding"
            )
        hidden, pre_mix, aux = self._run_blocks(
            rows, hidden, pre_mix, context, 0, 40, aux_row_indices, lookup_outputs
        )
        return self._finish(hidden, pre_mix, aux, aux_row_indices)

    @torch.inference_mode()
    def local_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Compute this rank's FP32 vocabulary slice for the engine's TP gather."""
        if torch.is_autocast_enabled():
            raise RuntimeError("V4.1 FP32 logits require CUDA autocast off")
        if (
            hidden_states.ndim != 2
            or hidden_states.shape[1] != self.hidden_size
            or hidden_states.dtype != torch.bfloat16
            or hidden_states.device != self.head.device
        ):
            raise ValueError("V4.1 logits need the final normalized BF16 target hidden")
        return F.linear(hidden_states.float(), self.head)

    @torch.inference_mode()
    def logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.head_tp_size != 1:
            raise RuntimeError(
                "CP8 local logits require the engine's TP vocabulary gather before sampling"
            )
        return self.local_logits(hidden_states)

    @torch.inference_mode()
    def topk_logits(self, hidden_states: torch.Tensor, k: int, *, output_idx=None):
        """Return FP32 candidate logits without changing the downstream sampler."""
        from rtp_llm.models_py.modules.dsv41.deepselect import sampler_topk

        return sampler_topk(self.logits(hidden_states), k, output_idx=output_idx)

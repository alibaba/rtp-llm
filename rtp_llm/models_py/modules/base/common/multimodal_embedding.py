from typing import List, Optional, Sequence

import torch
from torch import nn

from rtp_llm.ops.compute_ops import PyModelInputs


# Keep this layout contract aligned with cpp/multimodal_processor/MultimodalInputUtils.h.
# Python consumes the flattened C++ representation after transport.
def reshape_extra_input_to_deepstack(
    extra_input: Sequence[torch.Tensor],
    multimodal_features: Sequence[torch.Tensor],
) -> List[torch.Tensor]:
    """Reshape flat 1-D extra-input tensors back into deepstack [layers, tokens, hidden].

    Each extra-input tensor is the flattened deepstack embedding for one image. Tokens and
    hidden are taken from the matching multimodal feature ([tokens, hidden]); the number of
    layers is derived from the element count. This is the model-specific inverse of the
    flatten done in the qwen3-vl producer.
    """
    deepstack: List[torch.Tensor] = []
    for flat, feature in zip(extra_input, multimodal_features):
        tokens = feature.size(0)
        hidden = feature.size(-1)
        layers = flat.numel() // (tokens * hidden)
        deepstack.append(flat.reshape(layers, tokens, hidden))
    return deepstack


class MultimodalEmbeddingInjector(nn.Module):
    """Insert multimodal features into the base embeddings at predefined offsets."""

    def forward(
        self,
        embeddings: torch.Tensor,
        multimodal_features: Sequence[torch.Tensor],
        multimodal_locs: torch.Tensor,
    ) -> torch.Tensor:
        if not multimodal_features:
            return embeddings

        if multimodal_locs.numel() != len(multimodal_features):
            raise ValueError(
                f"multimodal_locs has {multimodal_locs.numel()} entries "
                f"but {len(multimodal_features)} features were provided"
            )

        if embeddings.dim() != 2:
            raise ValueError(
                "embeddings must be a 2D tensor of shape [tokens, hidden_size]"
            )

        locs = multimodal_locs.to(device="cpu", dtype=torch.long).view(-1).tolist()

        hidden_size = embeddings.size(-1)
        for idx, (feature, loc) in enumerate(zip(multimodal_features, locs)):
            if feature is None or feature.numel() == 0:
                continue

            if feature.dim() != 2 or feature.size(-1) != hidden_size:
                raise ValueError(
                    f"feature[{idx}] must have shape [N, {hidden_size}], "
                    f"but got {feature.shape}"
                )

            if feature.dtype != embeddings.dtype:
                raise TypeError(
                    f"dtype mismatch: embeddings are {embeddings.dtype}, "
                    f"feature[{idx}] is {feature.dtype}"
                )

            if feature.device != embeddings.device:
                feature = feature.to(embeddings.device)

            if loc < 0:
                raise ValueError(f"feature[{idx}] loc must be non-negative, got {loc}")

            length = feature.size(0)
            if loc + length > embeddings.size(0):
                raise IndexError(
                    f"feature[{idx}] with length {length} cannot be placed at loc {loc} "
                    f"within embeddings of length {embeddings.size(0)}"
                )

            embeddings.narrow(0, loc, length).copy_(feature.contiguous())

        return embeddings


class MultimodalEmbedding(nn.Module):
    """Embed text tokens and insert multimodal features."""

    def __init__(self, embedding: nn.Module, *, supports_mask: bool = True):
        super().__init__()
        self.embedding = embedding
        self.injector = MultimodalEmbeddingInjector()
        self._lookup = self._native_lookup if supports_mask else self._custom_lookup

    @property
    def weight(self) -> torch.Tensor:
        return self.embedding.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        token_types: Optional[torch.Tensor] = None,
        text_tokens_mask: Optional[torch.Tensor] = None,
        *,
        model_inputs: Optional[PyModelInputs] = None,
        token_offset: int = 0,
    ) -> torch.Tensor:
        # Callers supplying only lookup arguments handle feature injection themselves.
        if model_inputs is None:
            return self._lookup(input_ids, position_ids, token_types, text_tokens_mask)
        mm_inputs = model_inputs.multimodal_inputs
        features = mm_inputs.multimodal_features
        if not features:
            return self._lookup(input_ids, position_ids, token_types, text_tokens_mask)

        mask = model_inputs.embedding_inputs.text_tokens_mask
        if mask is None or mask.numel() != model_inputs.input_ids.numel():
            raise ValueError(
                "multimodal embedding requires an aligned text_tokens_mask"
            )
        locs = mm_inputs.mm_features_locs
        if locs is None or locs.numel() != len(features):
            raise ValueError("multimodal feature/location count mismatch")

        token_end = token_offset + input_ids.numel()
        mask = mask.reshape(-1)[token_offset:token_end].to(input_ids.device)
        token_types = model_inputs.embedding_inputs.combo_tokens_type_ids
        if token_types is not None and token_types.numel():
            token_types = token_types.reshape(-1)[token_offset:token_end].to(
                input_ids.device
            )
        embeddings = self._lookup(input_ids, position_ids, token_types, mask)

        # Offsets are in the current (possibly CP-local) input, not the full prompt.
        if token_offset or input_ids.numel() != model_inputs.input_ids.numel():
            chunk_features = []
            chunk_locs = []
            for feature, loc in zip(features, locs.cpu().reshape(-1).tolist()):
                start = max(loc, token_offset)
                end = min(loc + feature.size(0), token_end)
                if start < end:
                    chunk_features.append(feature[start - loc : end - loc])
                    chunk_locs.append(start - token_offset)
            features = chunk_features
            locs = torch.tensor(chunk_locs, dtype=torch.int32, device="cpu")
        return self.injector(embeddings, features, locs)

    def _native_lookup(
        self,
        input_ids: torch.Tensor,
        position_ids: Optional[torch.Tensor],
        token_types: Optional[torch.Tensor],
        mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if position_ids is None and token_types is None and mask is None:
            return self.embedding(input_ids)
        return self.embedding(
            input_ids,
            position_ids=position_ids,
            token_types=token_types,
            text_tokens_mask=mask,
        )

    def _custom_lookup(
        self,
        input_ids: torch.Tensor,
        position_ids: Optional[torch.Tensor],
        token_types: Optional[torch.Tensor],
        mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if mask is None:
            return self.embedding(input_ids)
        # Mask lookup IDs without changing the cache hash IDs.
        text_rows = mask.reshape(input_ids.shape).bool()
        embeddings = self.embedding(input_ids.masked_fill(~text_rows, 0))
        return embeddings.masked_fill(~text_rows.reshape(-1, 1), 0)


class MultimodalDeepstackInjector(nn.Module):
    """Add per-layer multimodal deepstack embeddings into the hidden states."""

    def forward(
        self,
        hidden: torch.Tensor,
        mm_deepstack_embeds: Sequence[torch.Tensor],
        multimodal_locs: "torch.Tensor | Sequence[int]",
        layer_id: int,
    ) -> torch.Tensor:
        if not mm_deepstack_embeds or layer_id < 0:
            return hidden

        if isinstance(multimodal_locs, torch.Tensor):
            if multimodal_locs.numel() != len(mm_deepstack_embeds):
                raise ValueError(
                    f"multimodal_locs has {multimodal_locs.numel()} entries "
                    f"but {len(mm_deepstack_embeds)} deepstack tensors were provided"
                )
            locs = multimodal_locs.to(device="cpu", dtype=torch.long).view(-1).tolist()
        else:
            if len(multimodal_locs) != len(mm_deepstack_embeds):
                raise ValueError(
                    f"multimodal_locs has {len(multimodal_locs)} entries "
                    f"but {len(mm_deepstack_embeds)} deepstack tensors were provided"
                )
            locs = multimodal_locs
        hidden_size = hidden.size(-1)

        for idx, (stack, loc) in enumerate(zip(mm_deepstack_embeds, locs)):
            if stack.dim() != 3:
                raise ValueError(
                    f"deepstack tensor[{idx}] must have shape [layers, tokens, {hidden_size}], "
                    f"but got {stack.shape}"
                )

            if layer_id >= stack.size(0):
                continue

            layer_embed = stack[layer_id]
            if layer_embed.size(-1) != hidden_size:
                raise ValueError(
                    f"deepstack tensor[{idx}] hidden size mismatch: expected {hidden_size}, "
                    f"got {layer_embed.size(-1)}"
                )

            if layer_embed.dtype != hidden.dtype:
                raise TypeError(
                    f"dtype mismatch: hidden is {hidden.dtype}, "
                    f"deepstack tensor[{idx}] is {layer_embed.dtype}"
                )

            if layer_embed.device != hidden.device:
                layer_embed = layer_embed.to(hidden.device)

            if loc < 0:
                raise ValueError(
                    f"deepstack tensor[{idx}] loc must be non-negative, got {loc}"
                )

            length = layer_embed.size(0)
            if loc + length > hidden.size(0):
                raise IndexError(
                    f"deepstack tensor[{idx}] with length {length} cannot be placed at "
                    f"loc {loc} within hidden of length {hidden.size(0)}"
                )

            hidden_slice = hidden.narrow(0, loc, length)
            hidden_slice.add_(layer_embed.contiguous())

        return hidden

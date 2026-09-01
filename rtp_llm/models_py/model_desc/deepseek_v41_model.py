"""DeepSeek V4.1 Flash target model on the V4 CP/EP and paged-cache runtime."""

from __future__ import annotations

from typing import Any

import torch

from rtp_llm.models_py.model_desc.deepseek_v4_model import DeepSeekV4Model


class _V41ImageEmbedding(torch.nn.Module):
    """Embedding wrapper splicing prepared image rows before the mHC expansion.

    Wrapping the embed call (instead of overriding the hidden preparation hook)
    keeps the prefill fast path enabled for image-carrying requests.
    """

    def __init__(self, base, owner):
        super().__init__()
        self.base = base
        self.owner = owner

    @property
    def weight(self):
        return self.base.weight

    def forward(self, input):
        hidden = self.base(input)
        return self.owner._inject_image_rows(hidden, input)


def configure_v41(model, model_config) -> None:
    config = dict(getattr(model_config, "deepseek_v41_config", None) or {})
    if not config:
        raise ValueError("DeepSeek V4.1 requires deepseek_v41_config")
    model._v4_args.v41_config = config
    # V4.1 routers have no token-to-expert hash layers.
    model._v4_args.n_hash_layers = 0


class DeepSeekV41Model(DeepSeekV4Model):
    def __init__(self, model_config, *args, **kwargs) -> None:
        super().__init__(model_config, *args, **kwargs)
        configure_v41(self, model_config)
        self._engram_hash_state = None
        self._engram_layers = ()
        self._image_plan = None

    def cuda_graph_engram_window_size(self) -> int:
        return 4

    def _load_extra_weights(self, weights) -> None:
        from rtp_llm.models_py.modules.dsv4.engram import (
            Engram,
            EngramLayout,
            NgramHashState,
        )

        if not isinstance(self.v4.embed, _V41ImageEmbedding):
            self.v4.embed = _V41ImageEmbedding(self.v4.embed, self)
        config = self._v4_args.v41_config
        layout = EngramLayout(config)
        if not layout.layer_ids:
            return
        checkpoint_path = config["checkpoint_path"]
        modules = []
        device = self.v4.embed.weight.device
        self._engram_hash_state = NgramHashState(layout, checkpoint_path, device=device)
        for layer_id in layout.layer_ids:
            if layer_id >= len(self.v4.layers):
                continue
            layer = self.v4.layers[layer_id]
            layer.engram = Engram.from_checkpoint(
                config, layer_id, checkpoint_path, device
            )
            modules.append((layer_id, layout.layer_ids.index(layer_id)))
        self._engram_layers = tuple(modules)

    def _prepare_engram(self, inputs) -> None:
        if not self._engram_layers:
            return
        windows = getattr(inputs, "engram_token_windows", None)
        if windows is None or windows.numel() == 0:
            raise RuntimeError(
                "DeepSeek V4.1 Engram requires engine token windows with the "
                "current token and its three predecessors"
            )
        if windows.ndim != 2 or windows.shape[1] != 4:
            raise ValueError(f"Invalid Engram token windows shape: {windows.shape}")
        if windows.shape[0] != inputs.input_ids.numel():
            raise ValueError(
                "Engram token windows must follow the same CP split and padding "
                f"as input_ids: {windows.shape[0]} vs {inputs.input_ids.numel()}"
            )
        config = self._v4_args.v41_config
        windows = windows.to(device=self.v4.embed.weight.device, non_blocking=True)
        image_token = int(config.get("image_token_id", 129264))
        hashes = self._engram_hash_state.hash_token_windows(
            windows, dead_mask=windows == image_token
        )
        token_mask = (windows[:, 0] >= 0) & (windows[:, 0] != image_token)
        for layer_id, hash_index in self._engram_layers:
            layer = self.v4.layers[layer_id]
            layer.engram_hashes = hashes[:, hash_index].contiguous()
            layer.engram_token_mask = token_mask

    def _prepare_image_features(self, inputs) -> None:
        """Stash the gathered ViT features and their batch locations for the
        embedding wrapper. Under CP the locations stay in the pre-split batch
        coordinate space; the rank-local rows are derived in
        ``_image_row_indices`` from the CP shuffle metadata."""
        features = getattr(inputs, "multimodal_features", None)
        if not features:
            self._image_plan = None
            return
        locs = inputs.mm_features_locs
        if not locs.is_cuda:
            locs = locs.to(device=features[0].device, non_blocking=True)
        lengths = [int(feature.shape[0]) for feature in features]
        values = features[0] if len(features) == 1 else torch.cat(features)
        cp = getattr(inputs.attention_inputs, "context_parallel_info", None)
        spans = getattr(inputs, "mm_features_spans", None)
        if spans is not None and spans.numel():
            # The engine has already split features and locations to CP-local rows.
            cp = None
        self._image_plan = (values, locs, lengths, cp)

    def _global_image_rows(self, total, locs, lengths, device):
        rows = torch.full((total,), -1, dtype=torch.int64, device=device)
        offset = 0
        for loc, length in zip(locs.tolist(), lengths):
            if loc >= 0 and length > 0 and loc < total:
                visible = min(length, total - loc)
                rows[loc : loc + visible] = torch.arange(
                    offset, offset + visible, device=device
                )
            offset += length
        return rows

    def _image_row_indices(self, num_tokens, device):
        """Map each (rank-local) token to its row in the concatenated image
        features; -1 for text rows. Reuses are handled by clamping to the
        visible window, so image spans may straddle any reuse boundary."""
        values, locs, lengths, cp = self._image_plan
        shuffle = (
            getattr(cp, "prefill_shuffle_indices", None) if cp is not None else None
        )
        if shuffle is None or not int(shuffle.numel()):
            return self._global_image_rows(num_tokens, locs, lengths, device)
        actual = cp.prefill_actual_input_lengths_cpu.tolist()
        num_decode = num_tokens - int(shuffle.numel())
        prefill_lengths = actual[num_decode:]
        total = num_decode + sum(prefill_lengths)
        rows = self._global_image_rows(total, locs, lengths, device)
        chunk = cp.prefill_cp_chunk_lengths.to(device=device, dtype=torch.int64)
        shuffle = shuffle.to(device=device, dtype=torch.int64)
        boundaries = torch.cumsum(chunk, 0)
        local_index = torch.arange(int(shuffle.numel()), device=device)
        request = torch.searchsorted(boundaries, local_index, right=True)
        offsets = [num_decode]
        for length in prefill_lengths[:-1]:
            offsets.append(offsets[-1] + length)
        request_offsets = torch.tensor(offsets, device=device, dtype=torch.int64)
        positions = request_offsets[request] + shuffle.clamp(min=0)
        # Zigzag shuffle indices live in the per-request PADDED coordinate
        # space (the odd pair reads from the padded tail, so entries can
        # reach the padded length): entries at or beyond the request's
        # actual length are CP padding. They must stay text rows, or the
        # gather below would read out of bounds on the last request and
        # silently cross into the next request's span otherwise.
        request_lengths = torch.tensor(prefill_lengths, device=device, dtype=torch.int64)
        padding = (shuffle < 0) | (shuffle >= request_lengths[request])
        local_rows = torch.where(
            padding,
            torch.full_like(positions, -1),
            rows[positions.clamp(max=rows.numel() - 1)],
        )
        if num_decode:
            local_rows = torch.cat(
                (
                    torch.full((num_decode,), -1, dtype=torch.int64, device=device),
                    local_rows,
                )
            )
        return local_rows

    def _inject_image_rows(self, hidden, input_ids):
        if self._image_plan is None:
            return hidden
        rows = self._image_row_indices(int(input_ids.numel()), hidden.device)
        selected = rows >= 0
        if not bool(selected.any()):
            return hidden
        values = self._image_plan[0]
        hidden.view(input_ids.numel(), -1)[selected] = values[
            rows[selected].to(device=values.device)
        ]
        return hidden

    @torch.inference_mode()
    def forward(self, inputs, fmha_impl: Any = None):
        if self.kv_cache is not None:
            self._prepare_engram(inputs)
        self._prepare_image_features(inputs)
        try:
            return super().forward(inputs, fmha_impl)
        finally:
            # Request-local lookup rows must not keep whole prompt tensors live.
            for layer_id, _ in self._engram_layers:
                layer = self.v4.layers[layer_id]
                layer.engram_hashes = None
                layer.engram_token_mask = None
            self._image_plan = None


__all__ = ["DeepSeekV41Model"]

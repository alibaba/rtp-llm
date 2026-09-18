"""DeepSeek V4.1 Flash target model on the V4 CP/EP and paged-cache runtime."""

from __future__ import annotations

from typing import Any

import torch

from rtp_llm.models_py.model_desc.deepseek_v4_model import DeepSeekV4Model


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

    def cuda_graph_engram_window_size(self) -> int:
        return 4

    def _load_extra_weights(self, weights) -> None:
        from rtp_llm.models_py.modules.dsv4.engram import (
            Engram,
            EngramLayout,
            NgramHashState,
        )

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

    @torch.inference_mode()
    def forward(self, inputs, fmha_impl: Any = None):
        if self.kv_cache is not None:
            self._prepare_engram(inputs)
        try:
            return super().forward(inputs, fmha_impl)
        finally:
            # Request-local lookup rows must not keep whole prompt tensors live.
            for layer_id, _ in self._engram_layers:
                layer = self.v4.layers[layer_id]
                layer.engram_hashes = None
                layer.engram_token_mask = None


__all__ = ["DeepSeekV41Model"]

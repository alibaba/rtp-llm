"""DeepSeek-V4 predictor-based routed-expert Top-p selection.

The predictor is a separate, per-layer MLP trained to estimate the positive
norm of each routed expert.  Top-p is applied only to the six native routes;
the tensor keeps width six so existing MoE dispatch and CUDA graph contracts
remain unchanged.
"""

from __future__ import annotations

import json
import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

INVALID_EXPERT_ID = 256


def _first_env(*names: str) -> str:
    for name in names:
        value = os.environ.get(name, "").strip()
        if value:
            return value
    return ""


def dsv4_norm_predictor_dir() -> str:
    """Resolve the vLLM-compatible name plus RTP-local aliases."""

    return _first_env(
        "VLLM_DSV4_NORM_PREDICTOR_DIR",
        "RTP_DSV4_NORM_PREDICTOR_DIR",
        "DSV4_NORM_PREDICTOR_DIR",
    )


def dsv4_norm_top_p() -> Optional[float]:
    raw = _first_env(
        "VLLM_DSV4_NORM_TOP_P",
        "RTP_DSV4_NORM_TOP_P",
        "DSV4_NORM_TOP_P",
    )
    if not raw:
        return None
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"Invalid DSV4 norm Top-p value: {raw!r}") from exc
    if not 0.0 < value <= 1.0:
        raise ValueError(f"DSV4 norm Top-p must be in (0, 1], got {value}")
    return value


class NormPredictor(nn.Module):
    """Lazy loader for one ``norm_predictor_layer_{layer_id}.pt`` file."""

    def __init__(self, directory: str, layer_id: int, hidden_size: int = 4096):
        super().__init__()
        self.directory = Path(directory)
        self.layer_id = int(layer_id)
        self.hidden_size = int(hidden_size)
        self._weights = None
        self._device = None

    @property
    def path(self) -> Path:
        return self.directory / f"norm_predictor_layer_{self.layer_id}.pt"

    def _load(self, device: torch.device) -> None:
        path = self.path
        if not path.is_file():
            raise FileNotFoundError(
                f"DeepSeek-V4 norm predictor for layer {self.layer_id} not found: {path}"
            )
        try:
            state = torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:  # torch versions before weights_only
            state = torch.load(path, map_location="cpu")
        required = ("fc1.weight", "fc1.bias", "fc2.weight", "fc2.bias")
        missing = [key for key in required if key not in state]
        if missing:
            raise ValueError(f"Invalid norm predictor {path}: missing {missing}")
        fc1_w = state["fc1.weight"]
        fc1_b = state["fc1.bias"]
        fc2_w = state["fc2.weight"]
        fc2_b = state["fc2.bias"]
        if tuple(fc1_w.shape) != (1024, self.hidden_size):
            raise ValueError(
                f"Unexpected fc1 shape for layer {self.layer_id}: {tuple(fc1_w.shape)}"
            )
        if tuple(fc2_w.shape) != (256, 1024):
            raise ValueError(
                f"Unexpected fc2 shape for layer {self.layer_id}: {tuple(fc2_w.shape)}"
            )
        if tuple(fc1_b.shape) != (1024,) or tuple(fc2_b.shape) != (256,):
            raise ValueError(
                f"Unexpected bias shapes for layer {self.layer_id}: "
                f"fc1={tuple(fc1_b.shape)} fc2={tuple(fc2_b.shape)}"
            )
        self._weights = tuple(
            t.to(device=device, dtype=torch.float32)
            for t in (fc1_w, fc1_b, fc2_w, fc2_b)
        )
        self._device = device
        logger.info(
            "Loaded DeepSeek-V4 norm predictor: layer=%d path=%s",
            self.layer_id,
            path,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._weights is None or self._device != x.device:
            self.preload(x.device)
        fc1_w, fc1_b, fc2_w, fc2_b = self._weights
        hidden = F.linear(x.float(), fc1_w, fc1_b)
        hidden = F.relu(hidden)
        raw = F.linear(hidden, fc2_w, fc2_b)
        return F.softplus(raw) + 1.0e-6

    def preload(self, device: torch.device | str) -> None:
        device = torch.device(device)
        if self._weights is None or self._device != device:
            if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    "DeepSeek-V4 norm predictor was not loaded before CUDA graph capture; "
                    f"warm up layer {self.layer_id} first"
                )
            self._load(device)


def select_norm_top_p(
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    predicted_norms: torch.Tensor,
    top_p: float,
    *,
    invalid_expert_id: int = INVALID_EXPERT_ID,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Mask native routes using exclusive-prefix cumulative quality.

    The returned tensors retain the native ``[N, 6]`` shape.  Deleted slots
    have zero weight and ``invalid_expert_id``; retained weights are unchanged
    and are deliberately not renormalized.
    """

    if topk_weights.ndim != 2 or topk_ids.shape != topk_weights.shape:
        raise ValueError("top-k weights and ids must have the same rank-2 shape")
    if (
        predicted_norms.ndim != 2
        or predicted_norms.size(0) != topk_weights.size(0)
    ):
        raise ValueError("predicted norms must be [tokens, experts]")
    if not 0.0 < float(top_p) <= 1.0:
        raise ValueError(f"top_p must be in (0, 1], got {top_p}")
    if topk_weights.size(1) == 0:
        return topk_weights, topk_ids, topk_weights.new_zeros(
            (topk_weights.size(0),), dtype=torch.int64
        )

    gathered = predicted_norms.gather(
        1, topk_ids.clamp(min=0, max=predicted_norms.size(1) - 1)
    )
    contribution = torch.relu(topk_weights * gathered)
    total = contribution.sum(dim=-1, keepdim=True)
    quality = contribution / total.clamp_min(torch.finfo(contribution.dtype).eps)

    # A degenerate predictor row falls back to the native positive routing
    # weights, while preserving the original top-k weights in the output.
    native = torch.relu(topk_weights)
    native_total = native.sum(dim=-1, keepdim=True)
    native_quality = native / native_total.clamp_min(torch.finfo(native.dtype).eps)
    quality = torch.where(total > 0, quality, native_quality)

    sorted_quality, order = torch.sort(quality, dim=-1, descending=True, stable=True)
    previous = sorted_quality.cumsum(dim=-1) - sorted_quality
    keep_sorted = previous < float(top_p)
    keep_sorted[..., 0] = True
    keep = torch.zeros_like(keep_sorted).scatter(1, order, keep_sorted)

    kept_weights = topk_weights.masked_fill(~keep, 0)
    invalid = torch.full_like(topk_ids, int(invalid_expert_id))
    kept_ids = torch.where(keep, topk_ids, invalid)
    kept_count = keep.sum(dim=-1, dtype=torch.int64)
    return kept_weights, kept_ids, kept_count


class Dsv4NormTopP(nn.Module):
    """Apply predictor-based Top-p to one DeepSeek-V4 MoE layer."""

    def __init__(
        self,
        layer_id: int,
        dim: int,
        n_routed_experts: int,
        n_hash_layers: int = 3,
        model_type: str = "deepseek_v4",
    ):
        super().__init__()
        is_dsv4 = model_type == "deepseek_v4"
        directory = dsv4_norm_predictor_dir() if is_dsv4 else ""
        top_p = dsv4_norm_top_p() if is_dsv4 else None
        config = predictor_config(directory) if directory else {}
        if directory:
            validate_predictor_config(
                config,
                hidden_size=dim,
                n_routed_experts=n_routed_experts,
            )
        configured_layers = config.get("layers")
        layer_is_configured = (
            configured_layers is None or int(layer_id) in configured_layers
        )
        self.enabled = bool(
            is_dsv4
            and directory
            and top_p is not None
            and top_p < 1.0
            and layer_id >= n_hash_layers
            and layer_is_configured
        )
        self.layer_id = int(layer_id)
        self.top_p = float(top_p) if top_p is not None else 1.0
        self.n_routed_experts = int(n_routed_experts)
        self.predictor = (
            NormPredictor(directory, layer_id, hidden_size=dim) if self.enabled else None
        )
        self._audit_done = False

        if self.enabled and os.environ.get("RTP_DSV4_NORM_TOP_P_AUDIT", "0") == "1":
            logger.info(
                "Enabled DeepSeek-V4 norm Top-p: layer=%d top_p=%.4f predictor_dir=%s",
                layer_id,
                self.top_p,
                directory,
            )

    def forward(
        self,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.enabled:
            return topk_weights, topk_ids
        assert self.predictor is not None
        predicted = self.predictor(x)
        weights, ids, kept_count = select_norm_top_p(
            topk_weights,
            topk_ids,
            predicted,
            self.top_p,
            invalid_expert_id=self.n_routed_experts,
        )
        audit_enabled = os.environ.get("RTP_DSV4_NORM_TOP_P_AUDIT", "0") == "1"
        capturing = torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()
        if not self._audit_done and audit_enabled and not capturing:
            logger.info(
                "DeepSeek-V4 norm Top-p layer=%d first batch: tokens=%d "
                "avg_keep=%.4f min_keep=%d max_keep=%d",
                self.layer_id,
                kept_count.numel(),
                float(kept_count.float().mean().item())
                if kept_count.numel()
                else 0.0,
                int(kept_count.min().item()) if kept_count.numel() else 0,
                int(kept_count.max().item()) if kept_count.numel() else 0,
            )
            self._audit_done = True
        return weights, ids

    def preload(self, device: torch.device | str) -> None:
        if self.enabled:
            assert self.predictor is not None
            self.predictor.preload(device)


@lru_cache(maxsize=None)
def predictor_config(directory: str) -> dict:
    """Read the optional config for deployment diagnostics."""

    path = Path(directory) / "predictor_config.json"
    if not path.is_file():
        return {}
    with path.open() as handle:
        return json.load(handle)


def validate_predictor_config(
    config: dict,
    *,
    hidden_size: int,
    n_routed_experts: int,
) -> None:
    """Fail early when predictor metadata disagrees with the served model."""

    if not config:
        return
    expected = {
        "format": "dsv4-positive-norm-predictor-v3",
        "hidden_size": int(hidden_size),
        "bottleneck": 1024,
        "num_experts": int(n_routed_experts),
        "output_transform": "softplus",
    }
    mismatches = {
        key: (config.get(key), value)
        for key, value in expected.items()
        if config.get(key) != value
    }
    if mismatches:
        detail = ", ".join(
            f"{key}={actual!r} (expected {wanted!r})"
            for key, (actual, wanted) in mismatches.items()
        )
        raise ValueError(f"Incompatible DeepSeek-V4 predictor config: {detail}")
    layers = config.get("layers")
    if not isinstance(layers, list) or not all(isinstance(x, int) for x in layers):
        raise ValueError("DeepSeek-V4 predictor config 'layers' must be an integer list")

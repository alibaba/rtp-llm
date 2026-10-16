"""Explicit SM90 FlashInfer W4A8 candidate over the existing normal DeepEP.

Select with DSV4_MOE_STRATEGY=sm90_flashinfer. This does not change the default
Humming backend, attention parallelism, shared experts, dispatch, or combine.
The upstream numerical ordering differs from Humming (see local adapter).
"""

import logging
import os
from typing import Dict

import torch

from .._sm90_flashinfer_moe import Sm90FlashInferLocalExperts
from .base import MoeCfg, RoutedExpertsStrategy, register_strategy
from .deepep import DeepEPStrategy


def _forward_flashinfer_local(
    local, x, weights, indices, local_start, local_end, *, output_dtype=torch.float32
):
    """Backend-neutral local-compute callback for static normal DeepEP."""
    if local_start != 0 or local_end != local.num_experts:
        raise ValueError("static DeepEP callback requires local expert IDs")
    if not torch.cuda.is_current_stream_capturing():
        local.prepare(x.shape[0], indices.shape[1])
    return local.forward_local(
        x, weights, indices.to(torch.int64).contiguous(), output_dtype=output_dtype
    )


@register_strategy
class SM90FlashInferStrategy(RoutedExpertsStrategy):
    name = "sm90_flashinfer"

    @classmethod
    def can_handle(cls, cfg: MoeCfg) -> bool:
        return (
            torch.cuda.is_available()
            and torch.cuda.get_device_capability() == (9, 0)
            and cfg.ep_size > 0
            and cfg.n_routed_experts % cfg.ep_size == 0
            and cfg.n_local_experts == cfg.n_routed_experts // cfg.ep_size
            and cfg.local_expert_end - cfg.local_expert_start == cfg.n_local_experts
            and cfg.dim > 0
            and cfg.dim % 128 == 0
            and cfg.moe_inter_dim > 0
            and cfg.moe_inter_dim % 128 == 0
            and 1 <= cfg.n_activated_experts <= 8
            and cfg.swiglu_limit > 0
        )

    def __init__(self, cfg: MoeCfg):
        super().__init__(cfg)
        self._local = Sm90FlashInferLocalExperts(cfg)
        self._decode_graph = os.environ.get("DSV41_SM90_DECODE_GRAPH", "0") == "1"
        self._decode_graph_max_rows = int(
            os.environ.get("DSV41_SM90_DECODE_GRAPH_MAX_ROWS", "1")
        )
        if self._decode_graph_max_rows <= 0:
            raise ValueError("SM90 decode graph max rows must be positive")
        self._dispatch = None
        if cfg.ep_size > 1:
            self._dispatch = DeepEPStrategy(cfg)
            self._dispatch._local = self._local

    def setup_weights(self, layer_weights: Dict) -> None:
        self._local.setup_weights(layer_weights)
        logging.info(
            "V4.1 layer %d: explicit SM90 opensource FlashInfer CUTLASS W4A8; "
            "MXFP4/G32 weights, upstream activation quantization, "
            "route weights at GEMM2 finalize, BF16 local output; EP=%d",
            self.cfg.layer_id,
            self.cfg.ep_size,
        )

    def setup_runtime(self) -> None:
        if self.cfg.ep_size > 1:
            from rtp_llm.models_py.distributed.deepep_wrapper import (
                DeepEPMode,
                DeepEPWrapper,
            )

            wrapper = DeepEPWrapper._instance
            if wrapper is None or wrapper.mode != DeepEPMode.NORMAL:
                raise RuntimeError("SM90 FlashInfer requires DeepEP normal mode")
            if wrapper.config.ep_size != self.cfg.ep_size:
                raise RuntimeError("DeepEP and model EP sizes differ")
            if self._decode_graph and wrapper.config.use_deepep_internode:
                raise RuntimeError(
                    "SM90 FlashInfer decode graph requires intranode DeepEP"
                )
        if self._decode_graph:
            if self.cfg.ep_size > 1:
                rows = self._decode_graph_max_rows * self.cfg.ep_size
                topk = next(k for k in (2, 4, 8) if k >= self.cfg.n_activated_experts)
            else:
                rows = self._decode_graph_max_rows
                topk = self.cfg.n_activated_experts
            # Scratch/JIT/tuning are outside the weights VMM pool and graph.
            self._local.prepare(rows, topk, tune=True)

    def forward(self, x, weights, indices):
        if self._decode_graph and x.shape[0] <= self._decode_graph_max_rows:
            if self._dispatch is not None:
                from .._sm90_humming_graph import forward_normal_static

                return forward_normal_static(
                    self,
                    x,
                    weights,
                    indices,
                    self._decode_graph_max_rows,
                    local_forward=_forward_flashinfer_local,
                )
            # EP1 also captures smaller batches than the setup maximum.
            # Warm up their graph-owned scratch, separate from eager prefill.
            return _forward_flashinfer_local(
                self._local,
                x,
                weights,
                indices,
                self.cfg.local_expert_start,
                self.cfg.local_expert_end,
            )
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "SM90 FlashInfer capture requires DSV41_SM90_DECODE_GRAPH=1 "
                "and rows within DSV41_SM90_DECODE_GRAPH_MAX_ROWS"
            )
        if self._dispatch is not None:
            return self._dispatch(x, weights, indices)
        return self._local._forward_into_buf(
            x,
            weights,
            indices,
            self.cfg.local_expert_start,
            self.cfg.local_expert_end,
        )

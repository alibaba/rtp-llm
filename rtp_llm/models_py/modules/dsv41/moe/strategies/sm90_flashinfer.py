"""SM90 FlashInfer W4A8 with DeepEP V2 for prefill and decode.

This is the default SM90 routed-expert backend. Attention parallelism and
the independent Humming shared-expert path are unchanged.
The upstream numerical ordering differs from Humming (see local adapter).
"""

import logging
import os
from typing import Dict

import torch

from rtp_llm.device.device_type import is_cuda

from .._sm90_flashinfer_moe import Sm90FlashInferLocalExperts
from .base import MoeCfg, RoutedExpertsStrategy, register_strategy

_PREFILL_LOCAL_CHUNK_ROWS = 32768


def _forward_prefill_local(local, x, weights, indices):
    """BF16 local P output; input expert IDs are already rank-local.

    Slice only received rows, after dispatch and before combine. This bounds
    FI scratch without adding collectives or requiring equal rank loop counts.
    """
    if x.shape[0] <= _PREFILL_LOCAL_CHUNK_ROWS:
        return local._forward_into_buf(
            x.contiguous(),
            weights.contiguous(),
            indices.to(torch.int64).contiguous(),
            0,
            local.num_experts,
        ).to(x.dtype)
    output = torch.empty(x.shape, device=x.device, dtype=torch.bfloat16)
    for start in range(0, x.shape[0], _PREFILL_LOCAL_CHUNK_ROWS):
        end = min(start + _PREFILL_LOCAL_CHUNK_ROWS, x.shape[0])
        partial = local._forward_into_buf(
            x[start:end].contiguous(),
            weights[start:end].contiguous(),
            indices[start:end].to(torch.int64).contiguous(),
            0,
            local.num_experts,
        )
        # Preserve the existing FP32-to-BF16 boundary without allocating a
        # full-receive FP32 output and another full-receive conversion copy.
        output[start:end].copy_(partial)
        del partial
    return output


def _forward_flashinfer_local(
    local,
    x,
    weights,
    indices,
    local_start,
    local_end,
    *,
    output_dtype=torch.float32,
    recv_count=None
):
    """Local-compute callback for static DeepEP V2 receive capacity."""
    if local_start != 0 or local_end != local.num_experts:
        raise ValueError("static DeepEP callback requires local expert IDs")
    if not torch.cuda.is_current_stream_capturing():
        local.prepare(x.shape[0], indices.shape[1])
    return local.forward_local(
        x,
        weights,
        indices.to(torch.int64).contiguous(),
        output_dtype=output_dtype,
        recv_count=recv_count,
    )


@register_strategy
class SM90FlashInferStrategy(RoutedExpertsStrategy):
    name = "sm90_flashinfer"

    @classmethod
    def can_handle(cls, cfg: MoeCfg) -> bool:
        return (
            is_cuda()
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
        self._decode_graph = (
            cfg.is_decode_role and os.environ.get("DSV41_SM90_DECODE_GRAPH", "0") == "1"
        )
        self._decode_graph_max_rows = int(
            os.environ.get("DSV41_SM90_DECODE_GRAPH_MAX_ROWS", "1")
        )
        if self._decode_graph_max_rows <= 0:
            raise ValueError("SM90 decode graph max rows must be positive")
        # Keep a stale experimental AG/legacy setting from silently selecting
        # a different transport. P and D use the same V2 implementation.
        if os.environ.get("DSV41_SM90_MOE_COMM", "v2") != "v2":
            raise ValueError(
                "SM90 FlashInfer uses DeepEP V2; remove legacy/AG override"
            )
        self._v2_num_sms = int(os.environ.get("DSV41_SM90_V2_NUM_SMS", "0"))
        self._max_tokens_per_rank = (
            self._decode_graph_max_rows
            if self._decode_graph
            else cfg.max_tokens_per_rank
        )
        if self._max_tokens_per_rank <= 0:
            raise ValueError("SM90 V2 token capacity must be positive")
        self._comm_buffer = None

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
                raise RuntimeError(
                    "SM90 FlashInfer requires initialized DeepEP normal mode"
                )
            if wrapper.config.ep_size != self.cfg.ep_size:
                raise RuntimeError("DeepEP and model EP sizes differ")
            if wrapper.config.use_deepep_internode:
                raise RuntimeError("SM90 FlashInfer V2 currently requires intranode EP")
            group = wrapper.buffer.group
            if (
                group.size() != self.cfg.ep_size
                or group.rank() != self.cfg.ep_rank
                or self.cfg.local_expert_start
                != self.cfg.ep_rank * self.cfg.n_local_experts
                or self.cfg.local_expert_end
                != self.cfg.local_expert_start + self.cfg.n_local_experts
            ):
                raise ValueError("SM90 V2 requires aligned EP group and expert ranks")
            sm_count = torch.cuda.get_device_properties(
                torch.cuda.current_device()
            ).multi_processor_count
            if self._v2_num_sms and (
                self._v2_num_sms < 4
                or self._v2_num_sms % 2
                or self._v2_num_sms > sm_count
            ):
                raise ValueError(
                    "V2 SM count must be zero or a valid even device SM count"
                )
            from deep_ep import ElasticBuffer

            key = (
                self._max_tokens_per_rank,
                self.cfg.dim,
                8,
                self.cfg.n_routed_experts,
                self.cfg.ep_rank,
                torch.cuda.current_device(),
                id(group),
            )
            existing = getattr(wrapper, "_dsv41_elastic", None)
            if existing is None:
                buffer = ElasticBuffer(
                    group,
                    num_max_tokens_per_rank=key[0],
                    hidden=key[1],
                    num_topk=key[2],
                )
                wrapper._dsv41_elastic = (key, buffer)
                logging.info(
                    "SM90 DeepEP V2: role=%s capacity_per_rank=%d buffer_bytes=%d SMs=%d",
                    "decode" if self.cfg.is_decode_role else "prefill",
                    key[0],
                    buffer.num_bytes,
                    self._v2_num_sms,
                )
            elif existing[0] != key:
                raise ValueError("Incompatible SM90 ElasticBuffer capacity")
            self._comm_buffer = wrapper._dsv41_elastic[1]
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
        if self._comm_buffer is not None:
            if x.shape[0] > self._max_tokens_per_rank:
                raise ValueError("Input exceeds configured SM90 V2 token capacity")
            if self._decode_graph:
                from .._sm90_deepep_v2 import forward_elastic_static

                return forward_elastic_static(
                    self, x, weights, indices, local_forward=_forward_flashinfer_local
                )
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    "Dynamic SM90 V2 receive counts require eager execution"
                )
            from .._sm90_dispatch_pad import pad_deepep_routes

            indices_p, weights_p = pad_deepep_routes(indices, weights, pad_to=8)
            recv_x, recv_idx, recv_weights, handle, _ = self._comm_buffer.dispatch(
                x,
                topk_idx=indices_p,
                topk_weights=weights_p,
                num_experts=self.cfg.n_routed_experts,
                num_max_tokens_per_rank=self._max_tokens_per_rank,
                expert_alignment=1,
                num_sms=self._v2_num_sms,
                do_cpu_sync=True,
                do_expand=False,
            )
            # P uses actual receive rows and the single reusable eager scratch.
            # V2 expert IDs are already local; FI applies the weights once.
            if self.cfg.is_decode_role:
                local_output = self._local._forward_into_buf(
                    recv_x.contiguous(),
                    recv_weights.contiguous(),
                    recv_idx.to(torch.int64).contiguous(),
                    0,
                    self.cfg.n_local_experts,
                )
            else:
                local_output = _forward_prefill_local(
                    self._local, recv_x, recv_weights, recv_idx
                )
            combined, _, _ = self._comm_buffer.combine(local_output.to(x.dtype), handle)
            return combined.float()
        if self.cfg.ep_size > 1:
            raise RuntimeError("SM90 V2 runtime has not been initialized")
        if self._decode_graph and x.shape[0] <= self._decode_graph_max_rows:
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
        return self._local._forward_into_buf(
            x,
            weights,
            indices,
            self.cfg.local_expert_start,
            self.cfg.local_expert_end,
        )

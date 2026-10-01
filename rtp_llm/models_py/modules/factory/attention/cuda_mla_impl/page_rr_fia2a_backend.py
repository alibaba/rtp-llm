"""Packed Page-RR MLA with peer output or local reduction and all-to-all."""

import functools
import logging
import os

import cutlass
import cutlass.cute as cute
import torch
import torch.distributed as dist
import triton
import tvm_ffi

from rtp_llm.ops import DecodeCPMLAFusionMode, DecodeCPMLAA2ABackend

from ._fia2a.ops import _merge_splits_serial, _pack, merge_local_splits
from ._fia2a.mla_fused_fp8 import PageRRFusedMLAFP8
from ._fia2a.mla_fused_bf16 import PageRRFusedMLABF16
from ._fia2a.workspace import get_workspace


@functools.cache
def _compile_producer(heads, queries, world, capacity, page_size, fp32_wire, dtype="fp8"):
    cls = PageRRFusedMLAFP8 if dtype == "fp8" else PageRRFusedMLABF16
    config = dict(k_stages=2, v_stages=2) if dtype == "fp8" else {}
    kernel = cls(
        acc_dtype=cutlass.Float32, lse_dtype=cutlass.Float32,
        mma_qk_tiler_mn=(128, 128), mma_pv_tiler_mn=(128, 256),
        max_active_clusters=torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count // 2,
        page_size=page_size, skip_correction_threshold=0.0,
        is_persistent=True,
        peer_world=world, peer_capacity=capacity, peer_has_splits=fp32_wire,
        **config,
        num_heads=heads, seq_len_q=queries,
    )
    b, q, h = cute.sym_int(), cute.sym_int(), cute.sym_int()
    d, rope = cute.sym_int(divisibility=16), cute.sym_int(divisibility=16)
    pages, page, max_pages = cute.sym_int(), cute.sym_int(), cute.sym_int()
    wire_type = cutlass.Float32 if fp32_wire else cutlass.BFloat16
    input_type = cutlass.Float8E4M3FN if dtype == "fp8" else cutlass.BFloat16
    def strided(dtype, shape, align=16):
        return cute.runtime.make_fake_tensor(dtype, shape,
            stride=tuple(cute.sym_int() for _ in shape[:-1]) + (1,), assumed_align=align)
    args = [
        strided(input_type, (b,q,h,d)),
        strided(input_type, (b,q,h,rope)),
        strided(input_type, (pages,page,d)),
        strided(input_type, (pages,page,rope)),
        strided(cutlass.Int32, (b,max_pages), align=4),
        cute.runtime.make_fake_compact_tensor(wire_type, (b,q,h,d), stride_order=(3,2,1,0), assumed_align=16),
        cute.runtime.make_fake_compact_tensor(cutlass.Float32, (b,q,h), stride_order=(2,1,0), assumed_align=16),
        cutlass.Int32(1),
        cute.runtime.make_fake_tensor(cutlass.Int32, (b,), stride=(cute.sym_int(),), assumed_align=4),
        cute.runtime.make_fake_compact_tensor(cutlass.Int32, (cute.sym_int(),), assumed_align=16),
        cutlass.Float32(1.), cutlass.Float32(1.),
        cute.runtime.make_fake_compact_tensor(cutlass.Int64, (world,2), stride_order=(1,0), assumed_align=16),
        cutlass.Int32(0), tuple(cutlass.Int64(0) for _ in range(world)),
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
    ]
    return cute.compile(kernel, *args, options="--enable-tvm-ffi --opt-level 2")


class PageRRFIA2AMLABackend:
    def __init__(self, attn_configs, communicator, dtype, fusion_mode=DecodeCPMLAFusionMode.AUTO,
                 a2a_backend=DecodeCPMLAA2ABackend.AUTO):
        self.requested_mode = fusion_mode
        self.a2a_backend = a2a_backend
        self.communicator = communicator
        self.heads = communicator.heads
        self.page_size = attn_configs.kernel_tokens_per_block
        self.dtype = dtype
        if (
            not 1 <= self.heads <= 128
            or self.heads % communicator.size
            or attn_configs.kv_lora_rank != 512
            or attn_configs.rope_head_dim != 64
            or self.page_size not in (2, 4, 8, 16, 32, 64, 128)
            or dtype not in (torch.bfloat16, torch.float8_e4m3fn)
        ):
            raise ValueError("FIA2A requires H<=128 with equal head shards, L512/R64, P2..128, BF16 or E4M3")
        if torch.cuda.get_device_capability(communicator.device) != (10, 3):
            raise ValueError("FIA2A currently requires SM103")
        if not 1 < communicator.size <= 32:
            raise ValueError("FIA2A peer publication requires 2..32 TP ranks")
        self.sm_count = torch.cuda.get_device_properties(communicator.device).multi_processor_count
        self.workspace = get_workspace(communicator)
        self._shape = None

    def prepare(self, metadata, softmax_scale, output_scale):
        batch, queries = metadata.local_causal_lens.shape
        table = metadata.block_tables
        if table.shape[1] == 0:
            table = metadata.query_block_tables.view(batch, queries, -1)[:, 0]
        self.bounds = metadata.local_causal_lens
        self.table = table
        shape = (batch, queries, tuple(table.shape), table.stride(), softmax_scale, output_scale)
        if self._shape == shape:
            return
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("FIA2A must be warmed for its capture batch and query width")
        self.softmax_scale = softmax_scale
        self.output_scale = output_scale
        # A packed M128 query tile occupies a two-CTA cluster. Fill the
        # available SMs with query work before splitting KV; a graph bucket
        # keeps this geometry and communication path fixed across replays.
        query_ctas = batch * triton.cdiv(queries * self.heads, 128) * 2
        self.splits = min(32, max(1, self.sm_count // query_ctas))
        if self.requested_mode == DecodeCPMLAFusionMode.AUTO:
            self.fused = self.splits == 1
        elif self.requested_mode == DecodeCPMLAFusionMode.FUSED:
            self.fused = True
        elif self.requested_mode == DecodeCPMLAFusionMode.UNFUSED:
            self.fused = False
        else:
            raise ValueError("FIA2A fusion mode must be AUTO, FUSED or UNFUSED")
        self.mode = DecodeCPMLAFusionMode.FUSED if self.fused else DecodeCPMLAFusionMode.UNFUSED
        self.rows = batch * queries
        self.buffers = self.workspace.acquire(self.rows, self.splits, self.fused)
        self.a2a_buffers = None
        if self.a2a_backend == DecodeCPMLAA2ABackend.AUTO:
            # AUTO limits retained arenas to the measured logical wire range.
            # Explicit CUSTOM also permits small, unaligned, and larger wires.
            use_custom_a2a = (self.communicator.size in (4, 8)
                              and 768 <= self.rows * self.heads <= 49152
                              and (self.rows * self.communicator.local_heads) % 4 == 0)
        else:
            use_custom_a2a = self.a2a_backend == DecodeCPMLAA2ABackend.CUSTOM
        if not self.fused and use_custom_a2a:
            self.a2a_buffers = self.workspace.acquire_all_to_all(self.rows)
        self.query = torch.empty((batch, queries, self.heads, 576),
                                 device=self.communicator.device, dtype=self.dtype)
        # A single local destination specializes the same output epilogue into
        # local stores. Page-RR masks still use the original per-query bounds.
        self.output_world = self.communicator.size if self.fused else 1
        self.output_rank = self.communicator.rank if self.fused else 0
        self.merge_kernel = None
        if self.fused:
            self.output = torch.empty(
                (self.communicator.local_heads, self.rows, 512),
                device=self.communicator.device, dtype=torch.bfloat16,
            )
            self.merge_kernel = _merge_splits_serial
            self.merge_grid = (self.rows, 1, self.communicator.local_heads)
            self.merge_args = (
                self.buffers.output, self.buffers.lse, self.output,
                self.rows, self.communicator.local_heads, self.buffers.capacity,
                self.output_world, self.splits, 512,
                self.output.stride(1), self.output.stride(0), 512,
            )
        elif self.splits > 1:
            self.output = torch.empty((self.rows, self.heads, 512),
                                      device=self.communicator.device, dtype=torch.bfloat16)
            self.output_lse = torch.empty((self.rows, self.heads),
                                          device=self.communicator.device, dtype=torch.float32)
            self.merge_kernel = merge_local_splits
            merge_rows = 8 if self.splits <= 4 else 1
            self.merge_grid = ((self.rows, self.heads, 4) if merge_rows == 1
                               else (triton.cdiv(self.rows * self.heads, merge_rows), 4))
            self.merge_args = (
                self.buffers.output, self.buffers.lse, self.output, self.output_lse,
                self.rows, self.heads, self.splits,
                triton.next_power_of_2(self.splits), 128, merge_rows,
            )
        else:
            # S1 already has a normalized local context; no split reducer.
            self.output = self.buffers.output[0, :self.rows]
            self.output_lse = self.buffers.lse[0, :self.rows]
        self.out_descriptor = self.buffers.output.flatten()[:self.rows*self.heads*512].view(batch, queries, self.heads, 512)
        self.lse_descriptor = self.buffers.lse.flatten()[:self.rows*self.heads].view(batch, queries, self.heads)
        self.kernel = _compile_producer(
            self.heads, queries, self.output_world, self.buffers.capacity,
            self.page_size, self.splits > 1,
            "fp8" if self.dtype == torch.float8_e4m3fn else "bf16",
        )
        self.query_words = self.query.view(torch.int64)
        self.pack_words = 576 * self.query.element_size() // 8
        self.pack_grid = (triton.cdiv(self.rows * self.heads * self.pack_words, 1024),)

        signature = (*shape, self.page_size, self.splits, self.buffers.capacity, self.dtype, self.mode,
                     self.a2a_buffers is not None)
        if signature not in self.workspace.prepared:
            # warmup only compiles: the prepared query is a dtype/alignment
            # prototype for the future layer's Q and KV pointers. KV strides
            # are runtime arguments, so strided layer pools share this JIT.
            _pack.warmup(
                self.query_words, self.query_words, self.rows, self.heads,
                self.pack_words, BLOCK=1024, num_warps=4, grid=self.pack_grid,
            )
            if self.merge_kernel is not None:
                self.merge_kernel.warmup(*self.merge_args, num_warps=4, grid=self.merge_grid)
            if self.a2a_buffers is not None:
                self.a2a_buffers.warmup(self.rows)
            if self.fused or self.a2a_buffers is not None:
                # Finish all ranks' first-use compilation before any producer
                # can enter a peer wait. This is preparation, not a layer step.
                torch.cuda.synchronize()
                dist.barrier(group=self.communicator.group)
            self.workspace.prepared.add(signature)
            if os.environ.get("KIMI_K3_SMOKE_EVIDENCE", "0") == "1":
                logging.info(
                    "[FIA2A_PLAN] world_rank=%d TP=%d B=%d Q=%d H=%d S=%d mode=%s dtype=%s "
                    "requested_mode=%s wire_dtype=%s a2a_backend=%s a2a_transport=%s",
                    dist.get_rank(), self.communicator.size, batch, queries,
                    self.heads, self.splits, self.mode.name.lower(), self.dtype,
                    self.requested_mode.name, self.buffers.output.dtype, self.a2a_backend.name,
                    "producer" if self.fused else "custom" if self.a2a_buffers is not None else "nccl",
                )
        self._shape = shape

    def forward(self, gathered, kv):
        query = self.query
        _pack[self.pack_grid](
            gathered.view(torch.int64), self.query_words, self.rows, self.heads,
            self.pack_words, BLOCK=1024, num_warps=4,
        )
        with tvm_ffi.use_torch_stream():
            self.kernel(
                query[..., :512], query[..., 512:], kv[..., :512], kv[..., 512:],
                self.table, self.out_descriptor, self.lse_descriptor, cutlass.Int32(self.splits),
                self.bounds[:, -1], self.bounds.view(-1),
                cutlass.Float32(self.softmax_scale), cutlass.Float32(self.output_scale),
                self.buffers.pointers, cutlass.Int32(self.output_rank), self.buffers.output_ptrs,
            )
        if self.fused:
            # Stream ordering completes all producer stores before publication.
            self.buffers.synchronize_peers()
            self.merge_kernel[self.merge_grid](*self.merge_args, num_warps=4)
            # Every receiver has finished before any layer/step reuses storage.
            self.buffers.synchronize_peers()
            return self.output
        else:
            if self.splits > 1:
                self.merge_kernel[self.merge_grid](*self.merge_args, num_warps=4)
            return self.communicator.combine(
                self.output, self.output_lse, self.bounds, a2a_buffers=self.a2a_buffers,
            )

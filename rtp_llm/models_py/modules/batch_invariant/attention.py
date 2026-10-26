"""One attention arithmetic path shared by prefill, chunks and decode."""

from types import SimpleNamespace

import torch

from rtp_llm.models_py.modules.factory.attention import common
from rtp_llm.models_py.modules.factory.attention.cuda_impl.flashinfer_rotary_emb import (
    MhaRotaryEmbeddingOp,
)
from rtp_llm.models_py.modules.factory.attention.cuda_impl.kv_cache_write_op import (
    KVCacheWriteOp,
)
from rtp_llm.models_py.modules.factory.attention.fmha_impl_base import FMHAImplBase
from rtp_llm.ops.compute_ops import rtp_llm_ops


class BatchInvariantAttention(FMHAImplBase):
    def __init__(self, configs, inputs, *, warmup=False):
        self.configs = configs
        self.warmup = warmup
        self.warmup_cache = None
        self.params = rtp_llm_ops.FlashInferMlaAttnParams()
        self.rope = MhaRotaryEmbeddingOp(configs)
        if self.rope.cos_sin_cache is None:
            raise ValueError(
                "batch invariant requires the fixed RoPE cache; dynamic fallback is unsupported"
            )
        self.rope.set_params(self.params)
        self.writer = KVCacheWriteOp(
            configs.kv_head_num, configs.size_per_head, configs.kernel_tokens_per_block
        )
        self.writer.set_params(self.params)
        self.block_table = None
        self.query_start = None
        self.prepare(inputs)

    def prepare(self, inputs, replay=False):
        table = inputs.kv_cache_kernel_block_id
        if table is None or table.numel() == 0:
            table = inputs.kv_cache_kernel_block_id_device
        if self.warmup:
            if replay:
                raise ValueError(
                    "batch invariant Graph replay requires a real KV cache"
                )
            lengths = inputs.input_lengths.cpu()
            if inputs.is_prefill:
                lengths = lengths + inputs.prefix_lengths.cpu()
            else:
                lengths = inputs.sequence_lengths.cpu() + 1
            width = max(
                1,
                (int(lengths.max()) + self.configs.kernel_tokens_per_block - 1)
                // self.configs.kernel_tokens_per_block,
            )
            # Only native model initialization without cache resources enables
            # this path. Live requests with missing KV fail instead of inventing KV.
            table = torch.arange(len(lengths) * width, dtype=torch.int32).view(
                len(lengths), width
            )
            self.warmup_cache = torch.zeros(
                (
                    table.numel(),
                    2,
                    self.configs.kv_head_num,
                    self.configs.kernel_tokens_per_block,
                    self.configs.size_per_head,
                ),
                dtype=torch.bfloat16,
                device="cuda",
            )
        if table is None or table.ndim != 2:
            raise ValueError(
                "batch invariant attention requires a paged KV block table"
            )
        # Metadata is prepared before capture, using the same planner for both phases.
        if inputs.input_lengths.is_cuda and not self.warmup:
            from rtp_llm.models_py.modules.factory.attention.cuda_impl.py_flashinfer_mha import (
                _device_or,
            )

            self.params.fill_params_mha_device(
                _device_or(inputs.prefix_lengths_device, inputs.prefix_lengths),
                inputs.sequence_lengths,
                _device_or(inputs.input_lengths_device, inputs.input_lengths),
                _device_or(inputs.kv_cache_kernel_block_id_device, table),
                self.configs.kernel_tokens_per_block,
                replay,
            )
        else:
            from rtp_llm.models_py.modules.factory.attention.cuda_impl.py_flashinfer_mha import (
                _host_i32,
            )

            self.params.fill_params(
                _host_i32(inputs.prefix_lengths),
                _host_i32(inputs.sequence_lengths),
                _host_i32(inputs.input_lengths),
                _host_i32(table),
                self.configs.kernel_tokens_per_block,
                forbid_realloc=replay,
            )
        if replay:
            if self.block_table.shape != table.shape:
                raise ValueError(
                    "batch invariant graph replay requires the captured block-table shape"
                )
            self.block_table.copy_(table, non_blocking=True)
        else:
            self.block_table = table.to(device="cuda", dtype=torch.int32).clone()
        batch = table.shape[0]
        if inputs.input_lengths.numel() != batch:
            raise ValueError("batch invariant query lengths must match the block table")
        if not replay:
            self.query_start = torch.empty(batch + 1, device="cuda", dtype=torch.int32)
        elif self.query_start.numel() != batch + 1:
            raise ValueError(
                "batch invariant Graph replay requires the captured batch size"
            )
        # The MHA device planner deliberately leaves MLA's qo_indptr untouched.
        # Build our packed offsets from the actual query lengths on device,
        # preserving the allocation across replay and host/device metadata paths.
        lengths = inputs.input_lengths.to(device="cuda", dtype=torch.int32)
        decode_count = (
            inputs.sequence_lengths.numel()
            if inputs.sequence_lengths is not None
            else 0
        )
        if decode_count:
            lengths = lengths.clone()
            lengths[:decode_count] = 1
        self.query_start[:1].zero_()
        torch.cumsum(lengths, dim=0, dtype=torch.int32, out=self.query_start[1:])
        self.inputs = inputs
        self.write_cache_store_impl = common.create_write_cache_store_impl(inputs)

    def prepare_cuda_graph(self, inputs):
        self.prepare(inputs, replay=True)

    def support_cuda_graph(self):
        return True

    @staticmethod
    def support(configs, inputs):
        return not configs.use_mla and configs.dtype == torch.bfloat16

    def forward(self, qkv, kv_cache, layer_idx=0):
        from .kernels import paged_attention

        if kv_cache is None and not self.warmup:
            raise ValueError(
                "batch invariant attention requires an initialized KV cache"
            )
        cache = kv_cache.kv_cache_base if kv_cache is not None else self.warmup_cache
        if cache.ndim == 2:
            cache = common.reshape_paged_kv_cache(
                cache,
                self.configs.kv_head_num,
                self.configs.kernel_tokens_per_block,
                self.configs.size_per_head,
            )
        q, k, v = self.rope.forward(qkv)
        self.writer.forward(k, v, SimpleNamespace(kv_cache_base=cache))
        if kv_cache is not None:
            common.apply_write_cache_store(
                self.write_cache_store_impl, self.inputs, kv_cache
            )
        rows = q.shape[0]
        return paged_attention(
            q,
            cache,
            self.block_table,
            self.params.batch_indice_d[:rows],
            self.params.positions_d[:rows],
            query_start=self.query_start,
        )

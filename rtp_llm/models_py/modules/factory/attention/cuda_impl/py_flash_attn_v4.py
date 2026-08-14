import functools
import importlib
import logging
from typing import Any, Callable, NamedTuple, Optional

import torch

from rtp_llm.models_py.modules.factory.attention import common
from rtp_llm.models_py.modules.factory.attention.cuda_impl.flashinfer_rotary_emb import (
    MhaRotaryEmbeddingOp,
)
from rtp_llm.models_py.modules.factory.attention.cuda_impl.py_flashinfer_mha import (
    PyFlashinferPrefillImplBase,
)
from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.flashinfer_mla import (
    check_attention_inputs,
)
from rtp_llm.ops import AttentionConfigs, ParallelismConfig, RopeStyle
from rtp_llm.ops.compute_ops import LayerKVCache, PyAttentionInputs, rtp_llm_ops

_FA4_TILE_M = 64
_FA4_TILE_N = 32
_FA4_MAX_SPLITS = 128


def _ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


def _get_num_splits(
    *,
    sm_count: int,
    batch_size: int,
    query_len: int,
    num_q_heads: int,
    num_kv_heads: int,
    max_kv_len: int,
) -> int:
    packed_q_rows = query_len * (num_q_heads // num_kv_heads)
    total_m_blocks = batch_size * num_kv_heads * _ceil_div(packed_q_rows, _FA4_TILE_M)
    num_n_blocks = _ceil_div(max_kv_len, _FA4_TILE_N)
    if num_n_blocks <= 4 or total_m_blocks == 0:
        return 1
    return max(
        1,
        min(
            num_n_blocks,
            sm_count // total_m_blocks,
            _FA4_MAX_SPLITS,
        ),
    )


@functools.cache
def _load_fa_forward() -> Optional[Callable[..., Any]]:
    """Load the FA4 forward entry, caching both availability and failure.

    fa_logging pins a level on the "flash_attn" logger at import time (plus a
    stdout handler when FA_LOG_LEVEL >= 1).  That is the same logger namespace as
    the installed FlashAttention 2 wheel: the generated-package rename rewrites
    flash_attn.cute module paths but not the bare logger name. Hand the level back
    to rtp_llm's logging config when FA logging is off.
    """
    try:
        fa_logging = importlib.import_module("vllm_flash_attention.cute.fa_logging")
        if fa_logging.get_fa_log_level() == 0:
            logging.getLogger("flash_attn").setLevel(logging.NOTSET)
        interface = importlib.import_module("vllm_flash_attention.cute.interface")
        return interface._flash_attn_fwd
    except Exception as error:
        logging.warning("FA4 is unavailable and will be skipped: %s", error)
        return None


class FlashAttn4MTPParams(NamedTuple):
    batch_size: int
    query_len: int
    num_splits: int
    input_lengths: torch.Tensor
    prefix_lengths: torch.Tensor
    kv_lengths: torch.Tensor
    page_table: torch.Tensor


class FlashAttn4MTPOp:
    """SM90 paged attention for fixed-width MTP target verify and draft prefill forwards."""

    def __init__(
        self, attn_configs: AttentionConfigs, _attn_inputs: PyAttentionInputs
    ) -> None:
        self.head_dim = attn_configs.size_per_head
        self.head_num = attn_configs.head_num
        self.kv_head_num = attn_configs.kv_head_num
        self.page_size = attn_configs.kernel_tokens_per_block
        self.max_kv_len = attn_configs.max_seq_len
        self.query_len = attn_configs.gen_num_per_cycle + 1
        if self.query_len <= 1:
            raise ValueError("FlashAttn4MTPOp requires multi-token MTP inputs")
        self.softmax_scale = (
            attn_configs.softmax_extra_scale
            / attn_configs.q_scaling
            * self.head_dim**-0.5
        )
        self.fmha_params = rtp_llm_ops.FlashInferMlaAttnParams()
        self.mtp_params: FlashAttn4MTPParams | None = None
        fa_forward = _load_fa_forward()
        if fa_forward is None:
            raise RuntimeError("FA4 is unavailable in the current runtime package")
        self._forward = fa_forward

    def set_params(self, params: rtp_llm_ops.FlashInferMlaAttnParams) -> None:
        self.fmha_params = params

    def prepare(self, attn_inputs: PyAttentionInputs) -> None:
        check_attention_inputs(attn_inputs)

        total_tokens = int(attn_inputs.total_tokens)
        if total_tokens <= 0 or total_tokens % self.query_len != 0:
            raise ValueError(
                "FlashAttn4MTPOp requires dense MTP tokens divisible by query_len"
            )
        batch_size = total_tokens // self.query_len
        page_table_capacity = attn_inputs.kv_cache_kernel_block_id_device
        if page_table_capacity is None:
            raise ValueError("FlashAttn4MTPOp requires a device page table")
        if page_table_capacity.shape[0] < batch_size:
            raise ValueError(
                "FlashAttn4MTPOp page table does not cover the dense batch"
            )
        page_table = page_table_capacity[:batch_size]
        prefix_lengths_capacity = attn_inputs.prefix_lengths_device
        if prefix_lengths_capacity is None:
            prefix_lengths_capacity = attn_inputs.prefix_lengths
        if prefix_lengths_capacity is None:
            raise ValueError("FlashAttn4MTPOp requires device prefix lengths")
        if prefix_lengths_capacity.shape[0] < batch_size:
            raise ValueError(
                "FlashAttn4MTPOp prefix lengths do not cover the dense batch"
            )
        prefix_lengths = prefix_lengths_capacity[:batch_size]
        kv_lengths = torch.empty_like(prefix_lengths)
        input_lengths = torch.full_like(kv_lengths, self.query_len)
        sm_count = torch.cuda.get_device_properties(
            prefix_lengths.device
        ).multi_processor_count
        num_splits = _get_num_splits(
            sm_count=sm_count,
            batch_size=batch_size,
            query_len=self.query_len,
            num_q_heads=self.head_num,
            num_kv_heads=self.kv_head_num,
            max_kv_len=self.max_kv_len,
        )
        self.mtp_params = FlashAttn4MTPParams(
            batch_size=batch_size,
            query_len=self.query_len,
            num_splits=num_splits,
            input_lengths=input_lengths,
            prefix_lengths=prefix_lengths,
            kv_lengths=kv_lengths,
            page_table=page_table,
        )

        torch.add(
            self.mtp_params.prefix_lengths,
            self.query_len,
            out=self.mtp_params.kv_lengths,
        )
        self.fmha_params.fill_params_mha_device(
            self.mtp_params.prefix_lengths,
            self.mtp_params.kv_lengths,
            self.mtp_params.input_lengths,
            self.mtp_params.page_table,
            self.page_size,
            False,
        )

    def prepare_cuda_graph(self, attn_inputs: PyAttentionInputs) -> None:
        check_attention_inputs(attn_inputs)
        if self.mtp_params is None:
            raise RuntimeError(
                "FA4 MTP metadata must be initialized before graph replay"
            )

        expected_tokens = self.mtp_params.batch_size * self.mtp_params.query_len
        if int(attn_inputs.total_tokens) != expected_tokens:
            raise RuntimeError("FA4 MTP token count cannot change during graph replay")

        page_table_capacity = attn_inputs.kv_cache_kernel_block_id_device
        prefix_lengths_capacity = attn_inputs.prefix_lengths_device
        if prefix_lengths_capacity is None:
            prefix_lengths_capacity = attn_inputs.prefix_lengths
        if page_table_capacity is None or prefix_lengths_capacity is None:
            raise RuntimeError("FA4 MTP graph replay requires device metadata buffers")
        page_table = page_table_capacity[: self.mtp_params.batch_size]
        prefix_lengths = prefix_lengths_capacity[: self.mtp_params.batch_size]
        if (
            page_table.shape != self.mtp_params.page_table.shape
            or prefix_lengths.shape != self.mtp_params.prefix_lengths.shape
        ):
            raise RuntimeError(
                "FA4 MTP metadata shape cannot change during graph replay"
            )
        if (
            page_table.data_ptr() != self.mtp_params.page_table.data_ptr()
            or prefix_lengths.data_ptr() != self.mtp_params.prefix_lengths.data_ptr()
        ):
            raise RuntimeError(
                "FA4 MTP metadata buffers cannot change during graph replay"
            )

        torch.add(
            self.mtp_params.prefix_lengths,
            self.query_len,
            out=self.mtp_params.kv_lengths,
        )
        self.fmha_params.fill_params_mha_device(
            self.mtp_params.prefix_lengths,
            self.mtp_params.kv_lengths,
            self.mtp_params.input_lengths,
            self.mtp_params.page_table,
            self.page_size,
            True,
        )

    def forward(
        self, query: torch.Tensor, kv_cache: Optional[LayerKVCache]
    ) -> torch.Tensor:
        if kv_cache is None or self.mtp_params is None:
            raise ValueError("FlashAttn4MTPOp requires prepared paged KV cache inputs")
        paged_kv_cache = common.reshape_paged_kv_cache(
            kv_cache.kv_cache_base,
            self.kv_head_num,
            self.page_size,
            self.head_dim,
        )
        original_shape = query.shape
        expected_tokens = self.mtp_params.batch_size * self.mtp_params.query_len
        if query.shape != (expected_tokens, self.head_num, self.head_dim):
            raise ValueError(
                "FA4 MTP query must contain every token in the fixed graph bucket"
            )
        if query.dtype != torch.bfloat16 or paged_kv_cache.dtype != query.dtype:
            raise TypeError("FA4 MTP requires matching BF16 query and KV cache")
        dense_query = query.reshape(
            self.mtp_params.batch_size,
            self.mtp_params.query_len,
            self.head_num,
            self.head_dim,
        )
        key_cache = paged_kv_cache.select(1, 0).permute(0, 2, 1, 3)
        value_cache = paged_kv_cache.select(1, 1).permute(0, 2, 1, 3)
        output = self._forward(
            dense_query,
            key_cache,
            value_cache,
            seqused_k=self.mtp_params.kv_lengths,
            max_seqlen_q=self.mtp_params.query_len,
            max_seqlen_k=self.max_kv_len,
            page_table=self.mtp_params.page_table,
            softmax_scale=self.softmax_scale,
            causal=True,
            tile_mn=(_FA4_TILE_M, _FA4_TILE_N),
            mma_pv_is_rs=True,
            intra_wg_overlap=True,
            num_threads=256,
            num_splits=self.mtp_params.num_splits,
            pack_gqa=True,
        )[0]
        return output.reshape(original_shape)


class FlashAttn4MTPImpl(PyFlashinferPrefillImplBase):
    """MTP attention with separate FlashInfer RoPE and KV cache write ops."""

    def __init__(
        self,
        attn_configs: AttentionConfigs,
        attn_inputs: PyAttentionInputs,
        parallelism_config: Optional[ParallelismConfig] = None,
    ) -> None:
        super().__init__(attn_configs, attn_inputs, parallelism_config)

    def _create_fmha_impl(
        self, attn_configs: AttentionConfigs, attn_inputs: PyAttentionInputs
    ) -> Any:
        return FlashAttn4MTPOp(attn_configs, attn_inputs)

    def _create_rope_impl(self, attn_configs: AttentionConfigs) -> Any:
        if attn_configs.rope_config.style == RopeStyle.No:
            return None
        return MhaRotaryEmbeddingOp(attn_configs)

    def prepare_cuda_graph(self, attn_inputs: PyAttentionInputs) -> None:
        self.fmha_impl.prepare_cuda_graph(attn_inputs)

    @staticmethod
    def support(attn_configs: AttentionConfigs, attn_inputs: PyAttentionInputs) -> bool:
        return False

    def support_cuda_graph(self) -> bool:
        return True

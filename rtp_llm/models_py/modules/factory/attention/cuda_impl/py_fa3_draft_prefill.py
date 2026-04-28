"""FlashAttention 3 based MTP draft-prefill attention.

This implementation mirrors ``py_fa3_target_verify.py`` but for the
draft-model prefill step in MTP (the first forward of the draft model that
consumes the verified hidden + accepted token from target verify).  The
draft model runs under its own CUDA graph instance with
``is_prefill_cuda_graph_mode=True`` and ``num_tokens_per_bs=gen_num_per_cycle+1``
(typically 4).

The legacy path (``PyFlashinferPrefillPagedAttnOp`` with the
``small2large``/``large2small`` repack) hits the FlashInfer
``BatchPrefillWithPagedKVCacheWrapper.plan/run`` aliasing bug under CUDA
graph capture (see ``project_draft_prefill_cg_root_cause`` memory): each
``plan()`` call returns fresh ``plan_info`` tensors, but the captured
graph kernel still launches with the capture-time ``plan_info`` data
pointers, so replays read stale per-batch scheduling parameters and
occasionally produce garbled output.

FA3's ``flash_attn_with_kvcache`` exposes a single fused launch with no
plan/run split, so all per-call scalars (cu_seqlens_q, cache_seqlens,
page_table) are read directly from device buffers at kernel runtime.  We
manage these buffers ourselves so they stay stable across replays.

Compared to target verify, draft prefill differs in two important ways:

* Q lengths vary per request (each batch has up to ``num_tokens_per_bs``
  query tokens, but the actual count depends on how many draft tokens
  this step is producing for each request).  Target verify uses a fixed
  ``num_tokens_per_bs`` per request.
* Q layout is *compact* (no aligned padding between batches) — rows
  ``[0, sum(input_lengths))`` are real, rows after that are stale.  The
  legacy FlashInfer impl repacked compact → aligned because FlashInfer's
  CG buffers needed a fixed ``qo_indptr``.  FA3 reads cu_seqlens_q at
  runtime so we keep the compact layout and just give FA3 a cu that
  reflects the actual per-replay request sizes.
"""

from typing import Any, Optional

import torch
from flash_attn_interface import flash_attn_with_kvcache

from rtp_llm.models_py.modules.factory.attention import common
from rtp_llm.models_py.modules.factory.attention.cuda_impl.flashinfer_rotary_emb import (
    MhaRotaryEmbeddingOp,
)
from rtp_llm.models_py.modules.factory.attention.cuda_impl.py_flashinfer_mha import (
    PyFlashinferPrefillImplBase,
)
from rtp_llm.models_py.modules.factory.attention.cuda_impl.utils import is_sm_100
from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.flashinfer_mla import (
    check_attention_inputs,
)
from rtp_llm.ops import AttentionConfigs, KvCacheDataType, RopeStyle
from rtp_llm.ops.compute_ops import (
    LayerKVCache,
    ParamsBase,
    PyAttentionInputs,
    rtp_llm_ops,
)


class PyFA3PagedDraftPrefillAttnOp(object):
    """MTP draft-prefill backed by FlashAttention 3.

    Inputs to ``forward`` follow the existing op contract:

    * ``q``: ``[total_q_tokens, num_heads, head_dim]`` packed varlen tensor.
      In CUDA-graph mode the caller pads ``total_q_tokens`` to
      ``max_bs * num_tokens_per_bs``; padding rows are quietly skipped
      because cu_seqlens_q's padding entries all point at the same offset.
    * ``kv_cache.kv_cache_base``: paged KV cache, either packed 2D (raw)
      or ``[num_pages, 2, num_kv_heads, page_size, head_dim]`` (HND).
    """

    def __init__(
        self,
        attn_configs: AttentionConfigs,
        attn_inputs: PyAttentionInputs,
    ) -> None:
        self.local_head_num = attn_configs.head_num
        self.local_kv_head_num = attn_configs.kv_head_num
        self.head_dim_qk = attn_configs.size_per_head
        self.head_dim_vo = attn_configs.size_per_head
        self.page_size = attn_configs.kernel_tokens_per_block
        self.max_seq_len = attn_configs.max_seq_len
        self.datatype = attn_configs.dtype
        self.kv_cache_dtype = attn_configs.kv_cache_dtype
        self.softmax_scale = float(self.head_dim_qk) ** -0.5
        self.fmha_params = rtp_llm_ops.FlashInferMlaAttnParams()
        self.enable_cuda_graph = getattr(attn_inputs, "is_cuda_graph", False)

        # Stable CG buffers for the three runtime-read tensors.
        # Allocated lazily on first prepare() so we know max_bs and
        # num_tokens_per_bs from the captured layout.
        self._cache_seqlens_buf: Optional[torch.Tensor] = None
        self._cu_seqlens_q_buf: Optional[torch.Tensor] = None
        self._page_table_ref: Optional[torch.Tensor] = None
        self._cuda_graph_initialized = False
        self._fixed_batch_size = 0
        self._max_q_per_request = 1

    def set_params(self, params: Any) -> None:
        self.fmha_params = params

    def _get_kv_dtype(self, attn_inputs: PyAttentionInputs) -> torch.dtype:
        if self.kv_cache_dtype == KvCacheDataType.INT8:
            return torch.int8
        if self.kv_cache_dtype == KvCacheDataType.FP8:
            return torch.float8_e4m3fn
        return attn_inputs.dtype

    def _ensure_cg_buffers(self, attn_inputs: PyAttentionInputs) -> None:
        if self._cuda_graph_initialized:
            return
        batch_size = attn_inputs.input_lengths.size(0)
        if attn_inputs.input_lengths.numel() > 0:
            max_q = max(int(attn_inputs.input_lengths.max().item()), 1)
        else:
            max_q = 1
        self._fixed_batch_size = batch_size
        self._max_q_per_request = max_q
        self._cache_seqlens_buf = torch.zeros(
            batch_size, dtype=torch.int32, device="cuda"
        )
        self._cu_seqlens_q_buf = torch.zeros(
            batch_size + 1, dtype=torch.int32, device="cuda"
        )
        # page_table reference: kv_cache_kernel_block_id_device is a stable
        # CG buffer maintained by the C++ runner (refreshed each prepareInputs).
        self._page_table_ref = attn_inputs.kv_cache_kernel_block_id_device
        self._cuda_graph_initialized = True

    def _refresh_cg_buffers(self, attn_inputs: PyAttentionInputs) -> None:
        """Refresh stable buffers from the live device tensors.

        Both ``input_lengths_d`` and ``prefix_lengths_d`` are stable CG
        buffers refreshed by ``CudaGraphRunner::prepareInputs`` each
        replay; padding rows are zeroed.
        """
        assert self._cache_seqlens_buf is not None
        assert self._cu_seqlens_q_buf is not None
        n = min(self._fixed_batch_size, self._cache_seqlens_buf.size(0))

        prefix_d = attn_inputs.prefix_lengths_d[:n].to(torch.int32)
        input_d = attn_inputs.input_lengths_d[:n].to(torch.int32)

        # cache_seqlens = prefix + input (KV positions visible to attention).
        # Padding rows are 0 because both prefix_lengths_d and input_lengths_d
        # are zeroed for padding batches by the C++ runner.
        torch.add(prefix_d, input_d, out=self._cache_seqlens_buf[:n])
        if n < self._cache_seqlens_buf.size(0):
            self._cache_seqlens_buf[n:].zero_()

        # cu_seqlens_q = [0, cumsum(input_lengths)]
        # Active entries cover [0, total_active_tokens); padding entries
        # equal total_active_tokens because input_d[active:] is 0 (so cumsum
        # plateaus).  FA3 sees padding batches as 0-length ranges and skips
        # them, so the stale tail of q (rows >= total_active_tokens) is never
        # read.
        self._cu_seqlens_q_buf[0].zero_()
        torch.cumsum(input_d, dim=0, out=self._cu_seqlens_q_buf[1 : n + 1])
        if n < self._fixed_batch_size:
            self._cu_seqlens_q_buf[n + 1 :].fill_(self._cu_seqlens_q_buf[n])

    def _build_cu_seqlens_q_nocg(
        self, attn_inputs: PyAttentionInputs, batch_size: int
    ) -> torch.Tensor:
        """Eager-mode cu_seqlens_q from live ``cu_seqlens`` (CPU pinned).

        For non-CG draft prefill (rare — usually all draft prefill runs
        through CG), we build a one-shot GPU cu directly from the live
        ``cu_seqlens`` tensor that PyWrappedModel already prepared.
        """
        cu = getattr(attn_inputs, "cu_seqlens", None)
        if cu is not None and cu.numel() >= batch_size + 1:
            cu_slice = cu[: batch_size + 1].to(torch.int32)
            if cu_slice.device.type != "cuda":
                cu_slice = cu_slice.to(device="cuda", non_blocking=True)
            return cu_slice
        # Fallback: cumsum on host.
        input_cpu = attn_inputs.input_lengths.to(torch.int32).cpu()
        cu_cpu = torch.zeros(batch_size + 1, dtype=torch.int32)
        torch.cumsum(input_cpu, dim=0, out=cu_cpu[1:])
        return cu_cpu.to(device="cuda", non_blocking=True)

    def prepare(
        self, attn_inputs: PyAttentionInputs, forbid_realloc: bool = False
    ) -> ParamsBase:
        check_attention_inputs(attn_inputs)
        # fill_params is required: KVCacheWriteOp / RoPE op consume
        # batch_indice_d, positions_d, page_indice_d, decode_page_indptr_d,
        # paged_kv_last_page_len_d, kvlen_d from the shared
        # FlashInferMlaAttnParams.  We do not bind these tensors as captured
        # FA3 kernel inputs (FA3 reads its own stable buffers below).
        self.fmha_params.fill_params(
            attn_inputs.prefix_lengths,
            attn_inputs.sequence_lengths,
            attn_inputs.input_lengths,
            attn_inputs.kv_cache_kernel_block_id_host,
            self.page_size,
            forbid_realloc,
        )

        if self.enable_cuda_graph:
            self._ensure_cg_buffers(attn_inputs)
            self._refresh_cg_buffers(attn_inputs)
        else:
            batch_size = attn_inputs.input_lengths.size(0)
            self._cu_seqlens_q_buf = self._build_cu_seqlens_q_nocg(
                attn_inputs, batch_size
            )
            self._cache_seqlens_buf = attn_inputs.prefix_lengths_d[:batch_size].to(
                torch.int32
            ) + attn_inputs.input_lengths_d[:batch_size].to(torch.int32)
            self._page_table_ref = attn_inputs.kv_cache_kernel_block_id_device
            if attn_inputs.input_lengths.numel() > 0:
                self._max_q_per_request = max(
                    int(attn_inputs.input_lengths.max().item()), 1
                )
            else:
                self._max_q_per_request = 1

        return self.fmha_params

    @staticmethod
    def support(attn_inputs: PyAttentionInputs) -> bool:
        return True

    def _get_kv_caches(
        self, kv_cache: LayerKVCache
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Slice K/V from the paged KV cache and convert HND-per-block layout
        to the ``[num_blocks, page_size, kv_heads, head_dim]`` layout FA3
        expects.  See py_fa3_target_verify._get_kv_caches for details.
        """
        paged = kv_cache.kv_cache_base
        if paged.dim() == 2:
            paged = common.reshape_paged_kv_cache(
                paged, self.local_kv_head_num, self.page_size, self.head_dim_qk
            )
        k = paged[:, 0]
        v = paged[:, 1]
        return k.permute(0, 2, 1, 3), v.permute(0, 2, 1, 3)

    def forward(
        self, q: torch.Tensor, kv_cache: Optional[LayerKVCache]
    ) -> torch.Tensor:
        assert kv_cache is not None, "kv_cache is required for draft prefill"
        assert (
            q.dim() == 3
        ), f"Expected q [total_tokens, H, D], got shape {tuple(q.shape)}"
        assert (
            self._cu_seqlens_q_buf is not None
            and self._cache_seqlens_buf is not None
            and self._page_table_ref is not None
        ), "PyFA3PagedDraftPrefillAttnOp.prepare() must run before forward()"

        k_cache, v_cache = self._get_kv_caches(kv_cache)
        out = flash_attn_with_kvcache(
            q=q,
            k_cache=k_cache,
            v_cache=v_cache,
            cache_seqlens=self._cache_seqlens_buf,
            page_table=self._page_table_ref,
            cu_seqlens_q=self._cu_seqlens_q_buf,
            max_seqlen_q=self._max_q_per_request,
            softmax_scale=self.softmax_scale,
            causal=True,
            pack_gqa=True,
        )
        if isinstance(out, tuple):
            out = out[0]
        return out


class PyFA3DraftPrefillImpl(PyFlashinferPrefillImplBase):
    """Wrapper Impl wiring the FA3 draft-prefill op into the factory.

    Reuses ``PyFlashinferPrefillImplBase`` for QKV split + RoPE + KV write
    plumbing; only the attention backend differs.
    """

    def _create_fmha_impl(
        self, attn_configs: AttentionConfigs, attn_inputs: PyAttentionInputs
    ) -> Any:
        return PyFA3PagedDraftPrefillAttnOp(attn_configs, attn_inputs)

    def _create_rope_impl(self, attn_configs: AttentionConfigs) -> Any:
        if attn_configs.rope_config.style == RopeStyle.No:
            return None
        return MhaRotaryEmbeddingOp(attn_configs)

    def _prepare_fmha_input(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
    ) -> torch.Tensor:
        return query

    def support_cuda_graph(self) -> bool:
        return True

    @staticmethod
    def support(attn_configs: AttentionConfigs, attn_inputs: PyAttentionInputs) -> bool:
        # Restrict to draft prefill in CUDA-graph-driven mode.  Initial
        # prompt prefill (variable user input length) goes through
        # PyFlashinferPagedPrefillImpl etc. — we don't want to touch it.
        # Detection: prefill_cuda_graph_copy_params is set only by the
        # C++ CudaGraphRunner for prefill CG mode (draft prefill or
        # embedding model).  Embedding model has num_tokens_per_bs ==
        # max_seq_len, which we additionally exclude.
        if not attn_inputs.is_prefill:
            return False
        if getattr(attn_inputs, "is_target_verify", False):
            return False
        if attn_configs.use_mla:
            return False
        if is_sm_100():
            return False
        copy_params = getattr(attn_inputs, "prefill_cuda_graph_copy_params", None)
        if copy_params is None:
            return False
        # Embedding model: max_seq_len == num_tokens_per_bs (single-shot prefill
        # over a fixed input window).  We only target draft prefill, where
        # max_seq_len > num_tokens_per_bs.
        max_sl = getattr(copy_params, "max_seq_len", None)
        if max_sl is not None and max_sl >= attn_configs.max_seq_len:
            return False
        return True

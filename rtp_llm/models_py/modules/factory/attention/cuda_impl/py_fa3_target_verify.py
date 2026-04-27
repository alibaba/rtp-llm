"""FlashAttention 3 based MTP target verify attention.

This implementation mirrors what SGLang's FlashAttention backend uses for the
target-verify path on Hopper (sm9x): a single ``flash_attn_with_kvcache`` call
in varlen + paged-KV mode with bottom-right causal masking.

Compared to the FlashInfer ``BatchPrefillWithPagedKVCacheWrapper`` based op
(``PyFlashinferPrefillPagedTargetVerifyAttnOp``), FA3:

* avoids the ``plan()`` step that bakes capture-time scheduling into the CUDA
  graph (root cause of the non-determinism documented in the
  ``project_target_verify_cg_bug`` memory).  FA3 exposes a single fused launch
  with no plan/run split, so there are no auxiliary scalars that need to match
  between capture and replay;
* consumes the 2-D ``page_table[B, max_pages_per_seq]`` directly from
  ``attn_inputs.kv_cache_kernel_block_id_device``, which is a stable CUDA
  buffer maintained by ``CudaGraphRunner::prepareInputs`` — no per-call
  derivation from the dynamically-reshaped ``page_indice_d`` tensors;
* writes ``cache_seqlens`` into a dedicated stable buffer so the kernel reads
  the same storage every replay.
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


class PyFA3PagedTargetVerifyAttnOp(object):
    """MTP target verify backed by FlashAttention 3.

    Inputs to ``forward`` follow the existing op contract:

    * ``q``: ``[total_q_tokens, num_heads, head_dim]`` packed varlen tensor.
      In CUDA-graph mode the caller pads ``total_q_tokens`` to
      ``max_bs * num_tokens_per_bs``; padding rows are quietly skipped via
      ``cu_seqlens_q``.
    * ``kv_cache.kv_cache_base``: paged KV cache, either packed 2D (raw) or
      ``[num_pages, 2, num_kv_heads, page_size, head_dim]`` (HND).
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

        # Stable CUDA-graph buffer for cache_seqlens.  cu_seqlens_q and
        # page_table are stable buffers maintained by the C++ CudaGraphRunner,
        # so we only stash references to them after the first prepare().
        self._cache_seqlens_buf: Optional[torch.Tensor] = None
        self._cu_seqlens_q_ref: Optional[torch.Tensor] = None
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

    def _build_cu_seqlens_q_nocg(
        self, attn_inputs: PyAttentionInputs, batch_size: int
    ) -> torch.Tensor:
        """Build cu_seqlens_q for the non-CUDA-graph path.

        ``decode_cu_seqlens_d`` is acceptable iff it actually encodes the
        per-request input lengths.  In single-token decode the C++ layer
        fills it with ``arange(0, bs+1)``, which would be wrong for target
        verify with > 1 draft tokens — fall back to a fresh cumsum then.
        """
        cu = getattr(attn_inputs, "decode_cu_seqlens_d", None)
        input_lengths = attn_inputs.input_lengths
        if (
            cu is not None
            and cu.numel() >= batch_size + 1
            and input_lengths.numel() > 0
            and int(cu[1].item()) == int(input_lengths[0].item())
        ):
            return cu[: batch_size + 1]
        input_cpu = input_lengths.to(torch.int32).cpu()
        cu_cpu = torch.zeros(batch_size + 1, dtype=torch.int32)
        torch.cumsum(input_cpu, dim=0, out=cu_cpu[1:])
        return cu_cpu.to(device="cuda", non_blocking=True)

    def _ensure_cg_buffers(self, attn_inputs: PyAttentionInputs) -> None:
        if self._cuda_graph_initialized:
            return
        batch_size = attn_inputs.input_lengths.size(0)
        # All requests in target-verify CG capture share the same
        # num_tokens_per_bs draft-token budget, so input_lengths.max() at
        # capture time is the per-request Q length used by FA3 for tile sizing.
        if attn_inputs.input_lengths.numel() > 0:
            max_q = max(int(attn_inputs.input_lengths.max().item()), 1)
        else:
            max_q = 1
        self._fixed_batch_size = batch_size
        self._max_q_per_request = max_q
        self._cache_seqlens_buf = torch.zeros(
            batch_size, dtype=torch.int32, device="cuda"
        )
        # Stash references to the stable C++-side CG buffers.  These tensor
        # objects (and their data_ptrs) must not change across replays, which
        # is guaranteed by CudaGraphRunner::initCaptureAttentionInputs* +
        # prepareInputs (it slice/copy_'s into the same storage).
        self._cu_seqlens_q_ref = attn_inputs.decode_cu_seqlens_d[: batch_size + 1]
        self._page_table_ref = attn_inputs.kv_cache_kernel_block_id_device
        self._cuda_graph_initialized = True

    def _refresh_cache_seqlens(self, attn_inputs: PyAttentionInputs) -> None:
        """Fill the stable buffer with ``prefix_lengths_d + input_lengths_d``.

        Both source tensors live in stable CG storage filled by the C++
        runner each ``prepareInputs``.  Padding rows already have zero
        ``input_lengths`` and zero ``prefix_lengths`` (cleared in the runner
        for target verify), so ``cache_seqlens=0`` for them — FA3 emits a
        no-op for the padding requests and never dereferences their (also
        zeroed) page_table rows.
        """
        assert self._cache_seqlens_buf is not None
        out = self._cache_seqlens_buf
        n = min(self._fixed_batch_size, out.size(0))
        prefix_d = attn_inputs.prefix_lengths_d[:n].to(torch.int32)
        input_d = attn_inputs.input_lengths_d[:n].to(torch.int32)
        torch.add(prefix_d, input_d, out=out[:n])
        if n < out.size(0):
            out[n:].zero_()

    def prepare(
        self, attn_inputs: PyAttentionInputs, forbid_realloc: bool = False
    ) -> ParamsBase:
        check_attention_inputs(attn_inputs)
        # fill_params still required: KVCacheWriteOp / RoPE op consume
        # batch_indice_d, positions_d, page_indice_d etc. from the shared
        # FlashInferMlaAttnParams.  We do NOT bind decode_page_indptr_d /
        # page_indice_d as captured kernel inputs — those tensors get
        # reshaped per call by FlashInferMlaParams::refreshBuffer and would
        # break CUDA-graph aliasing assumptions.
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
            self._refresh_cache_seqlens(attn_inputs)
        else:
            # Non-CG: build per-call (small tensors, kernel launch is fine).
            batch_size = attn_inputs.input_lengths.size(0)
            self._cu_seqlens_q_ref = self._build_cu_seqlens_q_nocg(
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
        expects.

        The permute keeps the last dim (head_dim) as the innermost storage
        axis, so ``stride(-1) == 1`` still holds and FA3's contiguity check
        passes without a memory copy.
        """
        paged = kv_cache.kv_cache_base
        if paged.dim() == 2:
            paged = common.reshape_paged_kv_cache(
                paged, self.local_kv_head_num, self.page_size, self.head_dim_qk
            )
        # paged: [num_pages, 2, kv_heads, page_size, head_dim]
        k = paged[:, 0]  # [num_pages, kv_heads, page_size, head_dim]
        v = paged[:, 1]
        # -> [num_pages, page_size, kv_heads, head_dim]
        return k.permute(0, 2, 1, 3), v.permute(0, 2, 1, 3)

    def forward(
        self, q: torch.Tensor, kv_cache: Optional[LayerKVCache]
    ) -> torch.Tensor:
        assert kv_cache is not None, "kv_cache is required for target verify"
        assert (
            q.dim() == 3
        ), f"Expected q [total_tokens, H, D], got shape {tuple(q.shape)}"
        assert (
            self._cu_seqlens_q_ref is not None
            and self._cache_seqlens_buf is not None
            and self._page_table_ref is not None
        ), "PyFA3PagedTargetVerifyAttnOp.prepare() must run before forward()"

        k_cache, v_cache = self._get_kv_caches(kv_cache)
        # ``pack_gqa=True`` lands on FA3's PackGQA fast path: with a high
        # GQA ratio (Qwen3.5: 8 q-heads / 1 kv-head per TP rank) and a tiny
        # per-request Q (4 draft tokens), packing the q heads into the same
        # tile cuts kernel time by ~25x in our timeline (matches what
        # SGLang's FA3 backend selects via `pack_gqa=true` heuristics on
        # Hopper, see `VarlenDynamicPersistentTileScheduler<...,true,...>`).
        out = flash_attn_with_kvcache(
            q=q,
            k_cache=k_cache,
            v_cache=v_cache,
            cache_seqlens=self._cache_seqlens_buf,
            page_table=self._page_table_ref,
            cu_seqlens_q=self._cu_seqlens_q_ref,
            max_seqlen_q=self._max_q_per_request,
            softmax_scale=self.softmax_scale,
            causal=True,
            pack_gqa=True,
        )
        if isinstance(out, tuple):
            out = out[0]
        return out


class PyFA3TargetVerifyImpl(PyFlashinferPrefillImplBase):
    """Wrapper Impl wiring the FA3 target-verify op into the factory.

    Reuses the FlashInfer prefill base for QKV split + RoPE + KV write
    plumbing; only the attention backend differs.
    """

    def _create_fmha_impl(
        self, attn_configs: AttentionConfigs, attn_inputs: PyAttentionInputs
    ) -> Any:
        return PyFA3PagedTargetVerifyAttnOp(attn_configs, attn_inputs)

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
        if not attn_inputs.is_prefill:
            return False
        if not getattr(attn_inputs, "is_target_verify", False):
            return False
        if attn_configs.use_mla:
            return False
        if is_sm_100():
            # FA3 does not target Blackwell; defer to other impls there.
            return False
        return True

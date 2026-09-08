"""FA3-based attention backends for PPU.

Provides MHA implementations that use the PPU FlashAttention-3
flash_attn_with_kvcache paged-KV path for decode and prefill, replacing
the default FlashInfer paths that dominate PPU long-sequence attention latency.

These classes are pure implementations (subclass-compatible with the open-source
``FMHAImplBase`` contract); registration into the attention factory's
``PREFILL_MHA_IMPS`` / ``DECODE_MHA_IMPS`` lists is done by the single seam in
``rtp_llm.platforms.ppu.modules.attention.register``.

Controlled by envs:
  - PPU_USE_FA3_DECODE (default "1")
  - PPU_USE_FA3_PREFILL (default "1")
  - PPU_USE_FA3_VERIFY (default "1"): use FA3 for the MTP uniform-query prefill
    forwards captured under CUDA graph -- both the target-verify step and the draft
    prefill step (multi-token-per-seq). FA3 is varlen-native, so it captures cleanly
    where FlashInfer's paged prefill would hit q.shape[0]==qo_indptr[-1] and needed
    the draft graph disabled.

For each FA3 env:
  - "1": enable the FA3 backend (highest priority, fallback to FlashInfer when safe)
  - "0": disable the FA3 backend. MRoPE has no PyFlashinfer prefill fallback on PPU
         because the open-source PyFlashinfer RoPE path is not true Qwen3.5 MRoPE.

Requires: flash_attn_3 whl installed (PPU-adapted build).
"""

import importlib
import importlib.util
import logging
import os
import sys
from typing import Any, Optional

# ---------------------------------------------------------------------------
# FA3 availability check (cached)
# ---------------------------------------------------------------------------

_fa3_available: Optional[bool] = None
_fa3_interface: Optional[Any] = None


def _maybe_contiguous(x):
    """Match vLLM/SGLang FA wrappers: only compact tensors whose last dim is strided."""
    return x.contiguous() if x is not None and x.stride(-1) != 1 else x


def _check_fa3_available() -> bool:
    global _fa3_available
    if _fa3_available is not None:
        return _fa3_available
    try:
        # Bazel pip rules place flash_attn_3 under pip_ppu_torch_flash_attn_3/site-packages/
        # which may not be on sys.path automatically. Add it if we can find it in runfiles.
        import importlib.util

        import torch  # noqa: F401 - must be loaded before flash_attn_3._C (libc10.so dependency)

        if importlib.util.find_spec("flash_attn_3") is None:
            import pathlib

            # Search runfiles for the flash_attn_3 site-packages directory
            for p in sys.path:
                candidate = (
                    pathlib.Path(p).parent
                    / "pip_ppu_torch_flash_attn_3"
                    / "site-packages"
                )
                if candidate.exists() and (candidate / "flash_attn_3").exists():
                    sys.path.insert(0, str(candidate))
                    logging.debug("[fa3_mha] Added %s to sys.path", candidate)
                    break
        import flash_attn_3._C  # noqa: F401

        _fa3_available = True
        logging.info("[fa3_mha] flash_attn_3 loaded successfully")
    except (ImportError, OSError) as e:
        logging.debug("[fa3_mha] flash_attn_3 not available: %s", e)
        _fa3_available = False
    return _fa3_available


def _import_fa3_interface():
    global _fa3_interface
    if _fa3_interface is not None:
        return _fa3_interface
    if not _check_fa3_available():
        raise ImportError("flash_attn_3 is not available")
    _fa3_interface = importlib.import_module("flash_attn_interface")
    return _fa3_interface


# ---------------------------------------------------------------------------
# FA3 Decode Attention Op
# ---------------------------------------------------------------------------


class FA3DecodeAttnOp:
    """FA3-based decode attention using paged KV cache.

    Uses the PPU FA3 wheel's flash_attn_with_kvcache paged path.
    RTP KV cache is [blocks, 2, kv_heads, block_size, head_dim]; FA3 expects
    [blocks, block_size, kv_heads, head_dim] for each of K and V.
    """

    def __init__(self, attn_configs, attn_inputs):
        import torch

        self.local_head_num = attn_configs.head_num
        self.local_kv_head_num = attn_configs.kv_head_num
        self.head_dim = attn_configs.size_per_head
        self.seq_size_per_block = attn_configs.kernel_tokens_per_block
        self.softmax_scale = 1.0 / (self.head_dim**0.5)
        self.enable_cuda_graph = attn_inputs.is_cuda_graph
        self.max_seqlen_k = int(getattr(attn_configs, "max_seq_len", 0) or 0)
        if self.max_seqlen_k <= 0:
            max_seq_len = getattr(attn_inputs, "max_seq_len", None)
            self.max_seqlen_k = int(max_seq_len or 0)

        self._batch_size = 0
        self._scheduler_metadata = None
        self.fmha_params = None
        self._graph_cache_seqlens: Optional[torch.Tensor] = None
        self._graph_cu_seqlens_q: Optional[torch.Tensor] = None
        self._graph_scheduler_metadata: Optional[torch.Tensor] = None
        logging.info(
            "[FA3DecodeAttnOp] init heads=%s kv_heads=%s head_dim=%s block=%s max_seq=%s cuda_graph=%s",
            self.local_head_num,
            self.local_kv_head_num,
            self.head_dim,
            self.seq_size_per_block,
            self.max_seqlen_k,
            self.enable_cuda_graph,
        )

    def _get_scheduler_metadata(
        self, batch_size, max_seqlen_k, cache_seqlens, cu_seqlens_q
    ):
        """Get or refresh FA3 scheduler metadata for tile/split planning."""
        import torch

        fa3 = _import_fa3_interface()

        metadata = fa3.get_scheduler_metadata(
            batch_size=batch_size,
            max_seqlen_q=1,
            max_seqlen_k=max_seqlen_k,
            num_heads_q=self.local_head_num,
            num_heads_kv=self.local_kv_head_num,
            headdim=self.head_dim,
            cache_seqlens=cache_seqlens,
            qkv_dtype=torch.bfloat16,
            cu_seqlens_q=cu_seqlens_q,
            page_size=self.seq_size_per_block,
            causal=True,
            num_splits=0,  # auto
        )
        return metadata

    def _get_max_seqlen_k(self, attn_inputs, cache_seqlens):
        if self.max_seqlen_k > 0:
            return self.max_seqlen_k
        max_seq_len = getattr(attn_inputs, "max_seq_len", None)
        if max_seq_len is not None:
            self.max_seqlen_k = int(max_seq_len)
            return self.max_seqlen_k
        # One-time fallback for non-graph unit paths where config does not carry max_seq_len.
        self.max_seqlen_k = int(cache_seqlens.max().item())
        return self.max_seqlen_k

    def _get_cache_seqlens(self, attn_inputs, batch_size):
        cache_seqlens = getattr(attn_inputs, "sequence_lengths_plus_1_device", None)
        if cache_seqlens is not None and cache_seqlens.numel() >= batch_size:
            return cache_seqlens[:batch_size]
        return attn_inputs.sequence_lengths.to("cuda", non_blocking=True) + 1

    def _init_graph_buffers(self, batch_size):
        import torch

        self._graph_cache_seqlens = torch.empty(
            batch_size, dtype=torch.int32, device="cuda"
        )
        # Full CUDA graphs execute one safe dummy query for every padded batch slot,
        # matching FlashInfer's fixed-batch decode contract.
        self._graph_cu_seqlens_q = torch.arange(
            batch_size + 1, dtype=torch.int32, device="cuda"
        )

    def _refresh_graph_cache_seqlens(self, attn_inputs):
        # CudaGraphRunner refreshes the real prefix and zeroes the host padding. Adding
        # one therefore produces the same KV lengths as FlashInfer: real seq_len + 1
        # and a one-token dummy sequence for every padded slot.
        assert self._graph_cache_seqlens is not None
        self._graph_cache_seqlens.copy_(
            attn_inputs.sequence_lengths, non_blocking=True
        ).add_(1)

    def _refresh_graph_scheduler_metadata(
        self, batch_size, max_seqlen_k, cache_seqlens, cu_seqlens_q
    ):
        metadata = self._get_scheduler_metadata(
            batch_size, max_seqlen_k, cache_seqlens, cu_seqlens_q
        )
        if self._graph_scheduler_metadata is None:
            self._graph_scheduler_metadata = metadata
        else:
            if self._graph_scheduler_metadata.shape != metadata.shape:
                raise RuntimeError(
                    "FA3 scheduler metadata shape changed during CUDA graph replay: "
                    f"captured={tuple(self._graph_scheduler_metadata.shape)}, "
                    f"replay={tuple(metadata.shape)}"
                )
            # CUDA Graph retains the capture-time pointer. Copying the full metadata
            # also resets the scheduler semaphore left by the previous replay.
            self._graph_scheduler_metadata.copy_(metadata)
        return self._graph_scheduler_metadata

    def _get_cu_seqlens_q(self, attn_inputs, batch_size):
        import torch

        cu_seqlens_q = getattr(attn_inputs, "decode_cu_seqlens_device", None)
        if cu_seqlens_q is not None and cu_seqlens_q.numel() >= batch_size + 1:
            return cu_seqlens_q[: batch_size + 1]

        return torch.arange(0, batch_size + 1, dtype=torch.int32, device="cuda")

    def prepare(self, attn_inputs):
        """Prepare FA3 parameters from RTP-LLM attention inputs."""
        batch_size = attn_inputs.sequence_lengths.size(0)
        if self.enable_cuda_graph:
            if self._batch_size not in (0, batch_size):
                raise RuntimeError(
                    f"FA3 decode graph batch changed from {self._batch_size} to {batch_size}"
                )
            self._batch_size = batch_size
            if self._graph_cache_seqlens is None:
                self._init_graph_buffers(batch_size)
            self._refresh_graph_cache_seqlens(attn_inputs)
            cache_seqlens = self._graph_cache_seqlens
            cu_seqlens_q = self._graph_cu_seqlens_q
        else:
            cache_seqlens = _maybe_contiguous(
                self._get_cache_seqlens(attn_inputs, batch_size)
            )
            cu_seqlens_q = _maybe_contiguous(
                self._get_cu_seqlens_q(attn_inputs, batch_size)
            )

        # Block table (device tensor)
        block_table = _maybe_contiguous(attn_inputs.kv_cache_kernel_block_id_device)
        max_seqlen_k = self._get_max_seqlen_k(attn_inputs, cache_seqlens)

        # Get scheduler metadata (FA3 tile planning)
        if self.enable_cuda_graph:
            self._scheduler_metadata = self._refresh_graph_scheduler_metadata(
                batch_size, max_seqlen_k, cache_seqlens, cu_seqlens_q
            )
        else:
            self._scheduler_metadata = self._get_scheduler_metadata(
                batch_size, max_seqlen_k, cache_seqlens, cu_seqlens_q
            )

        self.fmha_params = _FA3Params(
            batch_size=batch_size,
            cache_seqlens=cache_seqlens,
            block_table=block_table,
            max_seqlen_k=max_seqlen_k,
            cu_seqlens_q=cu_seqlens_q,
            scheduler_metadata=self._scheduler_metadata,
        )
        return self.fmha_params

    def prepare_for_cuda_graph_replay(self, attn_inputs):
        """Update in-place for CUDA graph replay."""
        batch_size = attn_inputs.sequence_lengths.size(0)
        if (
            self.fmha_params is None
            or self._graph_cache_seqlens is None
            or self._graph_cu_seqlens_q is None
            or batch_size != self._batch_size
        ):
            raise RuntimeError("FA3 decode CUDA graph replay was not prepared")

        self._refresh_graph_cache_seqlens(attn_inputs)
        max_seqlen_k = self._get_max_seqlen_k(attn_inputs, self._graph_cache_seqlens)
        self._scheduler_metadata = self._refresh_graph_scheduler_metadata(
            batch_size,
            max_seqlen_k,
            self._graph_cache_seqlens,
            self._graph_cu_seqlens_q,
        )

    def set_params(self, params):
        self.fmha_params = params

    def _split_kv_cache(self, kv_cache):
        from rtp_llm.models_py.modules.factory.attention import common

        paged_kv = kv_cache.kv_cache_base
        paged_kv = common.reshape_paged_kv_cache(
            paged_kv, self.local_kv_head_num, self.seq_size_per_block, self.head_dim
        )
        # [blocks, kv_heads, block_size, head_dim] -> [blocks, block_size, kv_heads, head_dim]
        return (
            paged_kv[:, 0].transpose(1, 2),
            paged_kv[:, 1].transpose(1, 2),
        )

    def forward(self, q, kv_cache, params):
        """Execute FA3 paged decode attention.

        Args:
            q: [batch_size, num_heads * head_dim] or [batch_size, num_heads, head_dim]
            kv_cache: LayerKVCache with paged kv_cache_base
            params: _FA3Params
        """
        fa3 = _import_fa3_interface()

        if self.enable_cuda_graph and q.shape[0] != self._batch_size:
            raise RuntimeError(
                f"FA3 decode graph expected {self._batch_size} query rows, got {q.shape[0]}"
            )

        # The PPU FA3 kvcache path consumes flattened varlen Q:
        # [total_q, num_heads, head_dim] with cu_seqlens_q.
        q = _maybe_contiguous(q.reshape(-1, self.local_head_num, self.head_dim))

        k_cache, v_cache = self._split_kv_cache(kv_cache)
        out = fa3.flash_attn_with_kvcache(
            q=q,
            k_cache=k_cache,
            v_cache=v_cache,
            cache_seqlens=params.cache_seqlens,
            page_table=params.block_table,
            cu_seqlens_q=params.cu_seqlens_q,
            softmax_scale=self.softmax_scale,
            causal=True,
            max_seqlen_q=1,
            max_seqlen_k=params.max_seqlen_k,
            scheduler_metadata=params.scheduler_metadata,
            num_splits=0,
        )

        # Reshape output: [batch_size, num_heads * head_dim]
        return out.reshape(-1, self.local_head_num * self.head_dim)


class FA3PrefillPagedAttnOp:
    """FA3-based prefill attention over RTP paged KV cache.

    The existing prefill pipeline already applies RoPE and writes current K/V into
    the paged cache. This op only replaces the attention read path with the PPU
    FA3 kvcache API.
    """

    def __init__(self, attn_configs, attn_inputs):
        self.local_head_num = attn_configs.head_num
        self.local_kv_head_num = attn_configs.kv_head_num
        self.head_dim = attn_configs.size_per_head
        self.seq_size_per_block = attn_configs.kernel_tokens_per_block
        self.softmax_scale = 1.0 / (self.head_dim**0.5)
        self.max_seq_len = int(getattr(attn_configs, "max_seq_len", 0) or 0)
        self.fmha_params = None
        self.fa3_params = None
        # CUDA-graph (MTP target-verify) state: a persistent scheduler_metadata buffer
        # that is refreshed in place across replays so the captured kernel keeps a
        # stable pointer (reassigning a fresh tensor would leave the graph stale).
        self.enable_cuda_graph = bool(getattr(attn_inputs, "is_cuda_graph", False))
        self._graph_scheduler_metadata = None
        self._graph_max_seqlen_k = 0
        logging.info(
            "[FA3PrefillPagedAttnOp] init heads=%s kv_heads=%s head_dim=%s block=%s max_seq=%s",
            self.local_head_num,
            self.local_kv_head_num,
            self.head_dim,
            self.seq_size_per_block,
            self.max_seq_len,
        )

    def set_params(self, params):
        self.fmha_params = params

    def _get_scheduler_metadata(
        self,
        batch_size,
        max_seqlen_q,
        max_seqlen_k,
        cache_seqlens,
        cu_seqlens_q,
        qkv_dtype,
    ):
        fa3 = _import_fa3_interface()
        return fa3.get_scheduler_metadata(
            batch_size=batch_size,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            num_heads_q=self.local_head_num,
            num_heads_kv=self.local_kv_head_num,
            headdim=self.head_dim,
            cache_seqlens=cache_seqlens,
            qkv_dtype=qkv_dtype,
            cu_seqlens_q=cu_seqlens_q,
            page_size=self.seq_size_per_block,
            causal=True,
            num_splits=0,
        )

    def _get_max_seqlens(self, attn_inputs):
        max_seqlen_q = int(attn_inputs.input_lengths.max().item())
        if (
            attn_inputs.prefix_lengths is not None
            and attn_inputs.prefix_lengths.numel() > 0
        ):
            max_seqlen_k = int(
                (attn_inputs.input_lengths + attn_inputs.prefix_lengths).max().item()
            )
        else:
            max_seqlen_k = max_seqlen_q
        return max_seqlen_q, max_seqlen_k

    def _is_graph_target_verify(self, attn_inputs) -> bool:
        """True only for MTP target verify replayed by the decode CUDA graph.

        Draft prefill is excluded: it is driven by prefill_cuda_graph_copy_params and
        legitimately carries 0 in inactive batch slots.
        """
        return (
            self.enable_cuda_graph
            and bool(getattr(attn_inputs, "is_target_verify", False))
            and getattr(attn_inputs, "prefill_cuda_graph_copy_params", None) is None
        )

    def _restore_uniform_verify_input_lengths(self, attn_inputs) -> None:
        """Put the capture-time uniform N back into the padded input_lengths rows.

        Target verify replays a fixed graph_bs and every request carries exactly N
        query tokens, so the capture buffer holds input_lengths == N in all rows.
        The runner zeroes the padded rows before calling us, which makes
        fillParamsInternal skip them: batch_indice/positions keep the previous
        replay's values, and a padded row emits no page_indice entry at all, so the
        page lookup walks past decode_page_indptr into stale real block ids.
        Restoring N makes each padded row re-emit its own tokens plus one page; the
        runner already zeroed the block-table padding, so those resolve to the
        reserved block 0.
        """
        if not self._is_graph_target_verify(attn_inputs):
            return
        attn_inputs.input_lengths.fill_(int(attn_inputs.input_lengths.max().item()))

    def prepare(self, attn_inputs, forbid_realloc: bool = False):
        from rtp_llm.ops.compute_ops import get_scalar_type

        if self.fmha_params is None:
            raise RuntimeError(
                "FA3PrefillPagedAttnOp params must be set before prepare()"
            )

        self._restore_uniform_verify_input_lengths(attn_inputs)

        self.fmha_params.fill_params(
            attn_inputs.prefix_lengths,
            attn_inputs.sequence_lengths,
            attn_inputs.input_lengths,
            attn_inputs.kv_cache_kernel_block_id,
            self.seq_size_per_block,
            forbid_realloc,
        )

        batch_size = attn_inputs.input_lengths.size(0)
        cu_seqlens_q = _maybe_contiguous(
            attn_inputs.cu_seqlens_device[: batch_size + 1]
        )
        cache_seqlens = _maybe_contiguous(self.fmha_params.kvlen_d)
        block_table = _maybe_contiguous(attn_inputs.kv_cache_kernel_block_id_device)

        graph_verify = self.enable_cuda_graph and (
            bool(getattr(attn_inputs, "is_target_verify", False))
            or getattr(attn_inputs, "prefill_cuda_graph_copy_params", None) is not None
        )
        if graph_verify:
            # MTP uniform-query prefill under CUDA graph — covers both target-verify and
            # draft-prefill. Every request has a constant N query tokens (N = num_tokens_per_bs),
            # so max_seqlen_q and cu_seqlens (= arange*N) are graph-stable. N is read from
            # input_lengths.max() (the capture buffer is full(num_tokens_per_bs); on the draft
            # path inactive batches are 0 so max() still yields N), avoiding a dependency on
            # prefill_cuda_graph_copy_params for the verify path. cache_seqlens
            # (kvlen_d), cu_seqlens_device and the block table are RTP-persistent device
            # buffers kept in place by fill_params(forbid_realloc=True), so the captured
            # graph re-reads updated content on replay. No small2large copy is needed
            # (FA3 is varlen-native); padding slots [active_bs:max_bs] just produce
            # discarded output rows.
            max_seqlen_q = int(attn_inputs.input_lengths.max().item())
            max_seqlen_k = self.max_seq_len
            if max_seqlen_k <= 0:
                # capture-time fallback only (one sync); reused across replays
                max_seqlen_k = self._graph_max_seqlen_k or int(
                    cache_seqlens.max().item()
                )
            self._graph_max_seqlen_k = max_seqlen_k
            meta = self._get_scheduler_metadata(
                batch_size,
                max_seqlen_q,
                max_seqlen_k,
                cache_seqlens,
                cu_seqlens_q,
                get_scalar_type(attn_inputs.dtype),
            )
            if (
                self._graph_scheduler_metadata is None
                or self._graph_scheduler_metadata.shape != meta.shape
            ):
                self._graph_scheduler_metadata = meta  # first (capture) allocation
            else:
                self._graph_scheduler_metadata.copy_(
                    meta
                )  # in-place refresh for replay
            self.fa3_params = _FA3PrefillParams(
                batch_size=batch_size,
                cu_seqlens_q=cu_seqlens_q,
                cache_seqlens=cache_seqlens,
                block_table=block_table,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_k=max_seqlen_k,
                scheduler_metadata=self._graph_scheduler_metadata,
            )
            return self.fmha_params

        max_seqlen_q, max_seqlen_k = self._get_max_seqlens(attn_inputs)
        scheduler_metadata = self._get_scheduler_metadata(
            batch_size,
            max_seqlen_q,
            max_seqlen_k,
            cache_seqlens,
            cu_seqlens_q,
            get_scalar_type(attn_inputs.dtype),
        )
        self.fa3_params = _FA3PrefillParams(
            batch_size=batch_size,
            cu_seqlens_q=cu_seqlens_q,
            cache_seqlens=cache_seqlens,
            block_table=block_table,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            scheduler_metadata=scheduler_metadata,
        )
        return self.fmha_params

    def prepare_for_cuda_graph_replay(self, attn_inputs):
        """Refresh graph-stable params in place for a CUDA-graph replay (verify path)."""
        self.prepare(attn_inputs, forbid_realloc=True)

    @staticmethod
    def support(attn_inputs) -> bool:
        return True

    def _split_kv_cache(self, kv_cache):
        from rtp_llm.models_py.modules.factory.attention import common

        paged_kv = common.reshape_paged_kv_cache(
            kv_cache.kv_cache_base,
            self.local_kv_head_num,
            self.seq_size_per_block,
            self.head_dim,
        )
        return (
            paged_kv[:, 0].transpose(1, 2),
            paged_kv[:, 1].transpose(1, 2),
        )

    def forward(self, q, kv_cache):
        if self.fa3_params is None:
            raise RuntimeError(
                "FA3PrefillPagedAttnOp.prepare() must run before forward()"
            )
        if kv_cache is None:
            raise RuntimeError("FA3 prefill paged attention requires kv_cache")
        if q.dim() != 3:
            raise ValueError(f"Expected q [tokens, heads, dim], got dim={q.dim()}")

        fa3 = _import_fa3_interface()
        k_cache, v_cache = self._split_kv_cache(kv_cache)
        params = self.fa3_params
        return fa3.flash_attn_with_kvcache(
            q=_maybe_contiguous(q),
            k_cache=k_cache,
            v_cache=v_cache,
            cache_seqlens=params.cache_seqlens,
            page_table=params.block_table,
            cu_seqlens_q=params.cu_seqlens_q,
            softmax_scale=self.softmax_scale,
            causal=True,
            max_seqlen_q=params.max_seqlen_q,
            max_seqlen_k=params.max_seqlen_k,
            scheduler_metadata=params.scheduler_metadata,
            num_splits=0,
        )


class _FA3Params:
    """Simple parameter container for FA3 decode."""

    def __init__(
        self,
        batch_size,
        cache_seqlens,
        block_table,
        max_seqlen_k,
        cu_seqlens_q,
        scheduler_metadata,
    ):
        self.batch_size = batch_size
        self.cache_seqlens = cache_seqlens
        self.block_table = block_table
        self.max_seqlen_k = max_seqlen_k
        self.cu_seqlens_q = cu_seqlens_q
        self.scheduler_metadata = scheduler_metadata


class _FA3PrefillParams:
    """Simple parameter container for FA3 prefill."""

    def __init__(
        self,
        batch_size,
        cu_seqlens_q,
        cache_seqlens,
        block_table,
        max_seqlen_q,
        max_seqlen_k,
        scheduler_metadata,
    ):
        self.batch_size = batch_size
        self.cu_seqlens_q = cu_seqlens_q
        self.cache_seqlens = cache_seqlens
        self.block_table = block_table
        self.max_seqlen_q = max_seqlen_q
        self.max_seqlen_k = max_seqlen_k
        self.scheduler_metadata = scheduler_metadata


# ---------------------------------------------------------------------------
# FA3 Decode FMHAImplBase Implementation
# ---------------------------------------------------------------------------


class FA3DecodeImpl:
    """FMHAImplBase-compatible FA3 decode implementation for PPU.

    Combines PpuMRopeOp for MRoPE (or FusedRopeKVCacheDecodeOp for other RoPE
    styles) with FA3 attention.
    """

    # The shared factory probes this capability before calling the constructor.
    accepts_fmha_config = False

    def __init__(self, attn_configs, attn_inputs, parallelism_config=None):
        from rtp_llm.models_py.modules.factory.attention.common import (
            create_write_cache_store_impl,
        )
        from rtp_llm.models_py.modules.factory.attention.cuda_impl.kv_cache_write_op import (
            KVCacheWriteOp,
        )
        from rtp_llm.models_py.modules.factory.attention.cuda_impl.py_flashinfer_mha import (
            FusedRopeKVCacheDecodeOp,
        )
        from rtp_llm.ops import RopeStyle
        from rtp_llm.ops.compute_ops import rtp_llm_ops
        from rtp_llm.platforms.ppu.modules.attention.ppu_mrope import PpuMRopeOp

        self.need_rope_kv_cache = attn_configs.need_rope_kv_cache
        self.fmha_impl = FA3DecodeAttnOp(attn_configs, attn_inputs)
        self.attn_configs = attn_configs
        self.attn_inputs = attn_inputs
        # Qwen3.5 MRoPE always uses the graph-safe PpuMRopeOp (precomputed
        # device cos/sin cache + gather; no positions.cpu() / JIT during capture).
        self.use_ppu_mrope = attn_configs.rope_config.style == RopeStyle.Mrope

        # Prepare initial params
        self.fmha_params = self.fmha_impl.prepare(attn_inputs)
        if self.use_ppu_mrope:
            self.rope_impl = PpuMRopeOp(attn_configs)
            self.kv_cache_write_op = KVCacheWriteOp(
                num_kv_heads=attn_configs.kv_head_num,
                head_size=attn_configs.size_per_head,
                token_per_block=attn_configs.kernel_tokens_per_block,
            )
            self.rope_params = rtp_llm_ops.FlashInferMlaAttnParams()
            self.rope_params.fill_params(
                attn_inputs.prefix_lengths,
                attn_inputs.sequence_lengths,
                attn_inputs.input_lengths,
                attn_inputs.kv_cache_kernel_block_id,
                attn_configs.kernel_tokens_per_block,
            )
            self.rope_impl.set_params(self.rope_params)
            self.kv_cache_write_op.set_params(self.rope_params)
        else:
            self.rope_impl = FusedRopeKVCacheDecodeOp(attn_configs)
            self.kv_cache_write_op = None
            self.rope_params = self.rope_impl.prepare(attn_inputs)
        self.write_cache_store_impl = create_write_cache_store_impl(attn_inputs)

    @classmethod
    def support(cls, attn_configs, attn_inputs) -> bool:
        """Check if FA3 decode is supported."""
        # Env switch
        if os.environ.get("PPU_USE_FA3_DECODE", "1") == "0":
            logging.debug("[FA3DecodeImpl.support] disabled by env")
            return False
        # MLA not supported
        if attn_configs.use_mla:
            logging.debug("[FA3DecodeImpl.support] MLA not supported")
            return False
        # Must be decode (not prefill)
        if attn_inputs.is_prefill:
            logging.debug("[FA3DecodeImpl.support] is_prefill=True, skip")
            return False
        # FA3 must be available
        result = _check_fa3_available()
        if not result:
            logging.info("[FA3DecodeImpl.support] flash_attn_3 not available")
        else:
            logging.info("[FA3DecodeImpl.support] SUPPORTED - will use FA3")
        return result

    def support_cuda_graph(self) -> bool:
        return True

    def prepare_cuda_graph(self, attn_inputs):
        """Prepare FA3/RoPE buffers for CUDA graph replay."""
        self.fmha_impl.prepare_for_cuda_graph_replay(attn_inputs)
        if self.use_ppu_mrope:
            self.rope_params.fill_params(
                attn_inputs.prefix_lengths,
                attn_inputs.sequence_lengths,
                attn_inputs.input_lengths,
                attn_inputs.kv_cache_kernel_block_id,
                self.attn_configs.kernel_tokens_per_block,
                forbid_realloc=True,
            )
        else:
            from rtp_llm.models_py.modules.factory.attention import common

            new_rope_params = self.rope_impl.prepare(attn_inputs)
            common.copy_kv_cache_offset(
                self.rope_params.kv_cache_offset, new_rope_params.kv_cache_offset
            )

    @classmethod
    def support_parallelism_config(cls, parallelism_config) -> bool:
        # Decode is phase-agnostic: every rank holds the full KV for its own
        # sequences, so a decode step is an ordinary paged decode whatever the
        # prefill side did. Allow it regardless of parallelism config.
        return True

    def forward(self, qkv, kv_cache, layer_idx=0):
        """Execute RoPE + KV cache write + FA3 attention."""
        from rtp_llm.models_py.modules.factory.attention import common

        # Apply RoPE and KV Cache processing
        if self.need_rope_kv_cache:
            if self.use_ppu_mrope:
                query, key, value = self.rope_impl.forward(qkv)
                self.kv_cache_write_op.forward(key, value, kv_cache)
                qkv = query
            else:
                qkv = self.rope_impl.forward(qkv, kv_cache, self.rope_params)
        # Apply write cache store if needed
        common.apply_write_cache_store(
            self.write_cache_store_impl, self.attn_inputs, kv_cache
        )

        # Execute FA3 forward
        return self.fmha_impl.forward(qkv, kv_cache, self.fmha_params)


class FA3PrefillPagedImpl:
    """FMHAImplBase-compatible FA3 paged prefill implementation for PPU."""

    # The shared factory probes this capability before calling the constructor.
    accepts_fmha_config = False

    def __init__(self, attn_configs, attn_inputs, parallelism_config=None):
        from rtp_llm.models_py.modules.factory.attention import common
        from rtp_llm.models_py.modules.factory.attention.cuda_impl.flashinfer_rotary_emb import (
            MhaRotaryEmbeddingOp,
        )
        from rtp_llm.models_py.modules.factory.attention.cuda_impl.kv_cache_write_op import (
            KVCacheWriteOp,
        )
        from rtp_llm.ops import RopeStyle
        from rtp_llm.ops.compute_ops import rtp_llm_ops
        from rtp_llm.platforms.ppu.modules.attention.ppu_mrope import PpuMRopeOp

        self.need_rope_kv_cache = attn_configs.need_rope_kv_cache
        self.attn_configs = attn_configs
        self.attn_inputs = attn_inputs
        self.fmha_impl = FA3PrefillPagedAttnOp(attn_configs, attn_inputs)
        # MRoPE (Qwen3.5) shares the same graph-safe PpuMRopeOp as decode;
        # other rope styles keep the standard MhaRotaryEmbeddingOp.
        _style = attn_configs.rope_config.style
        if _style == RopeStyle.No:
            self.rope_impl = None
        elif _style == RopeStyle.Mrope:
            self.rope_impl = PpuMRopeOp(attn_configs)
        else:
            self.rope_impl = MhaRotaryEmbeddingOp(attn_configs)
        self.kv_cache_write_op = KVCacheWriteOp(
            num_kv_heads=attn_configs.kv_head_num,
            head_size=attn_configs.size_per_head,
            token_per_block=attn_configs.kernel_tokens_per_block,
        )
        self.fmha_params = rtp_llm_ops.FlashInferMlaAttnParams()
        self.rope_params = self.fmha_params
        self.fmha_impl.set_params(self.fmha_params)
        if self.rope_impl is not None:
            self.rope_impl.set_params(self.rope_params)
        self.kv_cache_write_op.set_params(self.rope_params)
        self.fmha_impl.prepare(attn_inputs)
        self.write_cache_store_impl = common.create_write_cache_store_impl(attn_inputs)

    @classmethod
    def support(cls, attn_configs, attn_inputs) -> bool:
        if os.environ.get("PPU_USE_FA3_PREFILL", "1") == "0":
            logging.debug("[FA3PrefillPagedImpl.support] disabled by env")
            return False
        if not attn_inputs.is_prefill:
            return False
        if getattr(attn_configs, "use_mla", False):
            logging.debug("[FA3PrefillPagedImpl.support] MLA not supported")
            return False
        is_graph = bool(getattr(attn_inputs, "is_cuda_graph", False))
        is_verify = bool(getattr(attn_inputs, "is_target_verify", False))
        has_copy = (
            getattr(attn_inputs, "prefill_cuda_graph_copy_params", None) is not None
        )
        if is_graph:
            # Under CUDA graph, support the two MTP uniform-query prefill paths, both of
            # which have a constant num_tokens_per_bs query tokens per seq (so max_seqlen_q
            # and cu_seqlens are graph-stable, derived from input_lengths in prepare()):
            #   - target verify  : is_target_verify=True  (decode-capture path; copy_params
            #                       stays unset so FlashInfer's own path is unaffected)
            #   - MTP draft prefill: prefill_cuda_graph_copy_params set (prefill-capture path,
            #                       is_target_verify=False). FA3 is varlen-native so it does
            #                       not hit FlashInfer's q.size(0)==qo_indptr[-1] restriction,
            #                       which is why FlashInfer needed DISABLE_MTP_DRAFT_CUDA_GRAPH.
            # Any other graph-prefill shape (e.g. embedding full-seq) falls back to FlashInfer.
            if os.environ.get("PPU_USE_FA3_VERIFY", "1") == "0":
                logging.debug(
                    "[FA3PrefillPagedImpl.support] FA3 graph-prefill disabled by env"
                )
                return False
            if not (is_verify or has_copy):
                logging.debug(
                    "[FA3PrefillPagedImpl.support] graph but not verify/draft-prefill, skip"
                )
                return False
        block_table = getattr(attn_inputs, "kv_cache_kernel_block_id_device", None)
        if block_table is None:
            logging.debug("[FA3PrefillPagedImpl.support] missing block table")
            return False
        result = _check_fa3_available()
        if result:
            logging.info(
                "[FA3PrefillPagedImpl.support] SUPPORTED - will use FA3 (graph=%s)",
                is_graph,
            )
        else:
            logging.info("[FA3PrefillPagedImpl.support] flash_attn_3 not available")
        return result

    def support_cuda_graph(self) -> bool:
        # Only the MTP target-verify path reaches here under a graph (gated in support()).
        return True

    def prepare_cuda_graph(self, attn_inputs):
        """Refresh FA3 params (and the shared RoPE/KV-write params) in place for replay.

        RoPE and KV-cache-write ops share ``self.fmha_params`` (set_params above), so
        refreshing the FA3 op via fill_params(forbid_realloc=True) updates them in place too.
        Padded host block-table rows already point at the reserved block 0 by the time the
        runner calls this, so the rebuilt slot_mapping keeps dummy KV off real blocks.
        """
        self.fmha_impl.prepare_for_cuda_graph_replay(attn_inputs)

    @classmethod
    def support_parallelism_config(cls, parallelism_config) -> bool:
        if parallelism_config is None:
            return True
        return not parallelism_config.prefill_cp_config.is_enabled()

    def _split_qkv(self, qkv):
        qkv = qkv.reshape(qkv.shape[0], -1)
        num_heads = self.attn_configs.head_num
        num_kv_heads = self.attn_configs.kv_head_num
        head_dim = self.attn_configs.size_per_head
        q, k, v = qkv.split(
            [
                head_dim * num_heads,
                head_dim * num_kv_heads,
                head_dim * num_kv_heads,
            ],
            dim=-1,
        )
        return (
            q.reshape(q.shape[0], num_heads, head_dim),
            k.reshape(k.shape[0], num_kv_heads, head_dim),
            v.reshape(v.shape[0], num_kv_heads, head_dim),
        )

    def forward(self, qkv, kv_cache, layer_idx=0):
        from rtp_llm.models_py.modules.factory.attention import common

        if self.need_rope_kv_cache:
            if self.rope_impl is not None:
                query, key, value = self.rope_impl.forward(qkv)
            else:
                query, key, value = self._split_qkv(qkv)
            self.kv_cache_write_op.forward(key, value, kv_cache)
            fmha_input = query
        else:
            fmha_input = qkv

        common.apply_write_cache_store(
            self.write_cache_store_impl, self.attn_inputs, kv_cache
        )
        return self.fmha_impl.forward(fmha_input, kv_cache)

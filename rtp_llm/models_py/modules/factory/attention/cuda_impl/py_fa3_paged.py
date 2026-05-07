"""FlashAttention 3 based paged-attention op + Impl wrappers for MTP target
verify and draft prefill CG paths.

Single-file home for all FA3 attention pieces:

* ``PyFA3PagedAttnOp`` — the shared op (used by both target verify and
  draft prefill).  Was previously duplicated across two files with ~95%
  identical bodies; merged after the 2026-05-06 H2D capture fix made the
  two refresh paths identical.
* ``PyFA3TargetVerifyImpl`` — wrapper Impl + optional debug dump
  infrastructure used to diagnose past CG correctness bugs.
* ``PyFA3DraftPrefillImpl`` — wrapper Impl for the draft-prefill CG path.

The two ``Impl`` classes only differ in their ``support()`` predicates and
in target verify keeping a debug-dump override on ``forward()``.

Why FA3 over the legacy FlashInfer plan/run path:

* FA3 ``flash_attn_with_kvcache`` is a single fused launch with no
  ``plan/run`` split, so there are no captured-time scheduling scalars
  that must match between capture and replay (root cause of the
  non-determinism in ``project_target_verify_cg_bug`` and
  ``project_draft_prefill_cg_root_cause``);
* page_table is read directly from
  ``attn_inputs.kv_cache_kernel_block_id_device``, the C++-side stable CG
  buffer maintained by ``CudaGraphRunner::prepareInputs``;
* cache_seqlens and cu_seqlens_q live in op-owned stable CUDA buffers,
  refreshed each ``prepare()`` from a stable pinned host scratch — both
  source and destination keep their data_ptrs across replays so the
  captured H2D copy never reads invalidated memory.

Q layout:

* Target verify: every active request occupies exactly
  ``num_tokens_per_bs`` rows; padding requests have input_length=0 and so
  cu_seqlens_q plateaus at the active total.
* Draft prefill: variable per-request Q length up to
  ``num_tokens_per_bs``; rows ``[0, sum(input_lengths))`` are real, rest
  are stale.

In both cases ``cu_seqlens_q = [0, cumsum(input_lengths)]`` with padding
entries equal to the active total — FA3 sees padding batches as 0-length
ranges and skips them.
"""

import os
import threading
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

# =============================================================================
# Shared FA3 paged attention op
# =============================================================================


class PyFA3PagedAttnOp(object):
    """FA3 paged attention op shared by MTP target verify and draft prefill.

    ``forward`` contract:

    * ``q``: ``[total_q_tokens, num_heads, head_dim]`` packed varlen
      tensor.  In CUDA-graph mode the caller pads ``total_q_tokens`` to
      ``max_bs * num_tokens_per_bs``; padding rows are quietly skipped via
      ``cu_seqlens_q`` plateauing at the active total.
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

        # Stable CUDA buffers read by the captured FA3 kernel each replay.
        self._cache_seqlens_buf: Optional[torch.Tensor] = None
        self._cu_seqlens_q_buf: Optional[torch.Tensor] = None
        # Stable pinned host scratch backing the captured H2D copies
        # above. Without stable source data_ptrs, captured H2D ops would
        # DMA from transient `.to('cuda')` tensors that get freed after
        # capture (see project_fa3_h2d_capture_fix memory).
        self._cache_seqlens_host_scratch: Optional[torch.Tensor] = None
        self._cu_seqlens_q_host_scratch: Optional[torch.Tensor] = None
        # 2-D page_table; aliased to the C++ stable CG buffer
        # `kv_cache_kernel_block_id_device` (refreshed each prepareInputs
        # via tryAddStridedD2DCopy at cuda_graph_runner.cc:147-148).
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
        # input_lengths.max() at first prepare matches num_tokens_per_bs for
        # both target verify (uniform) and draft prefill (capture-time max).
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
        self._cache_seqlens_host_scratch = torch.zeros(
            batch_size, dtype=torch.int32, pin_memory=True
        )
        self._cu_seqlens_q_host_scratch = torch.zeros(
            batch_size + 1, dtype=torch.int32, pin_memory=True
        )
        self._page_table_ref = attn_inputs.kv_cache_kernel_block_id_device
        self._cuda_graph_initialized = True

    def _refresh_cg_buffers(self, attn_inputs: PyAttentionInputs) -> None:
        """Refresh stable CUDA buffers from the live host-pinned tensors.

        ``input_lengths`` and ``prefix_lengths`` are CPU-pinned stable CG
        buffers refreshed by ``CudaGraphRunner::prepareInputs`` H2H copies
        each replay (cuda_graph_runner.cc:211-217).  Padding rows are
        zeroed by the runner.

        We use the host versions (rather than ``*_d``) because for
        ``is_prefill_cuda_graph_mode_=True`` (draft prefill) the runner
        does NOT refresh ``prefix_lengths_d`` via D2D (the D2D block is
        gated on ``!is_prefill_cuda_graph_mode_``).  Same compute path
        works for target verify.
        """
        assert self._cache_seqlens_buf is not None
        assert self._cu_seqlens_q_buf is not None
        assert self._cache_seqlens_host_scratch is not None
        assert self._cu_seqlens_q_host_scratch is not None
        n = min(self._fixed_batch_size, self._cache_seqlens_buf.size(0))

        cs_scratch = self._cache_seqlens_host_scratch
        cu_scratch = self._cu_seqlens_q_host_scratch
        prefix_h = attn_inputs.prefix_lengths[:n].to(torch.int32)
        input_h = attn_inputs.input_lengths[:n].to(torch.int32)

        # cache_seqlens = prefix + input (KV positions visible to attention).
        # Padding rows are 0 because both prefix_lengths and input_lengths
        # are zeroed for padding batches by the C++ runner.
        torch.add(prefix_h, input_h, out=cs_scratch[:n])
        if n < cs_scratch.size(0):
            cs_scratch[n:].zero_()
        self._cache_seqlens_buf.copy_(cs_scratch, non_blocking=True)

        # cu_seqlens_q = [0, cumsum(input_lengths)]
        # Active entries cover [0, total_active_tokens); padding entries
        # plateau at total_active_tokens because input_h[active:] is 0
        # (cumsum stays flat).  FA3 sees padding batches as 0-length q
        # ranges and skips them.
        cu_scratch[0].zero_()
        torch.cumsum(input_h, dim=0, out=cu_scratch[1 : n + 1])
        if n < self._fixed_batch_size:
            cu_scratch[n + 1 :].fill_(cu_scratch[n])
        self._cu_seqlens_q_buf.copy_(cu_scratch, non_blocking=True)

    def _build_cu_seqlens_q_nocg(
        self, attn_inputs: PyAttentionInputs, batch_size: int
    ) -> torch.Tensor:
        """Eager-mode cu_seqlens_q from live ``cu_seqlens`` (CUDA, populated
        by ``PyWrappedModel::buildPyAttentionInputs`` via cumsum).

        Falls back to a host cumsum if the live tensor is somehow missing
        or undersized.
        """
        cu = getattr(attn_inputs, "cu_seqlens", None)
        if cu is not None and cu.numel() >= batch_size + 1:
            cu_slice = cu[: batch_size + 1].to(torch.int32)
            if cu_slice.device.type != "cuda":
                cu_slice = cu_slice.to(device="cuda", non_blocking=True)
            return cu_slice
        input_cpu = attn_inputs.input_lengths.to(torch.int32).cpu()
        cu_cpu = torch.zeros(batch_size + 1, dtype=torch.int32)
        torch.cumsum(input_cpu, dim=0, out=cu_cpu[1:])
        return cu_cpu.to(device="cuda", non_blocking=True)

    def prepare(
        self, attn_inputs: PyAttentionInputs, forbid_realloc: bool = False
    ) -> ParamsBase:
        check_attention_inputs(attn_inputs)
        # fill_params still required: KVCacheWriteOp / RoPE op consume
        # batch_indice_d, positions_d, page_indice_d, decode_page_indptr_d,
        # paged_kv_last_page_len_d, kvlen_d from the shared
        # FlashInferMlaAttnParams.  We do NOT bind those tensors as
        # captured FA3 kernel inputs — FA3 reads its own stable buffers
        # below.
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
            cs_host = attn_inputs.prefix_lengths[:batch_size].to(
                torch.int32
            ) + attn_inputs.input_lengths[:batch_size].to(torch.int32)
            self._cache_seqlens_buf = cs_host.to(device="cuda", non_blocking=True)
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
        assert kv_cache is not None, "kv_cache is required for FA3 paged attn"
        assert (
            q.dim() == 3
        ), f"Expected q [total_tokens, H, D], got shape {tuple(q.shape)}"
        assert (
            self._cu_seqlens_q_buf is not None
            and self._cache_seqlens_buf is not None
            and self._page_table_ref is not None
        ), "PyFA3PagedAttnOp.prepare() must run before forward()"

        k_cache, v_cache = self._get_kv_caches(kv_cache)
        # ``pack_gqa=True`` lands on FA3's PackGQA fast path: with a high
        # GQA ratio (Qwen3.5: 8 q-heads / 1 kv-head per TP rank) and a tiny
        # per-request Q (≤ 4 draft tokens), packing the q heads into the
        # same tile cuts kernel time ~16x in our timeline (matches what
        # SGLang's FA3 backend selects via `pack_gqa=true` heuristics on
        # Hopper, see `VarlenDynamicPersistentTileScheduler<...,true,...>`).
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


# =============================================================================
# Optional debug dump (target verify only — used to diagnose past CG bugs)
#
# Enable via env vars (all gated by FA3_TV_DUMP_DIR being set):
#   FA3_TV_DUMP_DIR        — directory to write .pt dumps to
#   FA3_TV_DUMP_MAX_CALLS  — cap on dumps per process (default 200)
#   FA3_TV_DUMP_RING       — when 1, ring-buffer mode: oldest dumps deleted
#                            once counter > max_calls so we always keep the
#                            most recent N (good for long runs that may
#                            accumulate enough state to trigger garbled).
#
# The dump path activates only when FA3_TV_DUMP_DIR is set; production
# inference is unaffected.  Workflow:
#   1) Run smoke test with FA3_TV_DUMP_DIR=/tmp/fa3_off + CG OFF
#      (DISABLE_SP_TARGET_VERIFY_CUDA_GRAPH=1)
#   2) Run smoke test with FA3_TV_DUMP_DIR=/tmp/fa3_on + CG ON
#   3) Use compare_fa3_tv_dumps.py to pair calls by cache_seqlens
#      fingerprint and report first divergent tensor.
# =============================================================================
_FA3_DUMP_DIR = os.environ.get("FA3_TV_DUMP_DIR")
_FA3_DUMP_MAX_CALLS = int(os.environ.get("FA3_TV_DUMP_MAX_CALLS", "200"))
_FA3_DUMP_RING = os.environ.get("FA3_TV_DUMP_RING", "0") == "1"
_FA3_DUMP_LOCK = threading.Lock()
_fa3_dump_counter = 0


def _fa3_dump_enabled() -> bool:
    if _FA3_DUMP_DIR is None:
        return False
    if not _FA3_DUMP_RING and _fa3_dump_counter >= _FA3_DUMP_MAX_CALLS:
        return False
    # Never dump from inside a CUDA graph capture: the .cpu().clone() copies
    # we do are not allowed during stream capture and will throw
    # "Offset increment outside graph capture encountered unexpectedly".
    if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
        return False
    return True


def _fa3_dump(tag: str, payload: "dict[str, Any]") -> None:
    """Atomic-ish counter-bumping save.  Counter is shared across tag types
    so prepare/forward/kvwrite calls all share the same global ordering.

    Synchronizes the device before reading tensors so the dumped values
    reflect what the next captured-kernel replay would actually see (and
    not pre-refresh stale state on a different stream).
    """
    global _fa3_dump_counter
    if _FA3_DUMP_DIR is None:
        return
    with _FA3_DUMP_LOCK:
        if not _FA3_DUMP_RING and _fa3_dump_counter >= _FA3_DUMP_MAX_CALLS:
            return
        idx = _fa3_dump_counter
        _fa3_dump_counter += 1
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    rank = int(os.environ.get("WORLD_RANK", os.environ.get("RANK", "0")))
    out_dir = _FA3_DUMP_DIR
    os.makedirs(out_dir, exist_ok=True)
    if _FA3_DUMP_RING and idx >= _FA3_DUMP_MAX_CALLS:
        old_idx = idx - _FA3_DUMP_MAX_CALLS
        for old_tag in ("prepare", "forward", "kvwrite"):
            old_path = os.path.join(out_dir, f"rank{rank}_{old_idx:06d}_{old_tag}.pt")
            try:
                os.unlink(old_path)
            except OSError:
                pass
    out_path = os.path.join(out_dir, f"rank{rank}_{idx:06d}_{tag}.pt")
    safe_payload: "dict[str, Any]" = {"call_idx": idx, "tag": tag}
    for k, v in payload.items():
        if isinstance(v, torch.Tensor):
            safe_payload[k] = v.detach().cpu().clone()
        else:
            safe_payload[k] = v
    torch.save(safe_payload, out_path)


def _snapshot_fmha_params(params: Any) -> "dict[str, Any]":
    """Snapshot the fmha_params tensors that get consumed by KVCacheWriteOp
    and downstream attention.  Records both data_ptr (to detect captured-
    kernel aliasing if the param tensor gets replaced between replays) and
    tensor contents (to verify the live values are correct)."""
    snap: "dict[str, Any]" = {}
    if params is None:
        return snap
    for name in (
        "batch_indice_d",
        "positions_d",
        "page_indice_d",
        "decode_page_indptr_d",
        "paged_kv_last_page_len_d",
        "kvlen_d",
    ):
        try:
            t = getattr(params, name)
        except Exception:
            continue
        if isinstance(t, torch.Tensor):
            snap[f"fmha_params_{name}"] = t
            snap[f"fmha_params_{name}_data_ptr"] = int(t.data_ptr())
    return snap


# =============================================================================
# Wrapper Impls (factory entry points)
# =============================================================================


class _PyFA3ImplBase(PyFlashinferPrefillImplBase):
    """Common wrapper plumbing for both FA3 Impl classes (target verify and
    draft prefill).  Subclasses only override ``support()``; target verify
    additionally overrides ``forward()`` to attach the optional dump path.
    """

    def _create_fmha_impl(
        self, attn_configs: AttentionConfigs, attn_inputs: PyAttentionInputs
    ) -> Any:
        return PyFA3PagedAttnOp(attn_configs, attn_inputs)

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


class PyFA3TargetVerifyImpl(_PyFA3ImplBase):
    """Wrapper Impl wiring the shared FA3 paged op into the factory for the
    MTP target-verify CG path.

    Owns an optional ``forward()`` override that snapshots KV-write inputs
    and cache pages when ``FA3_TV_DUMP_DIR`` is set — falls through to the
    parent forward when dumping is disabled (production-unaffected).
    """

    def forward(
        self,
        qkv: torch.Tensor,
        kv_cache: Optional[LayerKVCache],
        layer_idx: int = 0,
    ) -> torch.Tensor:
        """Override the base wrapper forward so we can dump KV inputs +
        the cache state at target pages BEFORE/AFTER the KV write.

        When dumping is disabled, falls through to the parent forward.
        """
        if not _fa3_dump_enabled() or not self.need_rope_kv_cache:
            return super().forward(qkv, kv_cache, layer_idx=layer_idx)

        # Inline the parent's forward path with KV write hooks.
        if self.rope_impl is not None:
            query, key, value = self.rope_impl.forward(qkv)
        else:
            query, key, value = self._split_qkv(qkv)

        # Snapshot cache pages that the KV write op is about to touch.
        # We use fmha_params.page_indice_d to know which pages.
        params = self.fmha_params
        before_pages: "dict[str, Any]" = {}
        if (
            kv_cache is not None
            and isinstance(getattr(params, "page_indice_d", None), torch.Tensor)
            and params.page_indice_d.numel() > 0
        ):
            page_ids = params.page_indice_d.detach().cpu()
            unique_pages = torch.unique(page_ids).tolist()[:8]  # cap at 8 for size
            paged = kv_cache.kv_cache_base
            for pid in unique_pages:
                if 0 <= pid < paged.size(0):
                    # K page only (V is symmetric); checksum to keep dumps small.
                    k_page = paged[int(pid), 0]
                    before_pages[f"page{int(pid)}_k_before_norm"] = float(
                        k_page.float().norm().item()
                    )

        # Run the KV write — captured in CG mode, eager otherwise.
        self.kv_cache_write_op.forward(key, value, kv_cache)

        # Snapshot AFTER write — confirms write actually happened.
        after_pages: "dict[str, Any]" = {}
        if before_pages and kv_cache is not None:
            paged = kv_cache.kv_cache_base
            for k_name in before_pages:
                pid = int(k_name.split("_")[0][len("page") :])
                if 0 <= pid < paged.size(0):
                    after_pages[k_name.replace("_before_", "_after_")] = float(
                        paged[pid, 0].float().norm().item()
                    )

        kv_dump: "dict[str, Any]" = {
            "mode": "cg" if self.fmha_impl.enable_cuda_graph else "eager",
            "layer_idx": int(layer_idx),
            "key_norm": float(key.float().norm().item()),
            "value_norm": float(value.float().norm().item()),
            "key_data_ptr": int(key.data_ptr()),
            "value_data_ptr": int(value.data_ptr()),
        }
        kv_dump.update(before_pages)
        kv_dump.update(after_pages)
        kv_dump.update(_snapshot_fmha_params(params))
        _fa3_dump("kvwrite", kv_dump)

        qkv = self._prepare_fmha_input(query, key, value)
        common.apply_write_cache_store(
            self.write_cache_store_impl, self.attn_inputs, kv_cache
        )
        return self.fmha_impl.forward(qkv, kv_cache)

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


class PyFA3DraftPrefillImpl(_PyFA3ImplBase):
    """Wrapper Impl wiring the shared FA3 paged op into the factory for the
    MTP draft-prefill CG path."""

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
        # Embedding model: max_seq_len == num_tokens_per_bs (single-shot
        # prefill over a fixed input window).  We only target draft
        # prefill, where max_seq_len > num_tokens_per_bs.
        max_sl = getattr(copy_params, "max_seq_len", None)
        if max_sl is not None and max_sl >= attn_configs.max_seq_len:
            return False
        return True

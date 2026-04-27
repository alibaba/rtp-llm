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

# ---------------------------------------------------------------------------
# CG-vs-eager debug dumping infrastructure.
#
# Enable via env vars (all gated by FA3_TV_DUMP_DIR being set):
#   FA3_TV_DUMP_DIR        — directory to write .pt dumps to
#   FA3_TV_DUMP_MAX_CALLS  — cap on dumps per process (default 200)
#   FA3_TV_DUMP_RING       — when 1, ring-buffer mode: oldest dumps deleted
#                            once counter > max_calls so we always keep the
#                            most recent N (good for long runs that may
#                            accumulate enough state to trigger garbled).
#
# Per forward pass we dump:
#   prepare.pt — inputs + fmha_params snapshot (data_ptrs + tensor contents)
#                fires at every replay-prep in CG mode AND every prepare in
#                eager mode.
#   forward.pt — q, output, KV-cache fingerprint (eager mode only; in CG
#                mode forward runs only at warmup/capture with zero inputs).
#   kvwrite.pt — key/value tensors entering KVCacheWriteOp + the page slots
#                they target + the cache state at those slots BEFORE the
#                write (so we can detect wrong-page writes that pollute KV).
#
# Workflow:
#   1) Run smoke test with FA3_TV_DUMP_DIR=/tmp/fa3_off + CG OFF
#      (DISABLE_SP_TARGET_VERIFY_CUDA_GRAPH=1)
#   2) Run smoke test with FA3_TV_DUMP_DIR=/tmp/fa3_on + CG ON
#   3) Use compare_fa3_tv_dumps.py to pair calls by cache_seqlens fingerprint
#      and report first divergent tensor.
# ---------------------------------------------------------------------------
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
    # Force-sync so any async optimizedCopyAsync (from C++ prepareInputs)
    # is visible to the .cpu() copies below.
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    rank = int(os.environ.get("WORLD_RANK", os.environ.get("RANK", "0")))
    out_dir = _FA3_DUMP_DIR
    os.makedirs(out_dir, exist_ok=True)
    # Ring buffer: best-effort delete the file that drops out of the window
    # (any tag for that idx).  Keeps the dir bounded to max_calls files.
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

        # Debug dump — captures the inputs that the kernel will read.  In CG
        # mode prepare() runs at every replay-prep (via prepare_cuda_graph)
        # so each replay produces one dump.  In eager mode prepare() runs
        # before each forward.
        if _fa3_dump_enabled():
            payload: "dict[str, Any]" = {
                "mode": "cg" if self.enable_cuda_graph else "eager",
                "fixed_batch_size": int(self._fixed_batch_size or 0),
                "max_q_per_request": int(self._max_q_per_request),
                "cache_seqlens_buf": self._cache_seqlens_buf,
                "page_table_ref": self._page_table_ref,
                "cu_seqlens_q_ref": self._cu_seqlens_q_ref,
                "input_prefix_lengths_d": attn_inputs.prefix_lengths_d,
                "input_input_lengths_d": attn_inputs.input_lengths_d,
                "input_lengths_cpu": attn_inputs.input_lengths,
                "prefix_lengths_cpu": attn_inputs.prefix_lengths,
                "sequence_lengths_cpu": attn_inputs.sequence_lengths,
                "decode_cu_seqlens_d": attn_inputs.decode_cu_seqlens_d,
                "kv_cache_kernel_block_id_host": attn_inputs.kv_cache_kernel_block_id_host,
            }
            # fmha_params snapshot: data_ptrs reveal whether fill_params kept
            # buffers stable (forbid_realloc working) or replaced them
            # (captured KV write op would then alias stale data_ptrs).
            payload.update(_snapshot_fmha_params(self.fmha_params))
            _fa3_dump("prepare", payload)
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
        # Debug dump — q + output for the eager path.  In CG mode this only
        # fires at warmup/capture (zero inputs → zero output), but we still
        # save the data_ptr fingerprint so the comparator can verify the
        # captured kernel was launched with the same q buffer at replay.
        if _fa3_dump_enabled():
            # The assert at function start guarantees kv_cache is not None.
            paged = kv_cache.kv_cache_base
            _fa3_dump(
                "forward",
                {
                    "mode": "cg" if self.enable_cuda_graph else "eager",
                    "q": q,
                    "output": out,
                    "q_data_ptr": int(q.data_ptr()),
                    "k_cache_data_ptr": int(k_cache.data_ptr()),
                    "v_cache_data_ptr": int(v_cache.data_ptr()),
                    "kv_cache_base_data_ptr": int(paged.data_ptr()),
                    "kv_cache_base_shape": list(paged.shape),
                    "cache_seqlens_buf": self._cache_seqlens_buf,
                    "page_table_ref": self._page_table_ref,
                    "cu_seqlens_q_ref": self._cu_seqlens_q_ref,
                    "max_seqlen_q": int(self._max_q_per_request),
                },
            )
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

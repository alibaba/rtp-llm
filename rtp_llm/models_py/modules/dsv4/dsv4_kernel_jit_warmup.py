"""Startup JIT warmup for DSV4 prefill kernels.

The real startup gRPC warmup exercises full forward paths, but some JIT keys
are driven by batch-only constexpr branches or by DeepGEMM's M-derived dense
GEMM heuristics.  This module directly launches the small set of kernels whose
cold compiles are otherwise likely to happen after the health gate opens.
"""

from __future__ import annotations

from functools import lru_cache, partial
import logging
import os
import time
from typing import Any, Dict, Iterable, Optional, Tuple

import torch

_DENSE_GEMM_FALLBACK_M_GRID = [
    1,
    2,
    4,
    8,
    16,
    24,
    32,
    48,
    64,
    96,
    128,
    192,
    256,
    384,
    512,
    768,
    1024,
    1536,
    2048,
    3072,
    4096,
    6144,
    8192,
    12288,
    16384,
    24576,
    32768,
    49152,
    65536,
    98304,
    131072,
    196608,
    262144,
    393216,
    524288,
    786432,
    1048576,
]

_DENSE_GEMM_HEURISTIC_SCAN_LIMIT = 32768
_SM100_DENSE_BLOCK_M_CANDIDATES = tuple(range(16, 257, 16))
_SM100_DENSE_BLOCK_N_CANDIDATES = (16,) + tuple(range(32, 257, 32))

_BRANCH_KERNEL_JIT_WARMED_KEYS: set[tuple] = set()
_DENSE_GEMM_JIT_WARMED_KEYS: set[tuple] = set()
_BATCHED_FP8_EINSUM_JIT_WARMED_KEYS: set[tuple] = set()
_MHC_PRENORM_GEMM_JIT_WARMED_KEYS: set[tuple] = set()
_FP8_MQA_LOGITS_JIT_WARMED_KEYS: set[tuple] = set()
_DEEPGEMM_WARMUP_COMPILE_RETRIES = 2


def _cp_padded_tokens_per_rank_bound(max_seq_len: int, cp_size: int) -> int:
    cp_size = max(int(cp_size), 1)
    max_seq_len = max(int(max_seq_len), 0)
    if cp_size <= 1 or max_seq_len == 0:
        return max_seq_len
    global_alignment = cp_size * 2
    padded_seq_len = (
        (max_seq_len + global_alignment - 1) // global_alignment
    ) * global_alignment
    return padded_seq_len // cp_size


def resolve_dense_gemm_warmup_max_m(
    *,
    max_seq_len: int,
    max_batch_size: int,
    role_type_name: str,
    prefill_chunk_size: int = 0,
    max_tokens_per_rank: int = 0,
    max_potential_token_num: int = 0,
    is_speculative: bool = False,
    gen_num_per_cycle: int = 0,
    cp_size: int = 1,
    cp_enabled: bool = False,
) -> int:
    """Return the largest M bucket DenseGEMM startup warmup should cover."""

    role = str(role_type_name).upper().split(".")[-1]
    if role != "DECODE":
        # Prefill/PDFusion runs dense kernels on chunked token blocks. Warm the
        # production chunk M directly instead of the full logical context M.
        prefill_chunk_size = int(prefill_chunk_size or 0)
        if prefill_chunk_size > 0:
            return prefill_chunk_size

        # If chunking is disabled, fall back to the runtime-resolved rank cap
        # (CP-local when CP is enabled) before using max_seq_len.
        max_tokens_per_rank = int(max_tokens_per_rank or 0)
        if max_tokens_per_rank > 0:
            return max(max_tokens_per_rank, 1)
        max_m = max(int(max_seq_len), 1)
        if cp_enabled:
            cp_size = max(int(cp_size or 1), 1)
            max_m = max(_cp_padded_tokens_per_rank_bound(max_m, cp_size), 1)
        return max_m

    max_potential_token_num = int(max_potential_token_num or 0)
    if max_potential_token_num > 0:
        return max_potential_token_num

    tokens_per_batch = 1
    if is_speculative:
        tokens_per_batch = max(int(gen_num_per_cycle or 0) + 1, 1)
    return max(int(max_batch_size), 1) * tokens_per_batch


def _dense_gemm_m_grid(max_m: int) -> list[int]:
    max_m = int(max_m)
    if max_m <= 0:
        return []
    m_grid = [int(m) for m in _DENSE_GEMM_FALLBACK_M_GRID if 0 < int(m) <= max_m]
    if max_m not in m_grid:
        m_grid.append(max_m)
    return sorted(set(m_grid))


def _dist_rank() -> int:
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            return int(dist.get_rank())
    except Exception:
        pass
    for env_name in ("WORLD_RANK", "RANK", "LOCAL_RANK"):
        env_value = os.environ.get(env_name)
        if env_value is None:
            continue
        try:
            return int(env_value)
        except ValueError:
            pass
    return 0


def _dist_barrier() -> None:
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            dist.barrier()
    except Exception:
        logging.exception("[DSV4 KernelWarmup] distributed barrier failed")
        raise


def _deepgemm_cache_dir() -> str:
    for env_name in ("DG_JIT_CACHE_DIR", "DEEP_GEMM_CACHE_DIR"):
        value = os.environ.get(env_name)
        if value:
            return value
    return os.path.join(os.path.expanduser("~"), ".deep_gemm", "cache")


def _deepgemm_warmup_lock_path() -> str:
    cache_dir = _deepgemm_cache_dir()
    try:
        os.makedirs(cache_dir, exist_ok=True)
        return os.path.join(cache_dir, ".rtp_llm_dsv4_deepgemm_warmup.lock")
    except Exception:
        return os.path.join("/tmp", f"rtp_llm_dsv4_deepgemm_warmup_{os.getuid()}.lock")


def _run_deepgemm_warmup_launches_serialized(
    label: str, launch_fn: Any
) -> None:
    """Serialize DeepGEMM dummy launches across rank processes sharing a cache."""

    import fcntl

    rank = _dist_rank()
    lock_path = _deepgemm_warmup_lock_path()
    wait_start = time.time()
    logging.info("[%s] rank=%d waiting for serialized DeepGEMM JIT lock", label, rank)
    with open(lock_path, "a", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            wait_time = time.time() - wait_start
            logging.info(
                "[%s] rank=%d entered serialized DeepGEMM JIT lock after %.2fs",
                label,
                rank,
                wait_time,
            )
            launch_fn()
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
    logging.info("[%s] rank=%d released serialized DeepGEMM JIT lock", label, rank)


def _is_deepgemm_nvcc_compile_error(error: BaseException) -> bool:
    return "NVCC compilation failed" in str(error)


def _run_deepgemm_warmup_launch_with_retry(
    label: str,
    detail: str,
    launch_fn: Any,
    *,
    device: torch.device,
) -> None:
    """Retry transient DeepGEMM NVCC failures without hiding stable bad specs."""

    last_error: BaseException | None = None
    attempts = _DEEPGEMM_WARMUP_COMPILE_RETRIES + 1
    for attempt in range(1, attempts + 1):
        try:
            launch_fn()
            return
        except RuntimeError as error:
            if not _is_deepgemm_nvcc_compile_error(error):
                raise
            last_error = error
            if attempt >= attempts:
                break
            logging.warning(
                "[%s] %s hit NVCC compilation failure on attempt %d/%d; retrying",
                label,
                detail,
                attempt,
                attempts,
            )
            _sync_cuda(device)
            _release_cuda_cache(device)
            time.sleep(0.2 * attempt)

    assert last_error is not None
    raise last_error


def _is_cuda_device(device: torch.device) -> bool:
    return torch.device(device).type == "cuda" and torch.cuda.is_available()


def _assert_not_capturing() -> None:
    if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
        raise RuntimeError(
            "DSV4 kernel JIT warmup must not run inside CUDA graph capture"
        )


def _sync_cuda(device: torch.device) -> None:
    if _is_cuda_device(device):
        torch.cuda.synchronize(device)


def _release_cuda_cache(device: torch.device) -> None:
    if _is_cuda_device(device):
        torch.cuda.empty_cache()


def _positive_ints(values: Iterable[Any]) -> list[int]:
    result: list[int] = []
    for value in values:
        try:
            ivalue = int(value)
        except (TypeError, ValueError):
            continue
        if ivalue > 0:
            result.append(ivalue)
    return result


def _collect_dsv4_branch_kernel_configs(v4: Any, v4_args: Any) -> Dict[str, tuple]:
    """Collect branch-kernel constexpr configs from the live V4 model."""

    ratios = _positive_ints(getattr(v4_args, "compress_ratios", []))
    if not ratios and v4 is not None:
        ratios = _positive_ints(
            getattr(getattr(layer, "attn", None), "compress_ratio", 0)
            for layer in getattr(v4, "layers", [])
        )
    ratios = sorted(set(ratios))

    window_size = int(getattr(v4_args, "window_size", 0) or 0)
    if window_size <= 0 and v4 is not None and getattr(v4, "layers", None):
        window_size = int(getattr(v4.layers[0].attn, "window_size", 0) or 0)
    if window_size <= 0:
        window_size = 1

    rope_head_dim = int(getattr(v4_args, "rope_head_dim", 0) or 0)
    if rope_head_dim <= 0 and v4 is not None and getattr(v4, "layers", None):
        rope_head_dim = int(getattr(v4.layers[0].attn, "rope_head_dim", 0) or 0)
    if rope_head_dim <= 0:
        rope_head_dim = 64

    topk = int(getattr(v4_args, "index_topk", 0) or 0)

    combine_configs = {(window_size, 1, 0)}
    compressor_configs = set()
    for ratio in ratios:
        combine_configs.add((window_size, ratio, topk))
        compressor_configs.add((512, rope_head_dim, ratio, ratio == 4))
        if ratio == 4:
            compressor_configs.add((128, rope_head_dim, ratio, True))

    return {
        "combine": tuple(sorted(combine_configs)),
        "compressor": tuple(sorted(compressor_configs)),
    }


@torch.inference_mode()
def warmup_compressor_combine_branch_kernels(
    *,
    device: torch.device,
    v4: Any = None,
    v4_args: Any = None,
    compress_ratio: Optional[int] = None,
    rope_head_dim: Optional[int] = None,
    window_size: Optional[int] = None,
    topk: Optional[int] = None,
    overlap: Optional[bool] = None,
) -> None:
    """Compile batched/varlen Triton branches that B=1 gRPC warmup misses."""

    device = torch.device(device)
    if not _is_cuda_device(device):
        return
    _assert_not_capturing()

    if v4 is not None or v4_args is not None:
        configs = _collect_dsv4_branch_kernel_configs(v4, v4_args)
    else:
        if compress_ratio is None or rope_head_dim is None or window_size is None:
            raise ValueError("branch warmup requires v4/v4_args or explicit config")
        ratio = int(compress_ratio)
        configs = {
            "combine": ((int(window_size), ratio, int(topk or 0)),),
            "compressor": (
                (
                    512,
                    int(rope_head_dim),
                    ratio,
                    bool(overlap),
                ),
            ),
        }

    warmup_key = (configs["combine"], configs["compressor"], str(device))
    if warmup_key in _BRANCH_KERNEL_JIT_WARMED_KEYS:
        return

    _dist_barrier()
    rank = _dist_rank()
    if rank == 0:
        logging.info(
            "[DSV4 BranchKernels] JIT warmup start: combine=%s compressor=%s",
            configs["combine"],
            configs["compressor"],
        )
    t0 = time.time()
    for cfg in configs["combine"]:
        window, ratio, cfg_topk = cfg
        _warmup_combine_topk_swa_indices_cp(
            window_size=window,
            compress_ratio=ratio,
            topk=cfg_topk,
            device=device,
        )
    for cfg in configs["compressor"]:
        head_dim, rope_dim, ratio, cfg_overlap = cfg
        _warmup_fused_kv_compress_norm_rope_insert(
            head_dim=head_dim,
            rope_head_dim=rope_dim,
            compress_ratio=ratio,
            overlap=cfg_overlap,
            device=device,
        )
    _sync_cuda(device)
    _dist_barrier()
    if rank == 0:
        logging.info("[DSV4 BranchKernels] JIT warmup done in %.2fs", time.time() - t0)
    _BRANCH_KERNEL_JIT_WARMED_KEYS.add(warmup_key)


def _warmup_combine_topk_swa_indices_cp(
    *,
    window_size: int,
    compress_ratio: int,
    topk: int,
    device: torch.device,
) -> None:
    from rtp_llm.models_py.modules.dsv4.fp8._swa_ops_triton import (
        combine_topk_swa_indices_cp,
    )

    num_tokens = 32
    topk_width = max(int(topk), 0)
    topk_indices = torch.zeros(
        (num_tokens, topk_width), dtype=torch.int32, device=device
    )
    global_positions = torch.arange(num_tokens, dtype=torch.int64, device=device)
    m_value = max(int(window_size) + max(topk_width, 1) + 128, 256)
    n_value = 64 if topk_width else 0

    combine_topk_swa_indices_cp(
        topk_indices=topk_indices,
        global_positions=global_positions,
        sp_int=0,
        window_size=int(window_size),
        compress_ratio=int(compress_ratio),
        topk=topk_width,
        M=m_value,
        N=n_value,
        req_id_per_token=None,
        prefix_lengths=None,
    )

    req_id_per_token = torch.zeros((num_tokens,), dtype=torch.int32, device=device)
    prefix_lengths = torch.zeros((1,), dtype=torch.int32, device=device)
    combine_topk_swa_indices_cp(
        topk_indices=topk_indices,
        global_positions=global_positions,
        sp_int=0,
        window_size=int(window_size),
        compress_ratio=int(compress_ratio),
        topk=topk_width,
        M=m_value,
        N=n_value,
        req_id_per_token=req_id_per_token,
        prefix_lengths=prefix_lengths,
    )


def _warmup_fused_kv_compress_norm_rope_insert(
    *,
    head_dim: int,
    rope_head_dim: int,
    compress_ratio: int,
    overlap: bool,
    device: torch.device,
) -> None:
    from rtp_llm.models_py.modules.dsv4.fp8._compressor_consts import (
        INDEXER_ENTRY_BYTES,
        KV_ENTRY_BYTES,
    )
    from rtp_llm.models_py.modules.dsv4.fp8._compressor_vllm_triton import (
        run_fused_compress_kv_write,
    )

    head_dim = int(head_dim)
    rope_head_dim = int(rope_head_dim)
    compress_ratio = int(compress_ratio)
    coff = 1 + int(bool(overlap))
    num_tokens = 32
    positions = torch.arange(
        num_tokens, dtype=torch.int64, device=device
    ) * compress_ratio + (compress_ratio - 1)
    max_position = int(positions[-1].item()) if num_tokens else 0
    n_raw = max(max_position + 1, coff * compress_ratio)
    raw_width = coff * head_dim

    token_to_req = torch.zeros((num_tokens,), dtype=torch.int32, device=device)
    slot_mapping = torch.arange(num_tokens, dtype=torch.int64, device=device)
    kv_slot_mapping = torch.zeros((num_tokens,), dtype=torch.int64, device=device)
    state_block_size = 256
    state_blocks = max(max_position // state_block_size + 1, 1)
    block_table = torch.zeros((1, state_blocks), dtype=torch.int32, device=device)
    state_cache = torch.zeros(
        (state_blocks, state_block_size, 2 * raw_width),
        dtype=torch.float32,
        device=device,
    )
    kv_raw = torch.zeros((n_raw, raw_width), dtype=torch.float32, device=device)
    score_raw = torch.zeros_like(kv_raw)
    ape = torch.zeros((compress_ratio, raw_width), dtype=torch.float32, device=device)
    rms_norm_weight = torch.ones((head_dim,), dtype=torch.bfloat16, device=device)
    cos_sin_cache = torch.zeros(
        (max_position + 1, rope_head_dim), dtype=torch.float32, device=device
    )

    kv_block_size = 64
    entry_bytes = KV_ENTRY_BYTES if head_dim == 512 else INDEXER_ENTRY_BYTES
    kv_cache = torch.zeros(
        (1, kv_block_size, entry_bytes), dtype=torch.uint8, device=device
    )

    for batched in (False, True):
        kwargs = {}
        if batched:
            kwargs["seq_start_per_req"] = torch.zeros(
                (1,), dtype=torch.int32, device=device
            )
            kwargs["cu_seq_per_req"] = torch.tensor(
                [0, n_raw], dtype=torch.int32, device=device
            )
        run_fused_compress_kv_write(
            state_cache=state_cache,
            token_to_req_indices=token_to_req,
            positions=positions,
            slot_mapping=slot_mapping,
            block_table=block_table,
            rms_norm_weight=rms_norm_weight,
            rms_norm_eps=1.0e-6,
            cos_sin_cache=cos_sin_cache,
            kv_cache=kv_cache,
            kv_slot_mapping=kv_slot_mapping,
            kv_raw=kv_raw,
            score_raw=score_raw,
            ape=ape,
            seq_start=0,
            disable_raw_path=False,
            head_dim=head_dim,
            rope_head_dim=rope_head_dim,
            compress_ratio=compress_ratio,
            overlap=bool(overlap),
            **kwargs,
        )


def _shape_key(kind: str, n_value: int, k_value: int) -> tuple[str, int, int]:
    return (str(kind), int(n_value), int(k_value))


def _maybe_add_shape(
    shapes: Dict[tuple[str, int, int], dict],
    key: tuple[str, int, int],
    info: dict,
) -> None:
    if key not in shapes:
        shapes[key] = info


def _collect_dsv4_dense_gemm_shapes(model: Any) -> Dict[tuple[str, int, int], dict]:
    """Collect representative dense DeepGEMM shapes from the live DSV4 model."""

    shapes: Dict[tuple[str, int, int], dict] = {}
    for module_name, module in model.named_modules():
        cls_name = module.__class__.__name__
        if (
            cls_name == "CudaFp8DeepGEMMLinear"
            and hasattr(module, "N")
            and hasattr(module, "K")
        ):
            weight = getattr(module, "weight", None)
            scale = getattr(module, "weight_scales", None)
            if weight is None or scale is None:
                continue
            key = _shape_key("fp8", int(module.N), int(module.K))
            _maybe_add_shape(
                shapes,
                key,
                {
                    "name": module_name,
                    "module": module,
                    "weight": weight,
                    "scale": scale,
                    "scale_ue8m0": bool(
                        getattr(module, "scale_ue8m0", scale.dtype == torch.int32)
                    ),
                },
            )
            continue

        if getattr(module, "storage", None) == "fp4":
            weight = getattr(module, "weight", None)
            scale = getattr(module, "scale_gemm", None)
            if weight is None or scale is None:
                continue
            n_value = int(getattr(module, "out_features", weight.shape[0]))
            k_value = int(getattr(module, "in_features", weight.shape[1] * 2))
            key = _shape_key("fp8_fp4", n_value, k_value)
            _maybe_add_shape(
                shapes,
                key,
                {"name": module_name, "weight": weight, "scale": scale},
            )

    for module_name, module in model.named_modules():
        cls_name = module.__class__.__name__
        if cls_name == "GroupedFP4Strategy":
            _collect_grouped_fp4_strategy_shapes(shapes, module_name, module)
        elif cls_name == "LocalLoopStrategy":
            _collect_local_loop_strategy_shapes(shapes, module_name, module)
        # NOTE: MegaMoEStrategy intentionally NOT walked here — its own
        # ``_maybe_warmup_jit_once`` covers ``fp8_fp4_mega_moe`` (a distinct
        # symm-mem kernel from ``fp8_fp4_gemm_nt`` used by GroupedFP4 /
        # LocalLoop). MegaMoE has no ``fp8_fp4_gemm_nt`` fallback, so a
        # dummy launch here would either fail or compile a kernel that is
        # never used.

    # Sanity log: makes prewarm coverage visible in the startup log so any
    # walk regression (missed module class, lazy strategy, layer rename)
    # shows up immediately. Compare against runtime ``Running NVCC`` lines —
    # if production still cold-JITs ``sm100_fp8(_fp4)?_gemm_1d1d`` after the
    # health gate opens, the (kind, N, K) collected here is incomplete.
    fp8_keys = sorted(k for k in shapes if k[0] == "fp8")
    fp8_fp4_keys = sorted(k for k in shapes if k[0] == "fp8_fp4")
    logging.info(
        "[DSV4 DenseGEMM] collected shapes: fp8=%d fp8_fp4=%d total=%d",
        len(fp8_keys),
        len(fp8_fp4_keys),
        len(shapes),
    )
    if fp8_keys:
        logging.info("[DSV4 DenseGEMM]   fp8 (N,K): %s", fp8_keys)
    if fp8_fp4_keys:
        logging.info("[DSV4 DenseGEMM]   fp8_fp4 (N,K): %s", fp8_fp4_keys)
    return shapes


def _collect_dsv4_mhc_prenorm_shapes(model: Any) -> Dict[tuple[int, int], dict]:
    """Collect mHC DeepGEMM prenorm GEMM shapes from live TileLang HC units."""

    shapes: Dict[tuple[int, int], dict] = {}
    for module_name, module in model.named_modules():
        if module.__class__.__name__ != "TileLangHCUnit":
            continue
        fn = getattr(module, "fn", None)
        if not isinstance(fn, torch.Tensor) or fn.dim() != 2:
            continue
        if fn.dtype != torch.float32:
            continue
        n_value = int(fn.shape[0])
        k_value = int(fn.shape[1])
        if n_value <= 0 or k_value <= 0:
            continue
        key = (n_value, k_value)
        if key not in shapes:
            shapes[key] = {"name": module_name, "fn": fn}

    logging.info(
        "[DSV4 mHC DeepGEMM] collected prenorm shapes: count=%d shapes=%s",
        len(shapes),
        sorted(shapes.keys()),
    )
    return shapes


def _collect_dsv4_batched_fp8_einsum_shapes(
    model: Any,
) -> Dict[tuple[int, int, int], dict]:
    """Collect wo_a DeepGEMM batched FP8 einsum shapes from live attention blocks."""

    shapes: Dict[tuple[int, int, int], dict] = {}
    for module_name, module in model.named_modules():
        weight = getattr(module, "_wo_a_stk_w", None)
        scale = getattr(module, "_wo_a_stk_s", None)
        if not isinstance(weight, torch.Tensor) or not isinstance(scale, torch.Tensor):
            continue
        if weight.dim() != 3 or scale.dim() != 3:
            continue
        groups = int(weight.shape[0])
        n_value = int(weight.shape[1])
        k_value = int(weight.shape[2])
        if groups <= 0 or n_value <= 0 or k_value <= 0:
            continue
        if int(scale.shape[0]) != groups or int(scale.shape[1]) < n_value:
            continue
        key = (groups, n_value, k_value)
        if key not in shapes:
            shapes[key] = {"name": module_name, "weight": weight, "scale": scale}

    logging.info(
        "[DSV4 BatchedFP8Einsum] collected wo_a shapes: count=%d shapes=%s",
        len(shapes),
        sorted(shapes.keys()),
    )
    return shapes


def _collect_dsv4_fp8_mqa_logits_shapes(model: Any) -> Dict[tuple[int, int], dict]:
    """Collect non-paged FP8 MQA logits shapes from live FP8 indexer modules."""

    shapes: Dict[tuple[int, int], dict] = {}
    for module_name, module in model.named_modules():
        if module.__class__.__name__ != "IndexerFP8":
            continue
        num_heads = int(getattr(module, "n_heads", 0) or 0)
        head_dim = int(getattr(module, "head_dim", 0) or 0)
        if num_heads <= 0 or head_dim <= 0:
            continue
        key = (num_heads, head_dim)
        if key not in shapes:
            shapes[key] = {"name": module_name}

    logging.info(
        "[DSV4 FP8MQALogits] collected prefill indexer shapes: count=%d shapes=%s",
        len(shapes),
        sorted(shapes.keys()),
    )
    return shapes


def _collect_grouped_fp4_strategy_shapes(
    shapes: Dict[tuple[str, int, int], dict],
    module_name: str,
    strategy: Any,
) -> None:
    for name, weight_attr, scale_attr in (
        ("grouped_w13", "_w13", "_s13_dense_t"),
        ("grouped_w2", "_w2", "_s2_dense_t"),
    ):
        if not hasattr(strategy, weight_attr) or not hasattr(strategy, scale_attr):
            continue
        weight_stack = getattr(strategy, weight_attr)
        scale_stack_t = getattr(strategy, scale_attr)
        if weight_stack.dim() != 3:
            continue
        expert_idx = torch.zeros((1,), dtype=torch.long, device=weight_stack.device)
        weight = weight_stack[0]
        scale = (
            torch.index_select(scale_stack_t, 0, expert_idx).squeeze(0).transpose(0, 1)
        )
        n_value = int(weight.shape[0])
        k_value = int(weight.shape[1]) * 2
        key = _shape_key("fp8_fp4", n_value, k_value)
        _maybe_add_shape(
            shapes,
            key,
            {"name": f"{module_name}.{name}", "weight": weight, "scale": scale},
        )


def _collect_local_loop_strategy_shapes(
    shapes: Dict[tuple[str, int, int], dict],
    module_name: str,
    strategy: Any,
) -> None:
    for name, weight_attr, scale_attr in (
        ("local_w1", "_W1_w", "_W1_s_gemm_t"),
        ("local_w2", "_W2_w", "_W2_s_gemm_t"),
        ("local_w3", "_W3_w", "_W3_s_gemm_t"),
    ):
        if not hasattr(strategy, weight_attr) or not hasattr(strategy, scale_attr):
            continue
        weight_stack = getattr(strategy, weight_attr)
        scale_stack_t = getattr(strategy, scale_attr)
        if weight_stack.dim() != 3:
            continue
        expert_idx = torch.zeros((1,), dtype=torch.long, device=weight_stack.device)
        weight = weight_stack[0]
        scale = (
            torch.index_select(scale_stack_t, 0, expert_idx).squeeze(0).transpose(0, 1)
        )
        n_value = int(weight.shape[0])
        k_value = int(weight.shape[1]) * 2
        key = _shape_key("fp8_fp4", n_value, k_value)
        _maybe_add_shape(
            shapes,
            key,
            {"name": f"{module_name}.{name}", "weight": weight, "scale": scale},
        )


def _ceil_div(a: int, b: int) -> int:
    return (int(a) + int(b) - 1) // int(b)


def _align(value: int, alignment: int) -> int:
    return _ceil_div(value, alignment) * alignment


def _get_deep_gemm_num_sms(device: torch.device) -> int:
    try:
        import deep_gemm

        num_sms = int(deep_gemm.get_num_sms())
        if num_sms > 0:
            return num_sms
    except Exception:
        pass

    props = torch.cuda.get_device_properties(device)
    return int(props.multi_processor_count)


def _sm100_swizzle_mode(block_size: int, elem_size: int) -> int:
    for mode in (128, 64, 32, 16):
        if (int(block_size) * int(elem_size)) % mode == 0:
            return mode
    raise ValueError(f"unsupported swizzle block={block_size} elem={elem_size}")


def _sm100_dense_storage_signature(
    *,
    swap_ab: int,
    block_m: int,
    block_n: int,
    block_k: int,
    cluster_m: int,
    cluster_n: int,
    cd_elem_size: int,
) -> tuple[int, ...]:
    load_block_m = block_m // cluster_n
    load_block_n = block_n // cluster_m
    store_block_m = 16 if swap_ab else min(128, block_m)
    store_block_n = block_n
    swizzle_a_mode = _sm100_swizzle_mode(block_k, 1)
    swizzle_b_mode = _sm100_swizzle_mode(block_k, 1)
    swizzle_cd_mode = _sm100_swizzle_mode(store_block_n, cd_elem_size)
    smem_cd = (
        store_block_m * store_block_n * cd_elem_size * 2
        if swap_ab
        else store_block_m * swizzle_cd_mode * 2
    )
    smem_barriers = 32 * 8 * 3 + 2 * 8 * 2 + 8
    smem_tmem_ptr = 4
    smem_extra = smem_cd + smem_barriers + smem_tmem_ptr
    sf_block_m = _align(block_m, 128)
    sf_block_n = _align(block_n, 128)
    smem_per_stage = (
        load_block_m * block_k
        + load_block_n * block_k
        + sf_block_m * 4
        + sf_block_n * 4
    )
    num_stages = min((232448 - smem_extra) // smem_per_stage, 32)
    return (
        load_block_m,
        load_block_n,
        store_block_m,
        store_block_n,
        swizzle_a_mode,
        swizzle_b_mode,
        swizzle_cd_mode,
        num_stages,
    )


def _sm100_dense_layout_info(
    layout: tuple[int, int, int, int, int, int],
    *,
    m_value: int,
    n_value: int,
    num_groups: int,
    num_sms: int,
) -> tuple[int, int]:
    _, block_m, block_n, _, _, _ = layout
    num_blocks = (
        _ceil_div(m_value, block_m)
        * _ceil_div(n_value, block_n)
        * max(int(num_groups), 1)
    )
    num_waves = _ceil_div(num_blocks, num_sms)
    num_last_blocks = num_blocks % num_sms
    last_wave_util = num_sms if num_last_blocks == 0 else num_last_blocks
    return num_waves, last_wave_util


def _sm100_dense_layout_is_better(
    candidate: tuple[int, int, int, int, int, int],
    current: tuple[int, int, int, int, int, int],
    *,
    m_value: int,
    n_value: int,
    num_groups: int,
    num_sms: int,
) -> bool:
    cand_waves, cand_last = _sm100_dense_layout_info(
        candidate,
        m_value=m_value,
        n_value=n_value,
        num_groups=num_groups,
        num_sms=num_sms,
    )
    curr_waves, curr_last = _sm100_dense_layout_info(
        current,
        m_value=m_value,
        n_value=n_value,
        num_groups=num_groups,
        num_sms=num_sms,
    )
    if (cand_waves == 1 or curr_waves == 1) and cand_waves != curr_waves:
        return cand_waves < curr_waves

    cand_cluster = candidate[4] * candidate[5]
    curr_cluster = current[4] * current[5]
    if cand_cluster != curr_cluster:
        return cand_cluster > curr_cluster

    if cand_waves != curr_waves:
        return cand_waves < curr_waves

    if cand_last != curr_last:
        return cand_last > curr_last

    cand_shape = candidate[1] + candidate[2]
    curr_shape = current[1] + current[2]
    if cand_shape != curr_shape:
        return cand_shape < curr_shape

    return candidate[1] * candidate[2] < current[1] * current[2]


@lru_cache(maxsize=4096)
def _sm100_dense_layout_signature(
    *,
    m_value: int,
    n_value: int,
    k_value: int,
    kind: str,
    num_sms: int,
    num_groups: int = 1,
) -> tuple[int, ...]:
    """Mirror the M-dependent part of DeepGEMM SM100 dense layout selection.

    DeepGEMM does not compile literal M for RTP's ``compiled_dims="nk"``
    calls, but it does compile the heuristic-selected layout.  This local
    mirror is used only to pick one representative M for each likely layout;
    the actual source of truth remains the real dummy DeepGEMM launch.
    """

    gemm_type_key = 1 if str(kind) == "fp8_batched" else 0
    num_groups = max(int(num_groups), 1)
    block_k = 128
    candidates: list[tuple[int, int, int, int, int, int]] = []

    for swap_ab in (0, 1):
        if swap_ab:
            block_m_candidates = _SM100_DENSE_BLOCK_M_CANDIDATES
            block_n_candidates = (128,)
        else:
            if m_value <= 32:
                block_m_candidates = (32,)
            elif m_value <= 64:
                block_m_candidates = (64,)
            else:
                block_m_candidates = (128,)
            block_n_end = 128 if k_value <= 256 else 256
            block_n_candidates = tuple(
                n for n in _SM100_DENSE_BLOCK_N_CANDIDATES if n <= block_n_end
            )

        for cluster_m in (1, 2):
            if swap_ab and cluster_m > 1:
                continue
            for cluster_n in (1, 2):
                if cluster_m * cluster_n > 2:
                    continue
                if not swap_ab and cluster_n > 1:
                    continue
                if num_sms % (cluster_m * cluster_n) != 0:
                    continue

                for block_m in block_m_candidates:
                    if (block_m // cluster_n) % 8 != 0:
                        continue
                    if _ceil_div(m_value, block_m) % cluster_m != 0:
                        continue

                    for block_n in block_n_candidates:
                        if (block_n // cluster_m) % 8 != 0:
                            continue
                        if _ceil_div(n_value, block_n) % cluster_n != 0:
                            continue
                        if swap_ab and block_n != 128:
                            continue

                        sf_block_m = _align(block_m, 128)
                        sf_block_n = _align(block_n, 128)
                        tmem_sf_cols = sf_block_m // 32 + sf_block_n // 32
                        umma_n = block_m if swap_ab else block_n
                        if 2 * umma_n + tmem_sf_cols > 512:
                            continue

                        # RTP warmup tensors are K-major for A and B.  DeepGEMM
                        # keeps only layouts whose A/B swizzles are both 128B.
                        if (
                            _sm100_swizzle_mode(block_k, 1) != 128
                            or _sm100_swizzle_mode(block_k, 1) != 128
                        ):
                            continue
                        candidates.append(
                            (swap_ab, block_m, block_n, block_k, cluster_m, cluster_n)
                        )

    if not candidates:
        return (0, 128, 128, block_k, 1, 1)

    best = candidates[0]
    for candidate in candidates[1:]:
        if _sm100_dense_layout_is_better(
            candidate,
            best,
            m_value=m_value,
            n_value=n_value,
            num_groups=num_groups,
            num_sms=num_sms,
        ):
            best = candidate

    storage = _sm100_dense_storage_signature(
        swap_ab=best[0],
        block_m=best[1],
        block_n=best[2],
        block_k=best[3],
        cluster_m=best[4],
        cluster_n=best[5],
        cd_elem_size=2,
    )
    return best + storage + (num_groups, gemm_type_key)


def _candidate_dense_gemm_m_values(
    *,
    max_m: int,
    n_value: int,
    num_sms: int,
    num_groups: int = 1,
) -> list[int]:
    max_m = max(int(max_m), 0)
    if max_m <= 0:
        return []
    num_groups = max(int(num_groups), 1)

    values: set[int] = set()
    values.update(range(1, min(max_m, 64) + 1))

    scan_limit = min(max_m, _DENSE_GEMM_HEURISTIC_SCAN_LIMIT)
    values.update(range(80, scan_limit + 1, 16))

    for m_value in _DENSE_GEMM_FALLBACK_M_GRID:
        if 0 < m_value <= max_m:
            values.add(int(m_value))

    for chunk in (65536, 131072, 262144, 524288, 1048576):
        for delta in (-1, 0, 1):
            m_value = chunk + delta
            if 0 < m_value <= max_m:
                values.add(m_value)

    block_ns = tuple(range(16, min(256, max(n_value, 16)) + 1, 16))
    max_wave = max(
        10,
        _ceil_div(
            max_m * max(1, _ceil_div(n_value, 128)) * num_groups,
            16 * max(num_sms, 1),
        ),
    )
    waves = set(range(1, 11))
    wave = 16
    while wave <= max_wave:
        waves.add(wave)
        waves.add(max(1, wave - 1))
        wave *= 2

    for block_m in _SM100_DENSE_BLOCK_M_CANDIDATES:
        for block_n in block_ns:
            n_blocks = max(1, _ceil_div(n_value, block_n))
            for wave_value in waves:
                m_center = wave_value * num_sms * block_m // (n_blocks * num_groups)
                for raw_m in (m_center - 1, m_center, m_center + 1):
                    if raw_m <= 0:
                        continue
                    for m_value in (
                        raw_m,
                        _align(raw_m, 16),
                        max(1, _align(raw_m, block_m) - block_m + 1),
                        _align(raw_m, block_m),
                    ):
                        if 0 < m_value <= max_m:
                            values.add(m_value)

    return sorted(values)


@lru_cache(maxsize=1024)
def _generate_dense_gemm_warmup_m_grid(
    *,
    max_m: int,
    n_value: int,
    k_value: int,
    kind: str,
    num_sms: int,
    num_groups: int = 1,
) -> tuple[int, ...]:
    reps_by_signature: dict[tuple[int, ...], int] = {}
    for m_value in _candidate_dense_gemm_m_values(
        max_m=max_m,
        n_value=n_value,
        num_sms=num_sms,
        num_groups=num_groups,
    ):
        signature = _sm100_dense_layout_signature(
            m_value=m_value,
            n_value=n_value,
            k_value=k_value,
            kind=kind,
            num_sms=num_sms,
            num_groups=num_groups,
        )
        reps_by_signature.setdefault(signature, m_value)
    return tuple(sorted(reps_by_signature.values()))


def _mhc_prenorm_deepgemm_backend_enabled() -> bool:
    requested = os.environ.get("DSV4_MHC_PRE_GEMM_BACKEND", "").strip().lower()
    if requested in ("", "auto", "deepgemm", "dg"):
        return True
    if requested in ("tilelang", "single", "tilelang_single", "tilelang_splitk"):
        return False
    return requested == "deepgemm"


def _compute_mhc_prenorm_num_split(
    *,
    m_value: int,
    k_value: int,
    num_sms: int,
) -> int:
    block_m = 64
    block_k = 64
    grid_size = _ceil_div(max(int(m_value), 1), block_m)
    split_k = max(int(num_sms), 1) // max(grid_size, 1)
    num_block_k = _ceil_div(int(k_value), block_k)
    split_k = min(split_k, num_block_k // 4)
    return max(split_k, 1)


@lru_cache(maxsize=1024)
def _generate_mhc_prenorm_warmup_specs(
    *,
    max_m: int,
    k_value: int,
    num_sms: int,
) -> tuple[tuple[int, int], ...]:
    """Return ``(num_splits, representative_m)`` pairs for mHC prenorm GEMM.

    DeepGEMM compiles ``num_splits`` for this kernel, not M itself.  RTP's
    TileKernels wrapper derives ``num_splits`` from ``ceil(num_tokens / 64)``;
    choose the first M that reaches each split value up to the rank-local
    prefill token bound.
    """

    max_m = max(int(max_m), 0)
    if max_m <= 0:
        return ()

    reps_by_split: dict[int, int] = {}
    max_grid = _ceil_div(max_m, 64)
    for grid_size in range(1, max_grid + 1):
        m_value = (grid_size - 1) * 64 + 1
        num_splits = _compute_mhc_prenorm_num_split(
            m_value=m_value,
            k_value=k_value,
            num_sms=num_sms,
        )
        reps_by_split.setdefault(num_splits, m_value)
        if num_splits == 1:
            break

    return tuple((split, reps_by_split[split]) for split in sorted(reps_by_split))


def _fp8_mqa_logits_available() -> bool:
    try:
        import deep_gemm

        return hasattr(deep_gemm, "fp8_mqa_logits")
    except Exception:
        return False


@torch.inference_mode()
def warmup_dense_gemm_jit(
    shapes: Dict[tuple[str, int, int], dict],
    *,
    max_m: int,
    device: torch.device,
) -> None:
    """Compile representative DeepGEMM dense GEMM M buckets at startup."""

    device = torch.device(device)
    if not _is_cuda_device(device) or not shapes:
        return
    _assert_not_capturing()

    num_sms = _get_deep_gemm_num_sms(device)
    shape_keys = tuple(sorted(shapes.keys()))
    m_grids = {
        key: _generate_dense_gemm_warmup_m_grid(
            max_m=int(max_m),
            n_value=int(key[1]),
            k_value=int(key[2]),
            kind=str(key[0]),
            num_sms=num_sms,
        )
        for key in shape_keys
    }
    m_grids = {key: grid for key, grid in m_grids.items() if grid}
    if not m_grids:
        return

    warmup_key = (
        int(max_m),
        tuple((key, m_grids.get(key, ())) for key in shape_keys),
        int(num_sms),
        str(device),
    )
    if warmup_key in _DENSE_GEMM_JIT_WARMED_KEYS:
        return

    rank = _dist_rank()
    if rank == 0:
        total_launches = sum(len(grid) for grid in m_grids.values())
        logging.info(
            "[DSV4 DenseGEMM] JIT warmup start: %d shapes, %d representative M launches, num_sms=%d: %s",
            len(shape_keys),
            total_launches,
            num_sms,
            shape_keys,
        )
        logging.info(
            "[DSV4 DenseGEMM] representative M grids: %s",
            {key: m_grids.get(key, ()) for key in shape_keys},
        )
    def _run_warmup_launches() -> None:
        for key in shape_keys:
            info = shapes[key]
            for m_value in m_grids.get(key, ()):
                _run_deepgemm_warmup_launch_with_retry(
                    "DSV4 DenseGEMM",
                    f"shape={key} m={m_value}",
                    partial(
                        _launch_dummy_gemm,
                        key=key,
                        info=info,
                        m_value=m_value,
                        device=device,
                    ),
                    device=device,
                )
            _sync_cuda(device)
            _release_cuda_cache(device)

    t0 = time.time()
    _run_deepgemm_warmup_launches_serialized(
        "DSV4 DenseGEMM", _run_warmup_launches
    )
    if rank == 0:
        logging.info("[DSV4 DenseGEMM] JIT warmup done in %.2fs", time.time() - t0)
    _DENSE_GEMM_JIT_WARMED_KEYS.add(warmup_key)


@torch.inference_mode()
def warmup_batched_fp8_einsum_jit(
    shapes: Dict[tuple[int, int, int], dict],
    *,
    max_m: int,
    device: torch.device,
) -> None:
    """Compile reachable DeepGEMM wo_a batched FP8 einsum layouts."""

    device = torch.device(device)
    if not _is_cuda_device(device) or not shapes:
        return
    _assert_not_capturing()

    num_sms = _get_deep_gemm_num_sms(device)
    shape_keys = tuple(sorted(shapes.keys()))
    m_grids = {
        key: _generate_dense_gemm_warmup_m_grid(
            max_m=int(max_m),
            n_value=int(key[1]),
            k_value=int(key[2]),
            kind="fp8_batched",
            num_sms=num_sms,
            num_groups=int(key[0]),
        )
        for key in shape_keys
    }
    m_grids = {key: grid for key, grid in m_grids.items() if grid}
    if not m_grids:
        return

    warmup_key = (
        int(max_m),
        tuple((key, m_grids.get(key, ())) for key in shape_keys),
        int(num_sms),
        str(device),
    )
    if warmup_key in _BATCHED_FP8_EINSUM_JIT_WARMED_KEYS:
        return

    rank = _dist_rank()
    if rank == 0:
        total_launches = sum(len(grid) for grid in m_grids.values())
        logging.info(
            "[DSV4 BatchedFP8Einsum] JIT warmup start: %d shapes, %d representative M launches, num_sms=%d: %s",
            len(shape_keys),
            total_launches,
            num_sms,
            shape_keys,
        )
        logging.info(
            "[DSV4 BatchedFP8Einsum] representative M grids: %s",
            {key: m_grids.get(key, ()) for key in shape_keys},
        )
    def _run_warmup_launches() -> None:
        for key in shape_keys:
            info = shapes[key]
            for m_value in m_grids.get(key, ()):
                _run_deepgemm_warmup_launch_with_retry(
                    "DSV4 BatchedFP8Einsum",
                    f"shape={key} m={m_value}",
                    partial(
                        _launch_dummy_batched_fp8_einsum,
                        key=key,
                        info=info,
                        m_value=m_value,
                        device=device,
                    ),
                    device=device,
                )
            _sync_cuda(device)
            _release_cuda_cache(device)

    t0 = time.time()
    _run_deepgemm_warmup_launches_serialized(
        "DSV4 BatchedFP8Einsum", _run_warmup_launches
    )
    if rank == 0:
        logging.info(
            "[DSV4 BatchedFP8Einsum] JIT warmup done in %.2fs", time.time() - t0
        )
    _BATCHED_FP8_EINSUM_JIT_WARMED_KEYS.add(warmup_key)


@torch.inference_mode()
def warmup_mhc_prenorm_gemm_jit(
    shapes: Dict[tuple[int, int], dict],
    *,
    max_m: int,
    device: torch.device,
) -> None:
    """Compile reachable DeepGEMM mHC prenorm GEMM split-K variants."""

    device = torch.device(device)
    if not _is_cuda_device(device) or not shapes:
        return
    if not _mhc_prenorm_deepgemm_backend_enabled():
        return
    _assert_not_capturing()

    num_sms = _get_deep_gemm_num_sms(device)
    shape_keys = tuple(sorted(shapes.keys()))
    specs_by_shape = {
        key: _generate_mhc_prenorm_warmup_specs(
            max_m=int(max_m),
            k_value=int(key[1]),
            num_sms=num_sms,
        )
        for key in shape_keys
    }
    specs_by_shape = {key: specs for key, specs in specs_by_shape.items() if specs}
    if not specs_by_shape:
        return

    warmup_key = (
        int(max_m),
        tuple((key, specs_by_shape.get(key, ())) for key in shape_keys),
        int(num_sms),
        str(device),
    )
    if warmup_key in _MHC_PRENORM_GEMM_JIT_WARMED_KEYS:
        return

    rank = _dist_rank()
    if rank == 0:
        total_launches = sum(len(specs) for specs in specs_by_shape.values())
        logging.info(
            "[DSV4 mHC DeepGEMM] JIT warmup start: %d shapes, %d num_splits launches, max_m=%d, num_sms=%d: %s",
            len(shape_keys),
            total_launches,
            int(max_m),
            num_sms,
            shape_keys,
        )
        logging.info(
            "[DSV4 mHC DeepGEMM] representative split specs: %s",
            {key: specs_by_shape.get(key, ()) for key in shape_keys},
        )
    def _run_warmup_launches() -> None:
        for key in shape_keys:
            info = shapes[key]
            for num_splits, m_value in specs_by_shape.get(key, ()):
                _run_deepgemm_warmup_launch_with_retry(
                    "DSV4 mHC DeepGEMM",
                    f"shape={key} num_splits={num_splits} m={m_value}",
                    partial(
                        _launch_dummy_mhc_prenorm_gemm,
                        key=key,
                        info=info,
                        m_value=m_value,
                        num_splits=num_splits,
                        device=device,
                    ),
                    device=device,
                )
            _sync_cuda(device)
            _release_cuda_cache(device)

    t0 = time.time()
    _run_deepgemm_warmup_launches_serialized(
        "DSV4 mHC DeepGEMM", _run_warmup_launches
    )
    if rank == 0:
        logging.info(
            "[DSV4 mHC DeepGEMM] JIT warmup done in %.2fs", time.time() - t0
        )
    _MHC_PRENORM_GEMM_JIT_WARMED_KEYS.add(warmup_key)


@torch.inference_mode()
def warmup_fp8_mqa_logits_jit(
    shapes: Dict[tuple[int, int], dict],
    *,
    device: torch.device,
) -> None:
    """Compile DeepGEMM non-paged FP8 MQA logits used by prefill indexer."""

    device = torch.device(device)
    if not _is_cuda_device(device) or not shapes:
        return
    if not _fp8_mqa_logits_available():
        return
    _assert_not_capturing()

    shape_keys = tuple(sorted(shapes.keys()))
    warmup_key = (shape_keys, str(device))
    if warmup_key in _FP8_MQA_LOGITS_JIT_WARMED_KEYS:
        return

    rank = _dist_rank()
    if rank == 0:
        logging.info(
            "[DSV4 FP8MQALogits] JIT warmup start: %d shapes: %s",
            len(shape_keys),
            shape_keys,
        )

    def _run_warmup_launches() -> None:
        for key in shape_keys:
            _run_deepgemm_warmup_launch_with_retry(
                "DSV4 FP8MQALogits",
                f"shape={key}",
                partial(
                    _launch_dummy_fp8_mqa_logits,
                    key=key,
                    device=device,
                ),
                device=device,
            )
            _sync_cuda(device)
            _release_cuda_cache(device)

    t0 = time.time()
    _run_deepgemm_warmup_launches_serialized(
        "DSV4 FP8MQALogits", _run_warmup_launches
    )
    if rank == 0:
        logging.info(
            "[DSV4 FP8MQALogits] JIT warmup done in %.2fs", time.time() - t0
        )
    _FP8_MQA_LOGITS_JIT_WARMED_KEYS.add(warmup_key)


def _create_dummy_fp8_input(
    *,
    m_value: int,
    k_value: int,
    device: torch.device,
    scale_ue8m0: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    from rtp_llm.models_py.kernels.cuda.fp8_kernel import (
        create_per_token_group_quant_fp8_output_scale,
    )

    a = torch.zeros((m_value, k_value), dtype=torch.float8_e4m3fn, device=device)
    scale = create_per_token_group_quant_fp8_output_scale(
        x_shape=a.shape,
        device=device,
        group_size=128,
        column_major_scales=True,
        scale_tma_aligned=True,
        scale_ue8m0=scale_ue8m0,
    )
    if scale.dtype == torch.int32:
        scale.fill_(0x7F7F7F7F)
    else:
        scale.fill_(1.0)
    return a, scale


def _create_dummy_batched_fp8_einsum_input(
    *,
    m_value: int,
    groups: int,
    k_value: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    a = torch.zeros(
        (m_value, groups, k_value), dtype=torch.float8_e4m3fn, device=device
    )
    packed_sf_k = _ceil_div(k_value, 512)
    tma_m = _align(m_value, 4)
    scale_group_major = torch.empty(
        groups * packed_sf_k * tma_m, dtype=torch.int32, device=device
    ).as_strided(
        (groups, m_value, packed_sf_k),
        (packed_sf_k * tma_m, 1, tma_m),
    )
    scale_group_major.fill_(0x7F7F7F7F)
    return a, scale_group_major.transpose(0, 1)


def _launch_dummy_gemm(
    *,
    key: tuple[str, int, int],
    info: dict,
    m_value: int,
    device: torch.device,
) -> None:
    kind, n_value, k_value = key
    if kind == "fp8":
        from rtp_llm.models_py.kernels.cuda.deepgemm_wrapper import fp8_gemm_nt

        scale_ue8m0 = bool(info.get("scale_ue8m0", True))
        a, a_scale = _create_dummy_fp8_input(
            m_value=m_value,
            k_value=k_value,
            device=device,
            scale_ue8m0=scale_ue8m0,
        )
        out = torch.empty((m_value, n_value), dtype=torch.bfloat16, device=device)
        fp8_gemm_nt(
            (a, a_scale),
            (info["weight"], info["scale"]),
            out,
            c=None,
            disable_ue8m0_cast=not scale_ue8m0,
        )
        del a, a_scale, out
        return

    if kind == "fp8_fp4":
        from rtp_llm.models_py.kernels.cuda.deepgemm_wrapper import fp8_fp4_gemm_nt

        a, a_scale = _create_dummy_fp8_input(
            m_value=m_value,
            k_value=k_value,
            device=device,
            scale_ue8m0=True,
        )
        out = torch.empty((m_value, n_value), dtype=torch.bfloat16, device=device)
        fp8_fp4_gemm_nt(
            (a, a_scale),
            (info["weight"], info["scale"]),
            out,
            recipe_a=(1, 128),
            recipe_b=(1, 32),
        )
        del a, a_scale, out
        return

    raise ValueError(f"unknown dense GEMM warmup kind={kind!r}")


def _launch_dummy_batched_fp8_einsum(
    *,
    key: tuple[int, int, int],
    info: dict,
    m_value: int,
    device: torch.device,
) -> None:
    import deep_gemm

    groups, n_value, k_value = key
    a, a_scale = _create_dummy_batched_fp8_einsum_input(
        m_value=m_value,
        groups=groups,
        k_value=k_value,
        device=device,
    )
    out = torch.empty((m_value, groups, n_value), dtype=torch.bfloat16, device=device)
    deep_gemm.fp8_einsum(
        "bhr,hdr->bhd",
        (a, a_scale),
        (info["weight"], info["scale"]),
        out,
        recipe=(1, 1, 128),
    )
    del a, a_scale, out


def _launch_dummy_mhc_prenorm_gemm(
    *,
    key: tuple[int, int],
    info: dict,
    m_value: int,
    num_splits: int,
    device: torch.device,
) -> None:
    from rtp_llm.models_py.kernels.cuda.deepgemm_wrapper import tf32_hc_prenorm_gemm

    n_value, k_value = key
    x = torch.zeros((m_value, k_value), dtype=torch.bfloat16, device=device)
    out = torch.empty(
        (num_splits, m_value, n_value), dtype=torch.float32, device=device
    )
    sqrsum = torch.empty((num_splits, m_value), dtype=torch.float32, device=device)
    tf32_hc_prenorm_gemm(x, info["fn"], out, sqrsum, int(num_splits))
    del x, out, sqrsum


def _launch_dummy_fp8_mqa_logits(
    *,
    key: tuple[int, int],
    device: torch.device,
) -> None:
    import deep_gemm

    num_heads, head_dim = key
    block_q = max(1, 128 // int(num_heads))
    seq_len = max(1, block_q)
    seq_len_kv = 256

    q = torch.zeros(
        (seq_len, num_heads, head_dim), dtype=torch.float8_e4m3fn, device=device
    )
    k = torch.zeros((seq_len_kv, head_dim), dtype=torch.float8_e4m3fn, device=device)
    k_scale = torch.ones((seq_len_kv,), dtype=torch.float32, device=device)
    weights = torch.zeros((seq_len, num_heads), dtype=torch.float32, device=device)
    ks = torch.zeros((seq_len,), dtype=torch.int32, device=device)
    ke = torch.full((seq_len,), seq_len_kv, dtype=torch.int32, device=device)

    logits = deep_gemm.fp8_mqa_logits(
        q,
        (k, k_scale),
        weights,
        ks,
        ke,
        False,
        0,
    )
    del q, k, k_scale, weights, ks, ke, logits

"""Role-aware startup JIT warmup for Kimi K3.

The directed launches cover K3's Python JIT surface independently of the real
gRPC request warmup: DeepGEMM dense/MegaMoE layouts, AttnRes constexpr branches,
FP8 producers, and the role-local KDA Triton/cuLA paths.  CUDA extension kernels
such as collectives, routing, and FlashMLA remain covered by the normal engine
startup and full-request warmup because they do not have Python JIT cache keys.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Optional

import torch

from rtp_llm.models_py.modules.dsv4.dsv4_kernel_jit_warmup import (
    _run_triton_warmup_launch_with_retry,
    collect_dense_gemm_shapes,
    warmup_dense_gemm_jit,
)
from rtp_llm.models_py.modules.dsv4.moe.mega_jit_warmup import mega_moe_config_signature
from rtp_llm.models_py.modules.kimi_k3.utils import (
    collective_gemm_workspace_global_tokens,
    prefill_chunk_tokens,
)

_KIMI_K3_MEGA_WARMED_KEYS: set[tuple[Any, ...]] = set()
_KIMI_K3_TRITON_WARMED_KEYS: set[tuple[Any, ...]] = set()


@dataclass(frozen=True)
class KimiK3KernelWarmupPlan:
    """Role-local launch bounds used by K3's directed kernel warmup."""

    dense_max_m: int
    mega_max_tokens_per_rank: int
    mega_token_counts: tuple[int, ...]
    decode_batch_sizes: tuple[int, ...] = ()


@dataclass(frozen=True)
class KimiK3KernelWarmupTiming:
    kernel: str
    spec: str
    duration_s: float


class _WarmupTimings:
    def __init__(self, role: str, device: torch.device) -> None:
        self.role = role
        self.device = device
        self.records: list[KimiK3KernelWarmupTiming] = []

    def observe(self, kernel: str, spec: str, duration_s: float) -> None:
        record = KimiK3KernelWarmupTiming(kernel, spec, float(duration_s))
        self.records.append(record)
        logging.info(
            "[KimiK3 KernelWarmupTiming] role=%s rank=%d kernel=%s "
            "spec=%s duration_ms=%.3f",
            self.role,
            _distributed_rank(),
            kernel,
            spec,
            record.duration_s * 1000.0,
        )

    def launch(self, kernel: str, spec: str, launch: Callable[[], Any]) -> Any:
        torch.cuda.synchronize(self.device)
        begin = time.perf_counter()
        result = launch()
        torch.cuda.synchronize(self.device)
        self.observe(kernel, spec, time.perf_counter() - begin)
        return result

    def log_summary(self) -> None:
        totals: dict[str, float] = {}
        for record in self.records:
            totals[record.kernel] = totals.get(record.kernel, 0.0) + record.duration_s
        logging.info(
            "[KimiK3 KernelWarmupSummary] role=%s rank=%d launches=%d "
            "total_ms=%.3f per_kernel_ms=%s",
            self.role,
            _distributed_rank(),
            len(self.records),
            sum(totals.values()) * 1000.0,
            {name: round(seconds * 1000.0, 3) for name, seconds in totals.items()},
        )


def kimi_k3_kernel_jit_warmup_enabled() -> bool:
    flag = os.environ.get("KIMI_K3_STARTUP_REAL_WARMUP", "auto").strip().lower()
    return flag not in ("0", "false", "off", "no")


def _ceil_div(value: int, divisor: int) -> int:
    return (int(value) + int(divisor) - 1) // int(divisor)


def generate_kimi_k3_mega_moe_token_counts(
    *,
    num_ranks: int,
    num_experts: int,
    num_experts_per_rank: int,
    num_topk: int,
    intermediate_hidden: int,
    num_sms: int,
    max_tokens_per_rank: int,
) -> list[int]:
    """Return one token count per reachable DeepGEMM MegaMoE layout."""

    max_tokens = max(int(max_tokens_per_rank), 0)
    if max_tokens == 0:
        return []
    representatives: list[int] = []
    previous_signature: Optional[tuple[int, int, int]] = None
    for token_count in range(1, max_tokens + 1):
        signature = mega_moe_config_signature(
            num_ranks=num_ranks,
            num_experts=num_experts,
            num_experts_per_rank=num_experts_per_rank,
            num_tokens=token_count,
            num_topk=num_topk,
            intermediate_hidden=intermediate_hidden,
            num_sms=num_sms,
        )
        if signature != previous_signature:
            representatives.append(token_count)
            previous_signature = signature

    # K3 Prefill runs at the configured chunk cap. Exercise that exact final
    # launch while retaining only one representative for its heuristic key.
    if representatives:
        representatives[-1] = max_tokens
    return representatives


def _mega_moes(model: Any) -> tuple[Any, ...]:
    result = []
    seen: set[tuple[Any, ...]] = set()
    for module in model.modules():
        required = (
            "_mega_l1_w",
            "_mega_l2_w",
            "_mega_buf",
            "expert_num",
            "local_expert_count",
            "top_k",
            "latent_size",
            "_mega_intermediate_size",
        )
        if all(hasattr(module, name) for name in required):
            key = (
                module.__class__.__name__,
                int(module.ep_size),
                int(module.expert_num),
                int(module.local_expert_count),
                int(module.top_k),
                int(module.latent_size),
                int(module._mega_intermediate_size),
                int(module._mega_buf.num_max_tokens_per_rank),
                hasattr(module, "_mega_shared_hidden"),
            )
            if key not in seen:
                result.append(module)
                seen.add(key)
    return tuple(result)


def _first_mega_moe(model: Any) -> Optional[Any]:
    modules = _mega_moes(model)
    return modules[0] if modules else None


def _role_token_capacity(model: Any, init_resource: Any) -> tuple[int, int]:
    tp_size = max(int(model.parallelism_config.get_attn_tp_size()), 1)
    if bool(init_resource.is_decode_role):
        max_batch = max(
            int(
                getattr(
                    model,
                    "_max_generate_batch_size",
                    getattr(model, "_max_batch", 1),
                )
            ),
            int(getattr(init_resource, "max_decode_graph_batch_size", 1)),
        )
        tokens_per_batch = max(
            int(getattr(model.config, "gen_num_per_cycle", 0)) + 1, 1
        )
        global_tokens = max_batch * tokens_per_batch
    else:
        global_tokens = collective_gemm_workspace_global_tokens(
            int(model.config.max_seq_len),
            int(init_resource.max_context_batch_size),
            prefill_chunk_tokens(),
        )
    dense_max_m = max(_ceil_div(global_tokens, tp_size) * tp_size, 1)
    return dense_max_m, max(dense_max_m // tp_size, 1)


def build_kimi_k3_kernel_warmup_plan(
    model: Any,
    init_resource: Any,
    *,
    num_sms: Optional[int] = None,
) -> KimiK3KernelWarmupPlan:
    dense_max_m, local_token_capacity = _role_token_capacity(model, init_resource)
    decode_batch_sizes: tuple[int, ...] = ()
    if bool(init_resource.is_decode_role):
        configured = tuple(
            int(value)
            for value in getattr(init_resource, "decode_capture_batch_sizes", ())
            if int(value) > 0
        )
        decode_batch_sizes = tuple(
            sorted(
                set(
                    configured
                    or (int(getattr(init_resource, "max_decode_graph_batch_size", 1)),)
                )
            )
        )
    mega = _first_mega_moe(model)
    if mega is None:
        return KimiK3KernelWarmupPlan(dense_max_m, 0, (), decode_batch_sizes)

    buffer_capacity = int(mega._mega_buf.num_max_tokens_per_rank)
    mega_max_tokens = min(local_token_capacity, buffer_capacity)
    if num_sms is None:
        import deep_gemm

        num_sms = int(deep_gemm.get_num_sms())
    token_counts = generate_kimi_k3_mega_moe_token_counts(
        num_ranks=int(mega.ep_size),
        num_experts=int(mega.expert_num),
        num_experts_per_rank=int(mega.local_expert_count),
        num_topk=int(mega.top_k),
        intermediate_hidden=int(mega._mega_intermediate_size),
        num_sms=int(num_sms),
        max_tokens_per_rank=mega_max_tokens,
    )
    return KimiK3KernelWarmupPlan(
        dense_max_m=dense_max_m,
        mega_max_tokens_per_rank=mega_max_tokens,
        mega_token_counts=tuple(token_counts),
        decode_batch_sizes=decode_batch_sizes,
    )


def _distributed_rank() -> int:
    import torch.distributed as dist

    return int(dist.get_rank()) if dist.is_available() and dist.is_initialized() else 0


def _distributed_barrier() -> None:
    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def _kimi_k3_nvcc_rank_tmpdir(rank: int) -> str:
    root = (
        os.environ.get("DG_JIT_CACHE_DIR")
        or os.environ.get("TRITON_CACHE_DIR")
        or "/tmp"
    )
    return os.path.join(root, "rtp_llm_kimi_k3_mega_moe_nvcc", f"rank_{rank}")


def _activate_rank_tmpdir(rank: int) -> tuple[str, Optional[str]]:
    tmpdir = _kimi_k3_nvcc_rank_tmpdir(rank)
    try:
        os.makedirs(tmpdir, exist_ok=True)
    except Exception:
        tmpdir = os.path.join("/tmp", "rtp_llm_kimi_k3_mega_moe_nvcc", f"rank_{rank}")
        os.makedirs(tmpdir, exist_ok=True)
    previous = os.environ.get("TMPDIR")
    os.environ["TMPDIR"] = tmpdir
    return tmpdir, previous


def _restore_tmpdir(previous: Optional[str]) -> None:
    if previous is None:
        os.environ.pop("TMPDIR", None)
    else:
        os.environ["TMPDIR"] = previous


@torch.inference_mode()
def _warmup_mega_moe(
    module: Any,
    token_counts: Iterable[int],
    *,
    num_sms: int,
    timings: Optional[_WarmupTimings] = None,
) -> None:
    counts = tuple(sorted({int(value) for value in token_counts if int(value) > 0}))
    if not counts:
        return
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("Kimi K3 MegaMoE JIT warmup cannot run during capture")

    device = module._mega_l1_w.device
    fused_shared = hasattr(module, "_mega_shared_hidden")
    warmup_key = (
        module.__class__.__name__,
        int(module.ep_size),
        int(module.expert_num),
        int(module.local_expert_count),
        int(module.top_k),
        int(module.latent_size),
        int(module._mega_intermediate_size),
        int(num_sms),
        counts,
        str(device),
    )
    if warmup_key in _KIMI_K3_MEGA_WARMED_KEYS:
        return

    rank = _distributed_rank()
    if rank == 0:
        logging.info(
            "[KimiK3 MegaMoE] JIT warmup start: strategy=%s tokens=%s "
            "ep=%d experts=%d local_experts=%d topk=%d latent=%d "
            "intermediate=%d num_sms=%d",
            module.__class__.__name__,
            counts,
            int(module.ep_size),
            int(module.expert_num),
            int(module.local_expert_count),
            int(module.top_k),
            int(module.latent_size),
            int(module._mega_intermediate_size),
            int(num_sms),
        )

    maximum = max(counts)
    routed_input = torch.zeros(
        (maximum, int(module.latent_size)), dtype=torch.bfloat16, device=device
    )
    routing_weights = torch.zeros(
        (maximum, int(module.top_k)), dtype=torch.float32, device=device
    )
    local_start = int(module.ffn_tp_rank) * int(module.local_expert_count)
    local_ids = local_start + torch.arange(
        int(module.top_k), dtype=torch.long, device=device
    ) % max(int(module.local_expert_count), 1)
    expert_ids = local_ids.view(1, -1).expand(maximum, -1).contiguous()
    shared_input = (
        torch.zeros(
            (maximum, int(module._mega_shared_hidden)),
            dtype=torch.bfloat16,
            device=device,
        )
        if fused_shared
        else None
    )

    tmpdir, previous_tmpdir = _activate_rank_tmpdir(rank)
    try:
        if rank == 0:
            logging.info("[KimiK3 MegaMoE] rank-local JIT TMPDIR=%s", tmpdir)
        for token_count in counts:
            _distributed_barrier()

            def launch() -> None:
                if fused_shared:
                    module._deep_gemm_mega_expert_sum_with_shared(
                        routed_input[:token_count],
                        shared_input[:token_count],
                        expert_ids[:token_count],
                        routing_weights[:token_count],
                    )
                else:
                    module._deep_gemm_mega_expert_sum(
                        routed_input[:token_count],
                        expert_ids[:token_count],
                        routing_weights[:token_count],
                    )

            spec = (
                f"strategy={module.__class__.__name__} tokens={token_count} "
                f"experts={module.expert_num} local_experts={module.local_expert_count} "
                f"topk={module.top_k} latent={module.latent_size} "
                f"intermediate={module._mega_intermediate_size}"
            )
            if timings is None:
                launch()
                torch.cuda.synchronize(device)
            else:
                timings.launch("mega_moe", spec, launch)
        _distributed_barrier()
    finally:
        _restore_tmpdir(previous_tmpdir)
    _KIMI_K3_MEGA_WARMED_KEYS.add(warmup_key)
    if rank == 0:
        logging.info("[KimiK3 MegaMoE] JIT warmup done: tokens=%s", counts)


def _run_k3_triton_launch(
    timings: _WarmupTimings,
    kernel: str,
    spec: str,
    launch: Callable[[], Any],
) -> Any:
    return timings.launch(
        kernel,
        spec,
        lambda: _run_triton_warmup_launch_with_retry(
            f"KimiK3 {kernel}",
            spec,
            launch,
            device=timings.device,
            tmpdir_env=None,
            tmpdir_namespace="rtp_llm_kimi_k3_triton_warmup_tmp",
        ),
    )


def _warmup_attn_res_jit(model: Any, timings: _WarmupTimings) -> None:
    """Compile every AttnRes constexpr branch reachable by the live layers."""

    layers = tuple(getattr(model, "layers", ()) or ())
    if not layers:
        return
    hidden_size = int(model.config.hidden_size)
    block_count = int(getattr(model, "num_attn_res_blocks", 0) or 0)
    if block_count <= 0:
        return
    device = timings.device
    dtype = model.embedding_weight.dtype
    warmed: set[tuple[Any, ...]] = set()

    def run(
        module: Any,
        *,
        active_blocks: int,
        block_write_idx: int,
        has_delta: bool,
        output_norm_weight: torch.Tensor,
        output_norm_eps: float,
    ) -> None:
        token_rows = (2, 256) if active_blocks > 1 else (2,)
        for rows in token_rows:
            key = (
                module.__class__.__name__,
                hidden_size,
                active_blocks,
                block_write_idx,
                has_delta,
                True,
                rows >= 256,
                str(device),
            )
            if key in warmed or key in _KIMI_K3_TRITON_WARMED_KEYS:
                continue
            prefix = torch.zeros((rows, hidden_size), dtype=dtype, device=device)
            blocks = torch.zeros(
                (rows, block_count, hidden_size), dtype=dtype, device=device
            )
            delta = torch.zeros_like(prefix) if has_delta else None
            spec = (
                f"impl={module.__class__.__name__} rows={rows} hidden={hidden_size} "
                f"active_blocks={active_blocks} write={block_write_idx} "
                f"delta={int(has_delta)} output_norm=1"
            )
            _run_k3_triton_launch(
                timings,
                (
                    "attn_res_fp8"
                    if module.__class__.__name__ == "Fp8AttentionResidual"
                    else "attn_res_bf16"
                ),
                spec,
                lambda module=module, prefix=prefix, blocks=blocks, delta=delta: module(
                    prefix,
                    blocks,
                    output_norm_weight=output_norm_weight,
                    output_norm_eps=output_norm_eps,
                    delta=delta,
                    num_blocks=active_blocks,
                    block_write_idx=block_write_idx,
                ),
            )
            warmed.add(key)
            _KIMI_K3_TRITON_WARMED_KEYS.add(key)

    for layer in layers:
        if not hasattr(layer, "self_attention_residual"):
            continue
        layer_idx = int(layer.layer_idx)
        block_size = int(layer.attn_res_block_size)
        previous = min(
            (layer_idx + block_size - 1) // block_size,
            block_count,
        )
        writes = layer_idx % block_size == 0
        if previous > 0 or writes:
            run(
                layer.self_attention_residual,
                active_blocks=previous,
                block_write_idx=previous if writes and previous < block_count else -1,
                has_delta=False,
                output_norm_weight=layer.attention_norm.weight,
                output_norm_eps=float(layer.attention_norm.variance_epsilon),
            )
        run(
            layer.mlp_residual,
            active_blocks=previous + int(writes),
            block_write_idx=-1,
            has_delta=not writes,
            output_norm_weight=layer.mlp_norm.weight,
            output_norm_eps=float(layer.mlp_norm.variance_epsilon),
        )

    final_norm = getattr(model, "norm", None)
    if final_norm is not None and hasattr(final_norm, "attention_residual"):
        run(
            final_norm.attention_residual,
            active_blocks=block_count,
            block_write_idx=-1,
            has_delta=False,
            output_norm_weight=final_norm.final_norm.weight,
            output_norm_eps=float(final_norm.final_norm.variance_epsilon),
        )


def _warmup_fp8_producer_jit(model: Any, timings: _WarmupTimings, role: str) -> None:
    from rtp_llm.models_py.triton_kernels.kimi_kda.fp8_quant import (
        quantize_forget_latent_fp8,
    )

    device = timings.device
    dtype = torch.bfloat16
    for module in model.modules():
        cls_name = module.__class__.__name__
        if cls_name == "Fp8RMSNorm":
            k_value = int(module.weight.numel())
            key = (cls_name, k_value, bool(module.retain_bf16), str(device))
            if key in _KIMI_K3_TRITON_WARMED_KEYS:
                continue
            x = torch.zeros((2, k_value), dtype=dtype, device=device)
            spec = f"M=2 K={k_value} retain_bf16={int(module.retain_bf16)}"
            _run_k3_triton_launch(
                timings, "fp8_rmsnorm", spec, lambda module=module, x=x: module(x)
            )
            _KIMI_K3_TRITON_WARMED_KEYS.add(key)

        if cls_name == "KimiK3MLA" and getattr(module, "_fp8_enabled", False):
            k_value = int(module.local_heads) * int(module.value_dim)
            key = ("Fp8SigmoidGate", k_value, str(device))
            if key not in _KIMI_K3_TRITON_WARMED_KEYS:
                x = torch.zeros((2, k_value), dtype=dtype, device=device)
                gate = torch.zeros_like(x)
                _run_k3_triton_launch(
                    timings,
                    "fp8_sigmoid_gate",
                    f"M=2 K={k_value}",
                    lambda module=module, x=x, gate=gate: module.output_gate_op(
                        x, gate
                    ),
                )
                _KIMI_K3_TRITON_WARMED_KEYS.add(key)

        if cls_name != "KimiK3KDA" or not getattr(module, "_fp8_enabled", False):
            continue
        heads, head_dim = int(module.local_heads), int(module.head_dim)
        key = ("Fp8KdaOutputNorm", role, heads, head_dim, str(device))
        if key not in _KIMI_K3_TRITON_WARMED_KEYS:
            output = torch.zeros((1, 2, heads, head_dim), dtype=dtype, device=device)
            gate = torch.zeros_like(output)
            _run_k3_triton_launch(
                timings,
                f"fp8_kda_output_{role.lower()}",
                f"B=1 T=2 H={heads} D={head_dim}",
                lambda module=module, output=output, gate=gate: module.output_norm(
                    output, gate, role.lower()
                ),
            )
            _KIMI_K3_TRITON_WARMED_KEYS.add(key)
        if getattr(module, "_fp8_strided_forget", False):
            key = ("Fp8ForgetLatent", str(device))
            if key not in _KIMI_K3_TRITON_WARMED_KEYS:
                backing = torch.zeros((2, 256), dtype=dtype, device=device)
                forget = backing[:, 64:192]
                _run_k3_triton_launch(
                    timings,
                    "fp8_forget_latent",
                    f"M=2 K=128 row_stride={forget.stride(0)}",
                    lambda forget=forget: quantize_forget_latent_fp8(forget),
                )
                _KIMI_K3_TRITON_WARMED_KEYS.add(key)


def _first_kda(model: Any, role: str) -> Optional[Any]:
    for module in model.modules():
        if module.__class__.__name__ != "KimiK3KDA":
            continue
        if role == "Prefill" and getattr(module, "prefill_executor", None) is not None:
            return module
        if role == "Decode" and getattr(module, "decode_executor", None) is not None:
            return module
    return None


def _warmup_kda_prefill_jit(
    model: Any, module: Any, page_size: int, timings: _WarmupTimings
) -> None:
    from rtp_llm.models_py.modules.kimi_k3.kda.prefill import (
        prepare_kimi_kda_prefill_metadata,
    )
    from rtp_llm.models_py.triton_kernels.kimi_kda import (
        kimi_kda_load_recurrent_state,
        kimi_kda_short_conv_paged_prefill,
        kimi_kda_store_recurrent_checkpoints,
    )

    device = timings.device
    tokens = 64
    projection = int(module.projection_size)
    heads, head_dim = int(module.local_heads), int(module.head_dim)
    history = int(module.history_size)
    key = ("kda_prefill", projection, heads, head_dim, history, page_size, str(device))
    if key in _KIMI_K3_TRITON_WARMED_KEYS:
        return
    cu_host = torch.tensor([0, tokens], dtype=torch.int32)
    lengths_host = torch.tensor([tokens], dtype=torch.int32)
    prefixes_host = torch.tensor([0], dtype=torch.int32)
    metadata = prepare_kimi_kda_prefill_metadata(
        cu_host,
        lengths_host,
        prefixes_host,
        checkpoint_tokens=page_size,
        local_heads=heads,
        head_dim=head_dim,
        device=device,
    )
    mixed = torch.zeros((tokens, 3 * projection), dtype=torch.bfloat16, device=device)
    conv_cache = torch.zeros(
        (3, history, 3 * projection), dtype=torch.bfloat16, device=device
    )
    block_map = torch.tensor([[1, 2]], dtype=torch.int32, device=device)
    prefixes = prefixes_host.to(device=device)
    cu = cu_host.to(device=device)
    executor = module.prefill_executor
    assert executor is not None

    q, k, v, _ = _run_k3_triton_launch(
        timings,
        "kda_short_conv_prefill",
        f"tokens={tokens} projection={projection} page={page_size} continuation=0",
        lambda: kimi_kda_short_conv_paged_prefill(
            mixed,
            executor.fused_conv,
            conv_cache,
            block_map,
            prefixes,
            cu,
            page_size,
            metadata.conv,
        ),
    )
    current = torch.zeros(
        (1, history, 3 * projection), dtype=torch.bfloat16, device=device
    )
    continuation = torch.ones((1,), dtype=torch.bool, device=device)
    _run_k3_triton_launch(
        timings,
        "kda_short_conv_prefill",
        f"tokens={tokens} projection={projection} page={page_size} continuation=1",
        lambda: kimi_kda_short_conv_paged_prefill(
            mixed,
            executor.fused_conv,
            conv_cache,
            block_map,
            prefixes,
            cu,
            page_size,
            metadata.conv,
            current_conv_state=current,
            continuation_mask=continuation,
            return_final_state=True,
        ),
    )
    ssm_cache = torch.zeros(
        (3, heads, head_dim, head_dim), dtype=torch.float32, device=device
    )
    initial_state = _run_k3_triton_launch(
        timings,
        "kda_recurrent_cache_load",
        f"batch=1 H={heads} D={head_dim} page={page_size}",
        lambda: kimi_kda_load_recurrent_state(
            prefixes, block_map, ssm_cache, page_size
        ),
    )
    checkpoint_states = metadata.recurrent_checkpoints
    _run_k3_triton_launch(
        timings,
        "kda_cula_prefill",
        f"tokens={tokens} H={heads} D={head_dim} checkpoint={page_size}",
        lambda: executor._cula_checkpoint_prefill(
            q.reshape(1, tokens, heads, head_dim),
            k.reshape(1, tokens, heads, head_dim),
            v.reshape(1, tokens, heads, head_dim),
            torch.zeros(
                (1, tokens, heads, head_dim), dtype=torch.bfloat16, device=device
            ),
            torch.zeros((1, tokens, heads), dtype=torch.bfloat16, device=device),
            initial_state,
            cu_seqlens=cu,
            cu_seqlens_cpu=cu_host,
            checkpoint_interval=page_size,
            checkpoint_states=checkpoint_states,
        ),
    )
    _run_k3_triton_launch(
        timings,
        "kda_recurrent_cache_store",
        f"checkpoints={metadata.recurrent.total_checkpoints} H={heads} D={head_dim}",
        lambda: kimi_kda_store_recurrent_checkpoints(
            checkpoint_states, metadata.recurrent, block_map, ssm_cache
        ),
    )
    _KIMI_K3_TRITON_WARMED_KEYS.add(key)


def _warmup_kda_decode_jit(
    model: Any,
    module: Any,
    page_size: int,
    batch_sizes: Iterable[int],
    timings: _WarmupTimings,
) -> None:
    from rtp_llm.models_py.triton_kernels.kimi_kda import (
        fused_recurrent_kda,
        kimi_kda_short_conv_paged_decode,
        kimi_kda_short_conv_paged_target_verify,
    )
    from rtp_llm.utils.model_weight import W

    device = timings.device
    projection = int(module.projection_size)
    heads, head_dim = int(module.local_heads), int(module.head_dim)
    history = int(module.history_size)
    executor = module.decode_executor
    assert executor is not None
    for batch in sorted({int(value) for value in batch_sizes if int(value) > 0}):
        key = (
            "kda_decode",
            batch,
            projection,
            heads,
            head_dim,
            history,
            page_size,
            str(device),
        )
        if key in _KIMI_K3_TRITON_WARMED_KEYS:
            continue
        q = torch.zeros((batch, projection), dtype=torch.bfloat16, device=device)
        k = torch.zeros_like(q)
        v = torch.zeros_like(q)
        conv_cache = torch.zeros(
            (batch + 1, history, 3 * projection),
            dtype=torch.bfloat16,
            device=device,
        )
        block_map = torch.arange(1, batch + 1, dtype=torch.int32, device=device).view(
            batch, 1
        )
        sequence_lengths = torch.ones((batch,), dtype=torch.int32, device=device)
        q_conv, k_conv, v_conv = _run_k3_triton_launch(
            timings,
            "kda_short_conv_decode",
            f"batch={batch} projection={projection} page={page_size}",
            lambda: kimi_kda_short_conv_paged_decode(
                q,
                k,
                v,
                executor.fused_conv,
                conv_cache,
                block_map,
                sequence_lengths,
                page_size,
            ),
        )
        state = torch.zeros(
            (batch + 1, heads, head_dim, head_dim),
            dtype=torch.float32,
            device=device,
        )
        shape = (batch, 1, heads, head_dim)
        _run_k3_triton_launch(
            timings,
            "kda_fused_recurrent_decode",
            f"batch={batch} T=1 H={heads} D={head_dim} page={page_size}",
            lambda: fused_recurrent_kda(
                q_conv.reshape(shape),
                k_conv.reshape(shape),
                v_conv.reshape(shape),
                torch.zeros(shape, dtype=torch.bfloat16, device=device),
                torch.zeros((batch, 1, heads), dtype=torch.bfloat16, device=device),
                initial_state=state,
                A_log=executor.weights[W.linear_attn_alog],
                dt_bias=executor.weights[W.linear_attn_dt_b_kda],
                inplace_final_state=True,
                use_qk_l2norm_in_kernel=True,
                use_gate_in_kernel=True,
                use_beta_sigmoid_in_kernel=True,
                lower_bound=executor.gate_lower_bound,
                cu_seqlens=torch.arange(batch + 1, dtype=torch.int32, device=device),
                block_map=block_map,
                seq_size_per_block=page_size,
                sequence_lengths=sequence_lengths,
                state_v_first=False,
            ),
        )
        verify_tokens = max(int(getattr(model.config, "gen_num_per_cycle", 0)) + 1, 1)
        q_verify = q[:, None, :].expand(batch, verify_tokens, projection).contiguous()
        k_verify = torch.zeros_like(q_verify)
        v_verify = torch.zeros_like(q_verify)
        _run_k3_triton_launch(
            timings,
            "kda_short_conv_target_verify",
            f"batch={batch} T={verify_tokens} projection={projection} page={page_size}",
            lambda: kimi_kda_short_conv_paged_target_verify(
                q_verify,
                k_verify,
                v_verify,
                executor.fused_conv,
                conv_cache,
                block_map,
                sequence_lengths,
                page_size,
            ),
        )
        _KIMI_K3_TRITON_WARMED_KEYS.add(key)


def _warmup_kda_jit(
    model: Any,
    plan: KimiK3KernelWarmupPlan,
    role: str,
    timings: _WarmupTimings,
) -> None:
    module = _first_kda(model, role)
    page_size = int(getattr(model, "_kda_checkpoint_tokens", 0) or 0)
    if module is None or page_size <= 0:
        return
    if role == "Prefill":
        _warmup_kda_prefill_jit(model, module, page_size, timings)
    else:
        _warmup_kda_decode_jit(
            model, module, page_size, plan.decode_batch_sizes, timings
        )


@torch.inference_mode()
def warmup_kimi_k3_kernel_jit(model: Any, init_resource: Any) -> None:
    """Compile the complete role-local K3 Python JIT surface."""

    if not kimi_k3_kernel_jit_warmup_enabled():
        return
    device = model.embedding_weight.device
    if device.type != "cuda" or not torch.cuda.is_available():
        return
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("Kimi K3 kernel JIT warmup cannot run during capture")

    import deep_gemm

    num_sms = int(deep_gemm.get_num_sms())
    plan = build_kimi_k3_kernel_warmup_plan(model, init_resource, num_sms=num_sms)
    role = "Decode" if bool(init_resource.is_decode_role) else "Prefill"
    logging.info(
        "[KimiK3 KernelWarmup] role=%s dense_max_m=%d "
        "mega_max_tokens_per_rank=%d mega_token_counts=%s decode_batches=%s",
        role,
        plan.dense_max_m,
        plan.mega_max_tokens_per_rank,
        plan.mega_token_counts,
        plan.decode_batch_sizes,
    )
    begin = time.perf_counter()
    timings = _WarmupTimings(role, device)
    _warmup_attn_res_jit(model, timings)
    _warmup_fp8_producer_jit(model, timings, role)
    _warmup_kda_jit(model, plan, role, timings)
    dense_shapes = collect_dense_gemm_shapes(model, label="KimiK3")
    warmup_dense_gemm_jit(
        dense_shapes,
        max_m=plan.dense_max_m,
        device=device,
        label="KimiK3",
        launch_observer=timings.observe,
    )
    _, local_token_capacity = _role_token_capacity(model, init_resource)
    for mega in _mega_moes(model):
        max_tokens = min(
            local_token_capacity, int(mega._mega_buf.num_max_tokens_per_rank)
        )
        token_counts = generate_kimi_k3_mega_moe_token_counts(
            num_ranks=int(mega.ep_size),
            num_experts=int(mega.expert_num),
            num_experts_per_rank=int(mega.local_expert_count),
            num_topk=int(mega.top_k),
            intermediate_hidden=int(mega._mega_intermediate_size),
            num_sms=num_sms,
            max_tokens_per_rank=max_tokens,
        )
        _warmup_mega_moe(mega, token_counts, num_sms=num_sms, timings=timings)
    torch.cuda.synchronize(device)
    timings.log_summary()
    logging.info(
        "[KimiK3 KernelWarmup] role=%s done in %.2fs",
        role,
        time.perf_counter() - begin,
    )


__all__ = [
    "KimiK3KernelWarmupPlan",
    "KimiK3KernelWarmupTiming",
    "build_kimi_k3_kernel_warmup_plan",
    "generate_kimi_k3_mega_moe_token_counts",
    "kimi_k3_kernel_jit_warmup_enabled",
    "warmup_kimi_k3_kernel_jit",
]

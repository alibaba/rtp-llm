from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import statistics
import tempfile
import threading
import traceback
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Literal, Optional, Sequence

import torch

from rtp_llm.config.engine_config import EngineConfig
from rtp_llm.config.model_config import ModelConfig
from rtp_llm.config.py_config_modules import PyEnvConfigs
from rtp_llm.models_py.modules.factory.attention.cuda_impl.test.base_attention_test import (
    fill_paged_kv_cache,
)
from rtp_llm.models_py.modules.factory.attention.cuda_impl.test.bench_utils import (
    attention_tflops_per_sec_with_actual_seq_lens,
    bench_gpu_time_with_cudagraph,
)
from rtp_llm.ops import AttentionConfigs, KvCacheDataType, ParallelismConfig, RopeStyle
from rtp_llm.ops.compute_ops import (
    LayerKVCache,
    PyAttentionInputs,
    get_typemeta,
    init_exec_ctx,
)
from rtp_llm.test.utils.cuda_graph_util import capture_graph
from rtp_llm.test.utils.numeric_util import assert_close_with_mismatch_tolerance

Q_DTYPES = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
}
KV_DTYPES = {"bf16", "fp16", "fp8"}
PRESETS = {
    "qwen3-8b": (32, 8, 128),
    "llama3-8b": (32, 8, 128),
}

_LOG_LOCK = threading.Lock()


def log(message: str) -> None:
    with _LOG_LOCK:
        print(message, flush=True)


def short_error(error: Exception, max_len: int = 240) -> str:
    message = (
        str(error).strip().splitlines()[0]
        if str(error).strip()
        else type(error).__name__
    )
    return message if len(message) <= max_len else message[:max_len] + "..."


@dataclass(frozen=True)
class BenchOptions:
    q_dtypes: tuple[str, ...]
    kv_dtypes: tuple[str, ...]
    batch_size: int
    seq_lens: tuple[int, ...]
    reuse_cache_ratios: tuple[float, ...]
    head_num: int
    kv_head_num: int
    head_dim: int
    page_size: int
    repeat_replays: int
    run_impl_patterns: tuple[str, ...]
    check_correctness: bool
    profile: bool
    profile_dir: str
    profile_iters: int
    enable_csv_dump: bool
    csv_output_path: str
    worker_timeout_s: int

    @classmethod
    def from_args(cls, args: Any) -> "BenchOptions":
        q_dtypes = tuple(
            value.strip().lower() for value in args.q_dtype.split(",") if value.strip()
        )
        if not q_dtypes or any(value not in Q_DTYPES for value in q_dtypes):
            raise ValueError("--q-dtype accepts: bf16, fp16")
        if len(set(q_dtypes)) != len(q_dtypes):
            raise ValueError("--q-dtype contains duplicate values")

        kv_dtypes = tuple(
            value.strip().lower()
            for value in args.kv_cache_dtype.split(",")
            if value.strip()
        )
        if not kv_dtypes or any(value not in KV_DTYPES for value in kv_dtypes):
            raise ValueError("--kv-cache-dtype accepts: bf16, fp16, fp8")
        if len(set(kv_dtypes)) != len(kv_dtypes):
            raise ValueError("--kv-cache-dtype contains duplicate values")

        seq_lens = tuple(
            int(value) for value in args.input_len.split(",") if value.strip()
        )
        ratios = tuple(
            float(value) for value in args.reuse_cache_ratio.split(",") if value.strip()
        )
        if not seq_lens or any(value <= 0 for value in seq_lens):
            raise ValueError("--input-len values must be positive")
        if not ratios or any(value < 0 or value >= 1 for value in ratios):
            raise ValueError("--reuse-cache-ratio values must be in [0, 1)")

        if args.preset == "custom":
            if not args.head_num or not args.kv_head_num or not args.head_dim:
                raise ValueError(
                    "custom preset requires --head-num, --kv-head-num, and --head-dim"
                )
            head_num, kv_head_num, head_dim = (
                args.head_num,
                args.kv_head_num,
                args.head_dim,
            )
        else:
            preset = PRESETS[args.preset]
            head_num = args.head_num or preset[0]
            kv_head_num = args.kv_head_num or preset[1]
            head_dim = args.head_dim or preset[2]

        if args.batch_size <= 0 or args.page_size <= 0:
            raise ValueError("--batch-size and --page-size must be positive")
        if args.repeat <= 0:
            raise ValueError("--repeat must be positive")
        if args.worker_timeout <= 0:
            raise ValueError("--worker-timeout must be positive")
        if args.profile_iters <= 0:
            raise ValueError("--profile-iters must be positive")
        if args.profile and not args.profile_dir.strip():
            raise ValueError(
                "--profile-dir must not be empty when profiling is enabled"
            )
        if args.enable_csv_dump and not args.csv_output_path.strip():
            raise ValueError(
                "--csv-output-path must not be empty when CSV dump is enabled"
            )

        run_filter = args.run_impls.strip().lower()
        patterns = (
            ()
            if run_filter in ("", "all")
            else tuple(
                value.strip() for value in run_filter.split(",") if value.strip()
            )
        )
        return cls(
            q_dtypes=q_dtypes,
            kv_dtypes=kv_dtypes,
            batch_size=args.batch_size,
            seq_lens=seq_lens,
            reuse_cache_ratios=ratios,
            head_num=head_num,
            kv_head_num=kv_head_num,
            head_dim=head_dim,
            page_size=args.page_size,
            repeat_replays=args.repeat,
            run_impl_patterns=patterns,
            check_correctness=args.check_correctness,
            profile=args.profile,
            profile_dir=args.profile_dir,
            profile_iters=args.profile_iters,
            enable_csv_dump=args.enable_csv_dump,
            csv_output_path=args.csv_output_path,
            worker_timeout_s=args.worker_timeout,
        )


@dataclass(frozen=True)
class BenchCase:
    case_id: str
    seed: int
    q_dtype: str
    kv_dtype: str
    batch_size: int
    seq_len: int
    reuse_cache_ratio: float
    head_num: int
    kv_head_num: int
    head_dim: int
    page_size: int

    @property
    def prefix_len(self) -> int:
        return int(self.seq_len * self.reuse_cache_ratio)

    @property
    def input_len(self) -> int:
        return self.seq_len - self.prefix_len

    @property
    def total_tokens(self) -> int:
        return self.input_len * self.batch_size

    @property
    def mode_tag(self) -> str:
        return (
            "plain"
            if self.prefix_len == 0
            else f"reuse{int(self.reuse_cache_ratio * 100)}%"
        )

    @classmethod
    def create(
        cls,
        options: BenchOptions,
        q_dtype: str,
        kv_dtype: str,
        seq_len: int,
        ratio: float,
    ) -> "BenchCase":
        identity = (
            f"q={q_dtype},kv={kv_dtype},bs={options.batch_size},seq={seq_len},"
            f"reuse={ratio},h={options.head_num},hk={options.kv_head_num},d={options.head_dim},p={options.page_size}"
        )
        seed = int.from_bytes(
            hashlib.sha256(identity.encode("ascii")).digest()[:4], "little"
        )
        return cls(
            case_id=identity,
            seed=seed,
            q_dtype=q_dtype,
            kv_dtype=kv_dtype,
            batch_size=options.batch_size,
            seq_len=seq_len,
            reuse_cache_ratio=ratio,
            head_num=options.head_num,
            kv_head_num=options.kv_head_num,
            head_dim=options.head_dim,
            page_size=options.page_size,
        )


@dataclass(frozen=True)
class PlannedCase:
    case: BenchCase
    impl_names: tuple[str, ...]


class CasePlanner:
    def __init__(self, options: BenchOptions, impl_benches: Sequence[Any]) -> None:
        self.options = options
        self.impl_benches = tuple(
            impl_bench
            for impl_bench in impl_benches
            if not options.run_impl_patterns
            or any(
                pattern in impl_bench.impl.__name__.lower()
                for pattern in options.run_impl_patterns
            )
        )
        self.generated_case_count = 0
        self.scheduled_run_count = 0
        self.rejected_run_count = 0

    def plan(self) -> list[PlannedCase]:
        planned = []
        self.generated_case_count = 0
        self.scheduled_run_count = 0
        self.rejected_run_count = 0
        for q_dtype in self.options.q_dtypes:
            for kv_dtype in self.options.kv_dtypes:
                for seq_len in self.options.seq_lens:
                    for ratio in self.options.reuse_cache_ratios:
                        case = BenchCase.create(
                            self.options, q_dtype, kv_dtype, seq_len, ratio
                        )
                        self.generated_case_count += 1
                        impl_names = []
                        for impl_bench in self.impl_benches:
                            try:
                                supported = impl_bench.should_run_case(case)
                            except Exception as error:
                                raise RuntimeError(
                                    f"{impl_bench.impl.__name__}.should_run_case() failed for {case.case_id}: "
                                    f"{short_error(error)}"
                                ) from error
                            if supported:
                                impl_names.append(impl_bench.impl.__name__)
                            else:
                                self.rejected_run_count += 1
                        if impl_names:
                            self.scheduled_run_count += len(impl_names)
                            planned.append(PlannedCase(case, tuple(impl_names)))
        planned.sort(key=lambda item: item.case.seq_len, reverse=True)
        return planned


@dataclass(frozen=True)
class ReferencePolicy:
    hybrid: bool = False
    fp8_kv: bool = False
    quantize_q: bool = False


@dataclass(frozen=True)
class Tolerance:
    atol: float
    rtol: float


@dataclass
class CaseData:
    case: BenchCase
    attn_inputs: PyAttentionInputs
    block_ids: torch.Tensor
    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    qkv: torch.Tensor
    _references: dict[ReferencePolicy, torch.Tensor] = field(default_factory=dict)

    @classmethod
    def create(cls, case: BenchCase, device: torch.device) -> "CaseData":
        cpu_generator = torch.Generator(device="cpu").manual_seed(case.seed)
        cuda_generator = torch.Generator(device=device).manual_seed(case.seed)
        input_lens = [case.input_len] * case.batch_size
        prefix_lens = [case.prefix_len] * case.batch_size
        total_lens = [case.seq_len] * case.batch_size

        pages_per_batch = [math.ceil(length / case.page_size) for length in total_lens]
        total_pages = sum(pages_per_batch)
        block_ids = torch.zeros(
            (case.batch_size, max(pages_per_batch)), dtype=torch.int32
        )
        permutation = torch.randperm(
            total_pages, generator=cpu_generator, dtype=torch.int64
        ).to(torch.int32)
        offset = 0
        for batch_idx, page_count in enumerate(pages_per_batch):
            block_ids[batch_idx, :page_count] = permutation[
                offset : offset + page_count
            ]
            offset += page_count

        def pinned(values: list[int]) -> torch.Tensor:
            return torch.tensor(values, dtype=torch.int32).pin_memory()

        def cumulative(values: list[int]) -> torch.Tensor:
            result = [0]
            for value in values:
                result.append(result[-1] + value)
            return pinned(result)

        inputs = PyAttentionInputs()
        inputs.is_prefill = True
        # The bench replays the same input objects. Engine graph mode instead
        # requires implementations to accept inputs that change between replays.
        inputs.is_cuda_graph = False
        inputs.input_lengths = pinned(input_lens)
        inputs.sequence_lengths = pinned(total_lens)
        inputs.prefix_lengths = pinned(prefix_lens)
        block_ids_device = block_ids.to(device)
        inputs.kv_cache_block_id = block_ids
        inputs.kv_cache_block_id_device = block_ids_device
        inputs.kv_cache_kernel_block_id = block_ids
        inputs.kv_cache_kernel_block_id_device = block_ids_device
        inputs.cu_seqlens = cumulative(input_lens)
        inputs.cu_seqlens_device = inputs.cu_seqlens.to(device, non_blocking=True)
        inputs.cu_kv_seqlens_device = cumulative(total_lens).to(
            device, non_blocking=True
        )
        inputs.total_tokens = case.total_tokens
        inputs.context_total_kv_length = sum(total_lens)
        q_dtype = Q_DTYPES[case.q_dtype]
        inputs.dtype = get_typemeta(torch.empty(1, dtype=q_dtype))

        shape_q = (case.batch_size, case.input_len, case.head_num, case.head_dim)
        shape_kv = (case.batch_size, case.seq_len, case.kv_head_num, case.head_dim)
        q = (
            torch.rand(shape_q, dtype=q_dtype, device=device, generator=cuda_generator)
            * 2
            - 1
        )
        k = (
            torch.rand(shape_kv, dtype=q_dtype, device=device, generator=cuda_generator)
            * 2
            - 1
        )
        v = (
            torch.rand(shape_kv, dtype=q_dtype, device=device, generator=cuda_generator)
            * 2
            - 1
        )
        q_flat = q.reshape(case.total_tokens, case.head_num * case.head_dim)
        k_flat = k[:, case.prefix_len :].reshape(
            case.total_tokens, case.kv_head_num * case.head_dim
        )
        v_flat = v[:, case.prefix_len :].reshape(
            case.total_tokens, case.kv_head_num * case.head_dim
        )
        qkv = torch.cat((q_flat, k_flat, v_flat), dim=-1).contiguous()
        return cls(
            case=case,
            attn_inputs=inputs,
            block_ids=block_ids,
            q=q,
            k=k,
            v=v,
            qkv=qkv,
        )

    def make_attn_configs(self) -> AttentionConfigs:
        config = AttentionConfigs()
        config.head_num = self.case.head_num
        config.kv_head_num = self.case.kv_head_num
        config.size_per_head = self.case.head_dim
        config.tokens_per_block = self.case.page_size
        config.kernel_tokens_per_block = self.case.page_size
        config.use_mla = False
        config.is_causal = True
        config.need_rope_kv_cache = False
        config.rope_config.style = RopeStyle.No
        config.dtype = Q_DTYPES[self.case.q_dtype]
        config.kv_cache_dtype = (
            KvCacheDataType.FP8 if self.case.kv_dtype == "fp8" else KvCacheDataType.BASE
        )
        config.max_seq_len = max(self.case.seq_len, 8192)
        return config

    def make_cache(self, prefix_only: bool) -> LayerKVCache:
        case = self.case
        fill_len = case.prefix_len if prefix_only else case.seq_len
        cache_dtype = (
            torch.float8_e4m3fn if case.kv_dtype == "fp8" else Q_DTYPES[case.q_dtype]
        )
        # Page ids come from a random permutation over the full-sequence page
        # pool, so a page id may exceed what fill_len alone would allocate.
        # Thus total_pages must be max + 1.
        return fill_paged_kv_cache(
            self.k,
            self.v,
            [fill_len] * case.batch_size,
            self.block_ids,
            case.page_size,
            case.kv_head_num,
            case.head_dim,
            cache_dtype,
            self.qkv.device,
            total_pages=int(self.block_ids.max().item()) + 1,
        )

    def reference(self, policy: ReferencePolicy) -> torch.Tensor:
        if policy in self._references:
            return self._references[policy]

        from flashinfer import single_prefill_with_kv_cache
        from flashinfer.cascade import merge_state

        outputs = []
        for batch_idx in range(self.case.batch_size):
            q = self.q[batch_idx]
            k = self.k[batch_idx]
            v = self.v[batch_idx]
            if policy.fp8_kv:
                k = k.to(torch.float8_e4m3fn).to(k.dtype)
                v = v.to(torch.float8_e4m3fn).to(v.dtype)

            if policy.hybrid and policy.quantize_q:
                q_quant = q.to(torch.float8_e4m3fn).to(q.dtype)
                prefix_out, prefix_lse = single_prefill_with_kv_cache(
                    q_quant,
                    k[: self.case.prefix_len],
                    v[: self.case.prefix_len],
                    causal=False,
                    kv_layout="NHD",
                    return_lse=True,
                )
                new_out, new_lse = single_prefill_with_kv_cache(
                    q_quant,
                    k[self.case.prefix_len :],
                    v[self.case.prefix_len :],
                    causal=True,
                    kv_layout="NHD",
                    return_lse=True,
                )
                output = merge_state(new_out, new_lse, prefix_out, prefix_lse)[0]
            else:
                q_input = (
                    q.to(torch.float8_e4m3fn).to(q.dtype) if policy.quantize_q else q
                )
                output = single_prefill_with_kv_cache(
                    q_input, k, v, causal=True, kv_layout="NHD"
                )
            outputs.append(output)

        reference = torch.stack(outputs).reshape(
            self.case.total_tokens, self.case.head_num, self.case.head_dim
        )
        self._references[policy] = reference
        return reference


@dataclass
class PreparedRun:
    reference_output: Optional[torch.Tensor]
    tolerance: Tolerance
    invoke: Callable[[], torch.Tensor]
    output: Optional[torch.Tensor] = None

    def forward(self) -> None:
        self.output = self.invoke()

    def normalized_output(self, case: BenchCase) -> torch.Tensor:
        if self.output is None:
            raise RuntimeError("forward produced no output")
        if self.output.dim() == 2:
            return self.output.reshape(case.total_tokens, case.head_num, case.head_dim)
        return self.output


# Literal adds static checking while preserving string JSON/CSV values.
BenchStatus = Literal["PASS", "FAIL", "MISMATCH", "SKIP", "PROFILED"]


@dataclass
class BenchResult:
    case: BenchCase
    impl_name: str
    status: BenchStatus
    effective_kv_dtype: Optional[str]
    mean_ms: Optional[float] = None
    p50_ms: Optional[float] = None
    p95_ms: Optional[float] = None
    p99_ms: Optional[float] = None
    tflops: Optional[float] = None
    tolerance_ratio: Optional[float] = None
    atol: Optional[float] = None
    rtol: Optional[float] = None
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "BenchResult":
        value = dict(value)
        value["case"] = BenchCase(**value["case"])
        return cls(**value)


class UnsupportedImpl(Exception):
    pass


class BenchRunner:
    def __init__(self, options: BenchOptions, parallelism: ParallelismConfig) -> None:
        self.options = options
        self.parallelism = parallelism

    def run(self, impl_bench: Any, case_data: CaseData) -> BenchResult:
        case = case_data.case
        base = dict(
            case=case,
            impl_name=impl_bench.impl.__name__,
            effective_kv_dtype=case.kv_dtype,
        )
        try:
            prepared = impl_bench.prepare(
                case_data, self.parallelism, self.options.check_correctness
            )
        except UnsupportedImpl as error:
            return BenchResult(status="SKIP", note=str(error), **base)
        except Exception as error:
            return BenchResult(
                status="FAIL",
                note=f"prepare: {short_error(error)}\n{traceback.format_exc()}",
                **base,
            )

        graph = None
        samples_ms: tuple[float, ...] = ()
        if self.options.profile:
            try:
                graph = capture_graph(prepared.forward, num_warmups=3)
            except Exception as error:
                return BenchResult(
                    status="FAIL",
                    note=f"graph_capture: {short_error(error)}\n{traceback.format_exc()}",
                    **base,
                )
        else:
            try:
                # Every impl is timed inside a graph, including ones whose
                # support_cuda_graph() is False. That is sound here: the bench
                # replays one fixed input, so the graph only serves to strip host
                # launch overhead from the kernel timing. It is not a claim that
                # the impl is graph-capturable under the engine's varying inputs.
                samples_ms = tuple(
                    bench_gpu_time_with_cudagraph(
                        prepared.forward,
                        dry_run_iters=0,
                        repeat_iters=self.options.repeat_replays,
                        num_iters_within_graph=1,
                    )
                )
            except Exception as error:
                return BenchResult(
                    status="FAIL",
                    note=f"graph_measure: {short_error(error)}\n{traceback.format_exc()}",
                    **base,
                )

        tolerance_ratio = None
        if prepared.reference_output is not None:
            try:
                output = prepared.normalized_output(case).float()
                reference = prepared.reference_output.float()
                diff = (output - reference).abs()
                denominator = (
                    prepared.tolerance.atol + prepared.tolerance.rtol * reference.abs()
                )
                tolerance_ratio = float((diff / denominator).max().item())
                assert_close_with_mismatch_tolerance(
                    output,
                    reference,
                    atol=prepared.tolerance.atol,
                    rtol=prepared.tolerance.rtol,
                )
            except AssertionError as error:
                return BenchResult(
                    status="MISMATCH",
                    tolerance_ratio=tolerance_ratio,
                    atol=prepared.tolerance.atol,
                    rtol=prepared.tolerance.rtol,
                    note=str(error),
                    **base,
                )
            except Exception as error:
                return BenchResult(
                    status="FAIL",
                    note=f"correctness: {short_error(error)}\n{traceback.format_exc()}",
                    **base,
                )

        if self.options.profile:
            prefix = (
                f"{impl_bench.impl.__name__}_{case.q_dtype}_{case.kv_dtype}_seq{case.seq_len}_"
                f"prefix{case.prefix_len}_"
            )
            path = ""
            try:
                os.makedirs(self.options.profile_dir, exist_ok=True)
                descriptor, path = tempfile.mkstemp(
                    prefix=prefix,
                    suffix=".json",
                    dir=self.options.profile_dir,
                )
                os.close(descriptor)
                assert graph is not None
                with torch.profiler.profile(
                    activities=[
                        torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA,
                    ],
                    record_shapes=True,
                    with_stack=True,
                ) as profiler:
                    for _ in range(self.options.profile_iters):
                        graph.replay()
                    torch.cuda.synchronize()
                profiler.export_chrome_trace(path)
            except Exception as error:
                try:
                    if path:
                        os.unlink(path)
                except OSError:
                    pass
                return BenchResult(
                    status="FAIL",
                    note=f"profile: {short_error(error)}\n{traceback.format_exc()}",
                    **base,
                )
            return BenchResult(
                status="PROFILED",
                tolerance_ratio=tolerance_ratio,
                atol=prepared.tolerance.atol,
                rtol=prepared.tolerance.rtol,
                note=path,
                **base,
            )

        mean_ms = statistics.mean(samples_ms)
        # Nearest-rank percentiles: with a small --repeat, both p95 and p99 land on
        # the last sample, i.e. they report max.
        sorted_samples = sorted(samples_ms)
        q_lengths = torch.full(
            (case.batch_size,), case.input_len, dtype=torch.int32, device="cuda"
        )
        kv_lengths = torch.full(
            (case.batch_size,), case.seq_len, dtype=torch.int32, device="cuda"
        )
        tflops = attention_tflops_per_sec_with_actual_seq_lens(
            q_lengths,
            kv_lengths,
            case.head_dim,
            case.head_dim,
            case.head_num,
            causal=True,
            ms=mean_ms,
        )
        return BenchResult(
            status="PASS",
            mean_ms=mean_ms,
            p50_ms=statistics.median(samples_ms),
            p95_ms=sorted_samples[math.ceil(len(sorted_samples) * 0.95) - 1],
            p99_ms=sorted_samples[math.ceil(len(sorted_samples) * 0.99) - 1],
            tflops=tflops,
            tolerance_ratio=tolerance_ratio,
            atol=prepared.tolerance.atol if tolerance_ratio is not None else None,
            rtol=prepared.tolerance.rtol if tolerance_ratio is not None else None,
            **base,
        )


class BenchReport:
    CSV_FIELDS = (
        "impl_name",
        "status",
        "q_dtype",
        "kv_dtype",
        "effective_kv_dtype",
        "batch_size",
        "seq_len",
        "prefix_len",
        "input_len",
        "reuse_cache_ratio",
        "head_num",
        "kv_head_num",
        "head_dim",
        "page_size",
        "mean_ms",
        "p50_ms",
        "p95_ms",
        "p99_ms",
        "tflops",
        "tolerance_ratio",
        "atol",
        "rtol",
        "note",
        "case_id",
        "seed",
    )

    @staticmethod
    def dumps(results: Sequence[BenchResult]) -> str:
        return json.dumps([result.to_dict() for result in results])

    @staticmethod
    def loads(value: str) -> list[BenchResult]:
        return [BenchResult.from_dict(item) for item in json.loads(value)]

    @classmethod
    def dump_json(cls, results: Sequence[BenchResult], path: str) -> str:
        output_path = os.path.abspath(os.path.expanduser(path))
        output_dir = os.path.dirname(output_path)
        os.makedirs(output_dir, exist_ok=True)
        descriptor, temp_path = tempfile.mkstemp(
            prefix=f".{os.path.basename(output_path)}.",
            suffix=".tmp",
            dir=output_dir,
        )
        try:
            stream = os.fdopen(descriptor, "w", encoding="utf-8")
            descriptor = -1
            with stream:
                stream.write(cls.dumps(results))
            os.replace(temp_path, output_path)
        except BaseException:
            if descriptor >= 0:
                os.close(descriptor)
            try:
                os.unlink(temp_path)
            except FileNotFoundError:
                pass
            raise
        return output_path

    @classmethod
    def dump_csv(cls, results: Sequence[BenchResult], path: str) -> str:
        output_path = os.path.abspath(os.path.expanduser(path))
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        ordered = sorted(
            results,
            key=lambda result: (
                result.case.q_dtype,
                result.case.kv_dtype,
                result.impl_name,
                result.case.seq_len,
                result.case.reuse_cache_ratio,
                result.status,
            ),
        )
        with open(output_path, "w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=cls.CSV_FIELDS)
            writer.writeheader()
            for result in ordered:
                case = result.case
                writer.writerow(
                    {
                        "impl_name": result.impl_name,
                        "status": result.status,
                        "q_dtype": case.q_dtype,
                        "kv_dtype": case.kv_dtype,
                        "effective_kv_dtype": result.effective_kv_dtype,
                        "batch_size": case.batch_size,
                        "seq_len": case.seq_len,
                        "prefix_len": case.prefix_len,
                        "input_len": case.input_len,
                        "reuse_cache_ratio": case.reuse_cache_ratio,
                        "head_num": case.head_num,
                        "kv_head_num": case.kv_head_num,
                        "head_dim": case.head_dim,
                        "page_size": case.page_size,
                        "mean_ms": result.mean_ms,
                        "p50_ms": result.p50_ms,
                        "p95_ms": result.p95_ms,
                        "p99_ms": result.p99_ms,
                        "tflops": result.tflops,
                        "tolerance_ratio": result.tolerance_ratio,
                        "atol": result.atol,
                        "rtol": result.rtol,
                        "note": result.note,
                        "case_id": case.case_id,
                        "seed": case.seed,
                    }
                )
        return output_path

    @staticmethod
    def fully_skipped_impls(results: Sequence[BenchResult]) -> list[str]:
        """Return implementations whose scheduled runs all ended in SKIP."""
        skipped: dict[str, int] = {}
        measured: set[str] = set()
        for result in results:
            if result.status == "SKIP":
                skipped[result.impl_name] = skipped.get(result.impl_name, 0) + 1
            else:
                measured.add(result.impl_name)
        return sorted(name for name in skipped if name not in measured)

    @staticmethod
    def print(results: Sequence[BenchResult]) -> None:
        passed = sorted(
            (result for result in results if result.status == "PASS"),
            key=lambda result: (
                result.case.q_dtype,
                result.effective_kv_dtype or "",
                result.impl_name,
                result.case.seq_len,
                result.case.reuse_cache_ratio,
            ),
        )
        impl_width = max((len(result.impl_name) for result in passed), default=20) + 2
        header = (
            f"{'impl':<{impl_width}}{'q':<6}{'kv':<6}{'seq':>8}{'prefix':>9}{'input':>9}"
            f"{'mean_ms':>11}{'p50_ms':>11}{'p95_ms':>11}{'p99_ms':>11}{'TFLOPs/s':>11}{'tol_ratio':>12}"
        )
        log("\n" + "=" * len(header))
        log(header)
        log("-" * len(header))
        if not passed:
            log("(no PASS results)")
        for result in passed:
            case = result.case
            tolerance = (
                "-"
                if result.tolerance_ratio is None
                else f"{result.tolerance_ratio:.3e}"
            )
            log(
                f"{result.impl_name:<{impl_width}}{case.q_dtype:<6}{(result.effective_kv_dtype or '-'):<6}"
                f"{case.seq_len:>8}{case.prefix_len:>9}{case.input_len:>9}"
                f"{result.mean_ms:>11.3f}{result.p50_ms:>11.3f}{result.p95_ms:>11.3f}"
                f"{result.p99_ms:>11.3f}{result.tflops:>11.2f}{tolerance:>12}"
            )
        log("=" * len(header))

        skipped = [result for result in results if result.status == "SKIP"]
        profiled = [result for result in results if result.status == "PROFILED"]
        if profiled:
            log(f"\nPROFILED: {len(profiled)} trace(s) exported")
            for result in profiled:
                log(f"  {result.impl_name}: {result.note}")
        if skipped:
            grouped_skips: dict[tuple[str, str], int] = {}
            for result in skipped:
                grouped_skips[(result.impl_name, result.note)] = (
                    grouped_skips.get((result.impl_name, result.note), 0) + 1
                )
            log(f"\nSKIP: {len(skipped)} run(s) not measured")
            for (impl_name, note), count in sorted(grouped_skips.items()):
                log(f"  {impl_name} x{count}: {note or 'unspecified'}")
        fully_skipped = BenchReport.fully_skipped_impls(results)
        if fully_skipped:
            log(
                "\nWARNING: these implementations were scheduled but never measured "
                f"(every run SKIPped): {', '.join(fully_skipped)}"
            )

        failures = [
            result for result in results if result.status in ("FAIL", "MISMATCH")
        ]
        if failures:
            grouped: dict[tuple[str, str, str], list[BenchResult]] = {}
            for result in failures:
                first_line = result.note.splitlines()[0] if result.note else ""
                grouped.setdefault(
                    (result.impl_name, result.status, first_line), []
                ).append(result)
            log(
                f"\nFAIL/MISMATCH: {len(grouped)} unique from {len(failures)} occurrences"
            )
            for (impl_name, status, first_line), group in grouped.items():
                cases = ", ".join(
                    f"seq={item.case.seq_len}/{item.case.mode_tag}/{item.case.kv_dtype}"
                    for item in group[:5]
                )
                log(f"  {impl_name} {status}: {first_line} [{cases}]")


def setup_exec_context() -> None:
    env_configs = PyEnvConfigs()
    env_configs.runtime_config.fifo_scheduler_config.max_context_batch_size = 64
    engine_config = EngineConfig.create(env_configs, nccl_comm_config=None)
    model_config = ModelConfig()
    parallelism = engine_config.parallelism_config
    init_exec_ctx(
        device_id=parallelism.world_rank % parallelism.local_world_size,
        trace_memory=engine_config.profiling_debug_logging_config.trace_memory,
        enable_comm_overlap=engine_config.device_resource_config.enable_comm_overlap,
        mla_ops_type=int(model_config.mla_ops_type),
    )


def build_parallelism_config() -> ParallelismConfig:
    config = ParallelismConfig()
    config.tp_size = 1
    config.tp_rank = 0
    config.world_size = 1
    config.world_rank = 0
    config.local_world_size = 1
    config.local_rank = 0
    return config

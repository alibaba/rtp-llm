#!/usr/bin/env python3
"""Reproducible MiniMax M3VL ViT request-concurrency benchmark.

The benchmark isolates the ViT service core path:

  decoded CPU RGB tensor -> scheduler -> H2D -> GPU resize/normalize/fold
  -> batched ViT -> image-token assembly -> CUDA completion

Network download, JPEG decode, RPC transport, cache hits, and LLM prefill are
excluded intentionally. Each request contains exactly one image, so request
concurrency also equals candidate image concurrency when batch caps are above
the tested concurrency.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import logging
import os
import platform
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
import torchvision

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


@dataclass(frozen=True)
class ImageCase:
    name: str
    width: int
    height: int


@dataclass
class BenchmarkResult:
    image: str
    raw_width: int
    raw_height: int
    target_width: int
    target_height: int
    input_patches: int
    output_tokens: int
    concurrency: int
    requests: int
    request_per_second: float
    image_per_second: float
    rt_mean_ms: float
    rt_p50_ms: float
    rt_p95_ms: float
    rt_p99_ms: float
    gpu_util_avg: float
    gpu_util_p95: float
    gpu_util_max: float
    peak_allocated_mib: float
    peak_allocated_delta_mib: float
    peak_nvml_used_mib: float
    peak_nvml_delta_mib: float
    average_batch_images: float
    maximum_batch_images: int
    cuda_graph_hits: int
    cuda_graph_misses: int
    cuda_graph_captures: int
    cuda_graph_fallbacks: int
    external_busy_samples: int


class WorkItem:
    """Minimal MMWorkItem interface consumed by MMScheduler."""

    def __init__(self, preprocess_result: Any, mm_type: Any):
        self.preprocess_result = preprocess_result
        self.mm_type = mm_type
        self.mm_inputs = [None]
        self.mm_timeout_ms = 120000
        self.need_check_cache = False
        self.cache_key = ""
        self.embedding_result = None


class GpuMonitor(threading.Thread):
    def __init__(
        self,
        physical_device: int,
        interval_seconds: float = 0.05,
        external_busy_threshold: float = 50.0,
        external_memory_threshold_mib: float = 4096.0,
    ):
        super().__init__(daemon=True)
        import pynvml

        pynvml.nvmlInit()
        self._nvml = pynvml
        self._physical_device = physical_device
        self._handle = pynvml.nvmlDeviceGetHandleByIndex(physical_device)
        self._other_handles = [
            pynvml.nvmlDeviceGetHandleByIndex(index)
            for index in range(pynvml.nvmlDeviceGetCount())
            if index != physical_device
        ]
        self._interval_seconds = interval_seconds
        self._external_busy_threshold = external_busy_threshold
        self._external_memory_threshold_mib = external_memory_threshold_mib
        self._stop_event = threading.Event()
        self.utilization: List[float] = []
        self.memory_used_mib: List[float] = []
        self.external_busy_samples = 0

    def run(self) -> None:
        while not self._stop_event.is_set():
            try:
                utilization = self._nvml.nvmlDeviceGetUtilizationRates(self._handle).gpu
                memory = self._nvml.nvmlDeviceGetMemoryInfo(self._handle).used
                self.utilization.append(float(utilization))
                self.memory_used_mib.append(float(memory) / 1024.0 / 1024.0)
                if any(
                    self._nvml.nvmlDeviceGetUtilizationRates(handle).gpu
                    > self._external_busy_threshold
                    or self._nvml.nvmlDeviceGetMemoryInfo(handle).used / 1024.0 / 1024.0
                    > self._external_memory_threshold_mib
                    for handle in self._other_handles
                ):
                    self.external_busy_samples += 1
            except Exception:
                pass
            self._stop_event.wait(self._interval_seconds)

    def stop(self) -> None:
        self._stop_event.set()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default="/data2/xieshui.yyx/MiniMax-M3-MXFP8",
    )
    parser.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "docs/assets/multimodal_vit_baseline"),
    )
    parser.add_argument(
        "--concurrencies",
        default="1,2,4,8,16,32,64",
    )
    parser.add_argument(
        "--image-cases",
        default="small_448,1080p,2k_1440p",
        help="Comma-separated subset of: small_448,1080p,2k_1440p",
    )
    parser.add_argument("--requests-per-point", type=int, default=128)
    parser.add_argument("--min-waves", type=int, default=4)
    parser.add_argument("--minimum-point-seconds", type=float, default=10.0)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup-runs", type=int, default=3)
    parser.add_argument("--batch-wait-ms", type=int, default=10)
    parser.add_argument("--max-batch-size", type=int, default=64)
    parser.add_argument("--max-batch-images", type=int, default=64)
    parser.add_argument("--monitor-interval-ms", type=int, default=50)
    parser.add_argument("--busy-threshold", type=float, default=50.0)
    parser.add_argument("--idle-memory-threshold-mib", type=float, default=4096.0)
    parser.add_argument("--idle-seconds", type=float, default=5.0)
    parser.add_argument("--idle-poll-seconds", type=float, default=1.0)
    parser.add_argument("--idle-timeout-seconds", type=float, default=1800.0)
    parser.add_argument("--max-repeat-attempts", type=int, default=20)
    parser.add_argument(
        "--load-checkpoint-weights",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--validate-mixed-batch",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--enable-cuda-graph",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--allow-busy-gpu", action="store_true")
    return parser.parse_args()


def physical_device_index() -> int:
    visible_index = torch.cuda.current_device()
    cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not cuda_visible_devices:
        return visible_index
    return int(cuda_visible_devices.split(",")[visible_index])


def git_output(*args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args], cwd=REPO_ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"


def load_visual_checkpoint(mm: Any, checkpoint: str) -> None:
    from safetensors import safe_open

    from rtp_llm.multimodal.multimodal_mixins.minimax_m3_vl.minimax_m3_vl_vit import (
        get_fused_qkv_checkpoint_names,
    )

    index_path = Path(checkpoint) / "model.safetensors.index.json"
    with index_path.open() as handle:
        weight_map = json.load(handle)["weight_map"]

    state = mm.visual.state_dict()
    keys_by_shard: Dict[str, List[Tuple[str, str, int, int]]] = {}
    missing = []
    for live_key, target in state.items():
        checkpoint_keys = get_fused_qkv_checkpoint_names(live_key)
        if checkpoint_keys is None:
            checkpoint_keys = (live_key,)
        component_size = target.shape[0] // len(checkpoint_keys)
        for component_index, checkpoint_key in enumerate(checkpoint_keys):
            shard = weight_map.get(checkpoint_key)
            if shard is None:
                missing.append(checkpoint_key)
                continue
            keys_by_shard.setdefault(shard, []).append(
                (
                    live_key,
                    checkpoint_key,
                    component_index,
                    component_size,
                )
            )

    if missing:
        raise RuntimeError(
            f"{len(missing)} vision weight(s) missing from checkpoint, "
            f"first={missing[0]}"
        )

    with torch.no_grad():
        for shard, entries in keys_by_shard.items():
            shard_path = str(Path(checkpoint) / shard)
            with safe_open(shard_path, framework="pt", device="cpu") as tensors:
                for live_key, checkpoint_key, index, component_size in entries:
                    target = state[live_key]
                    source = tensors.get_tensor(checkpoint_key)
                    if target.shape[0] == component_size:
                        target.copy_(source)
                    else:
                        start = index * component_size
                        target[start : start + component_size].copy_(source)


def build_model(checkpoint: str, load_weights: bool) -> Any:
    # Keep the native extension/model import ahead of scheduler/metrics imports.
    from rtp_llm.multimodal.multimodal_mixins.minimax_m3_vl.minimax_m3_vl_mixin import (
        MiniMaxM3VLImageEmbedding,
    )
    from rtp_llm.utils.base_model_datatypes import VitParameters

    params = VitParameters()
    params.config = {"ckpt_path": checkpoint}
    mm = MiniMaxM3VLImageEmbedding(params)
    if load_weights:
        load_visual_checkpoint(mm, checkpoint)
    mm.visual = mm.visual.cuda().to(torch.bfloat16)
    MiniMaxM3VLImageEmbedding._data_type = property(lambda self: torch.bfloat16)
    return mm


def make_image_data(mm: Any, case: ImageCase) -> Tuple[Any, Dict[str, int]]:
    from rtp_llm.multimodal.multimodal_mixins.minimax_m3_vl.image_processor import (
        smart_resize,
    )

    raw = torch.zeros((3, case.height, case.width), dtype=torch.uint8)
    processor = mm.mm_processor
    factor = processor.patch_size * processor.merge_size
    target_height, target_width = smart_resize(
        case.height,
        case.width,
        factor=factor,
        min_pixels=processor.min_pixels,
        max_pixels=processor.max_pixels,
    )
    grid_height = target_height // processor.patch_size
    grid_width = target_width // processor.patch_size
    input_patches = grid_height * grid_width
    output_tokens = input_patches // (processor.merge_size**2) + 2
    return (
        raw,
        (target_height, target_width),
        None,
    ), {
        "target_height": target_height,
        "target_width": target_width,
        "input_patches": input_patches,
        "output_tokens": output_tokens,
    }


def wait_for_gpus_idle(
    physical_device: int,
    allow_busy: bool,
    busy_threshold: float,
    memory_threshold_mib: float,
    require_target_memory_idle: bool,
    idle_seconds: float,
    poll_seconds: float,
    timeout_seconds: float,
) -> None:
    import pynvml

    pynvml.nvmlInit()
    handles = [
        pynvml.nvmlDeviceGetHandleByIndex(index)
        for index in range(pynvml.nvmlDeviceGetCount())
    ]
    if allow_busy:
        return

    started = time.monotonic()
    idle_started = None
    last_log = 0.0
    while True:
        now = time.monotonic()
        utilization = [
            float(pynvml.nvmlDeviceGetUtilizationRates(handle).gpu)
            for handle in handles
        ]
        memory_used_mib = [
            float(pynvml.nvmlDeviceGetMemoryInfo(handle).used) / 1024.0 / 1024.0
            for handle in handles
        ]
        memory_idle = all(
            memory <= memory_threshold_mib
            for index, memory in enumerate(memory_used_mib)
            if require_target_memory_idle or index != physical_device
        )
        if max(utilization, default=0.0) <= busy_threshold and memory_idle:
            if idle_started is None:
                idle_started = now
            if now - idle_started >= idle_seconds:
                return
        else:
            idle_started = None

        if now - started > timeout_seconds:
            raise TimeoutError(
                f"GPUs did not stay below {busy_threshold}% for "
                f"{idle_seconds}s within {timeout_seconds}s; "
                f"last_utilization={utilization}, "
                f"last_memory_mib={memory_used_mib}"
            )
        if now - last_log >= 10.0:
            logging.error(
                "waiting for idle GPUs before benchmark: "
                "utilization=%s memory_mib=%s",
                utilization,
                [round(memory) for memory in memory_used_mib],
            )
            last_log = now
        time.sleep(poll_seconds)


def run_point(
    mm: Any,
    data: Any,
    image_case: ImageCase,
    image_info: Dict[str, int],
    mm_type: Any,
    concurrency: int,
    args: argparse.Namespace,
    physical_device: int,
) -> BenchmarkResult:
    from rtp_llm.multimodal.mm_scheduler import MMScheduler

    scheduler = MMScheduler(
        mm,
        batch_wait_ms=args.batch_wait_ms,
        max_batch_size=args.max_batch_size,
        max_batch_images=args.max_batch_images,
    )

    for _ in range(args.warmup_runs):
        warmup_outputs = mm.batched_embedding(
            [data] * concurrency, [mm_type] * concurrency
        )
        torch.cuda.synchronize()
        del warmup_outputs

    graph_stats_before = mm.vision_graph_stats()

    batch_sizes: List[int] = []
    batch_lock = threading.Lock()
    original_batched_embedding = mm.batched_embedding

    def recorded_batched_embedding(
        data_list: List[Any], mm_types: List[Any], **kwargs: Any
    ) -> Any:
        with batch_lock:
            batch_sizes.append(len(data_list))
        return original_batched_embedding(data_list, mm_types, **kwargs)

    mm.batched_embedding = recorded_batched_embedding
    requested = max(args.requests_per_point, concurrency * args.min_waves)
    iterations = (requested + concurrency - 1) // concurrency

    latencies: List[float] = []
    errors: List[BaseException] = []
    result_lock = threading.Lock()
    barrier = threading.Barrier(concurrency + 1)
    start_event = threading.Event()
    deadline = [0.0]

    def worker() -> None:
        local_latencies = []
        try:
            barrier.wait()
            start_event.wait()
            iteration = 0
            while iteration < iterations or time.perf_counter() < deadline[0]:
                work_item = WorkItem(data, mm_type)
                start = time.perf_counter()
                scheduler.submit_and_wait([work_item])
                # Scheduler completion means kernels were submitted. Synchronize
                # to measure when the embedding is actually ready for transport.
                torch.cuda.synchronize()
                local_latencies.append((time.perf_counter() - start) * 1000.0)
                iteration += 1
        except BaseException as error:
            with result_lock:
                errors.append(error)
        finally:
            with result_lock:
                latencies.extend(local_latencies)

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    baseline_allocated = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()

    import pynvml

    pynvml.nvmlInit()
    nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(physical_device)
    baseline_nvml = pynvml.nvmlDeviceGetMemoryInfo(nvml_handle).used / 1024.0 / 1024.0
    monitor = GpuMonitor(
        physical_device,
        args.monitor_interval_ms / 1000.0,
        args.busy_threshold,
        args.idle_memory_threshold_mib,
    )
    monitor.start()
    threads = [threading.Thread(target=worker) for _ in range(concurrency)]
    for thread in threads:
        thread.start()
    barrier.wait()
    start = time.perf_counter()
    deadline[0] = start + args.minimum_point_seconds
    start_event.set()
    for thread in threads:
        thread.join()
    torch.cuda.synchronize()
    wall_seconds = time.perf_counter() - start
    monitor.stop()
    monitor.join(timeout=2.0)

    mm.batched_embedding = original_batched_embedding
    scheduler.close()

    if errors:
        raise errors[0]
    if not latencies:
        raise RuntimeError("benchmark produced no latency samples")

    peak_nvml = max(monitor.memory_used_mib) if monitor.memory_used_mib else 0.0
    peak_allocated = torch.cuda.max_memory_allocated()
    request_rate = len(latencies) / wall_seconds
    utilization = monitor.utilization or [0.0]
    graph_stats_after = mm.vision_graph_stats()
    graph_stats = {
        key: graph_stats_after[key] - graph_stats_before[key]
        for key in graph_stats_after
    }

    return BenchmarkResult(
        image=image_case.name,
        raw_width=image_case.width,
        raw_height=image_case.height,
        target_width=image_info["target_width"],
        target_height=image_info["target_height"],
        input_patches=image_info["input_patches"],
        output_tokens=image_info["output_tokens"],
        concurrency=concurrency,
        requests=len(latencies),
        request_per_second=request_rate,
        image_per_second=request_rate,
        rt_mean_ms=float(np.mean(latencies)),
        rt_p50_ms=float(np.percentile(latencies, 50)),
        rt_p95_ms=float(np.percentile(latencies, 95)),
        rt_p99_ms=float(np.percentile(latencies, 99)),
        gpu_util_avg=float(np.mean(utilization)),
        gpu_util_p95=float(np.percentile(utilization, 95)),
        gpu_util_max=float(max(utilization)),
        peak_allocated_mib=peak_allocated / 1024.0 / 1024.0,
        peak_allocated_delta_mib=(peak_allocated - baseline_allocated)
        / 1024.0
        / 1024.0,
        peak_nvml_used_mib=peak_nvml,
        peak_nvml_delta_mib=max(0.0, peak_nvml - baseline_nvml),
        average_batch_images=float(np.mean(batch_sizes)),
        maximum_batch_images=max(batch_sizes),
        cuda_graph_hits=graph_stats["hit"],
        cuda_graph_misses=graph_stats["miss"],
        cuda_graph_captures=graph_stats["capture"],
        cuda_graph_fallbacks=graph_stats["fallback"],
        external_busy_samples=monitor.external_busy_samples,
    )


def write_csv(path: Path, results: Sequence[BenchmarkResult]) -> None:
    rows = [asdict(result) for result in results]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_json(
    path: Path,
    metadata: Dict[str, Any],
    results: Sequence[BenchmarkResult],
    repetitions: Sequence[BenchmarkResult],
    discarded: Sequence[BenchmarkResult],
) -> None:
    payload = {
        "metadata": metadata,
        "results": [asdict(result) for result in results],
        "repetitions": [asdict(result) for result in repetitions],
        "discarded_repetitions": [asdict(result) for result in discarded],
    }
    with path.open("w") as handle:
        json.dump(payload, handle, indent=2)


def write_plot(path: Path, results: Sequence[BenchmarkResult]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 2, figsize=(13, 9))
    colors = {
        "small_448": "#2374ab",
        "1080p": "#d1495b",
        "2k_1440p": "#2a9d5b",
    }

    grouped: Dict[str, List[BenchmarkResult]] = {}
    for result in results:
        grouped.setdefault(result.image, []).append(result)

    for image, rows in grouped.items():
        rows = sorted(rows, key=lambda row: row.concurrency)
        concurrency = [row.concurrency for row in rows]
        color = colors[image]
        axes[0, 0].plot(
            concurrency,
            [row.request_per_second for row in rows],
            marker="o",
            color=color,
            label=image,
        )
        axes[0, 1].plot(
            concurrency,
            [row.rt_p50_ms for row in rows],
            marker="o",
            color=color,
            label=f"{image} P50",
        )
        axes[0, 1].plot(
            concurrency,
            [row.rt_p99_ms for row in rows],
            marker="x",
            linestyle="--",
            color=color,
            label=f"{image} P99",
        )
        axes[1, 0].plot(
            concurrency,
            [row.gpu_util_avg for row in rows],
            marker="o",
            color=color,
            label=image,
        )
        axes[1, 1].plot(
            concurrency,
            [row.peak_allocated_delta_mib for row in rows],
            marker="o",
            color=color,
            label=image,
        )

    titles = [
        "Throughput",
        "Request latency",
        "Average GPU utilization",
        "Peak allocated memory above model baseline",
    ]
    ylabels = ["requests / second", "milliseconds", "percent", "MiB"]
    for axis, title, ylabel in zip(axes.flat, titles, ylabels):
        axis.set_title(title)
        axis.set_xlabel("request concurrency")
        axis.set_ylabel(ylabel)
        axis.set_xscale("log", base=2)
        axis.set_xticks([1, 2, 4, 8, 16, 32, 64])
        axis.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        axis.grid(True, alpha=0.25)
        axis.legend(fontsize=8)

    figure.suptitle("MiniMax M3VL ViT request-concurrency baseline")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    logging.getLogger().setLevel(logging.ERROR)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    all_image_cases = [
        ImageCase("small_448", 448, 448),
        ImageCase("1080p", 1920, 1080),
        ImageCase("2k_1440p", 2560, 1440),
    ]
    selected_case_names = {
        value.strip() for value in args.image_cases.split(",") if value.strip()
    }
    known_case_names = {case.name for case in all_image_cases}
    unknown_case_names = selected_case_names - known_case_names
    if unknown_case_names:
        raise ValueError(f"unknown --image-cases: {sorted(unknown_case_names)}")
    image_cases = [case for case in all_image_cases if case.name in selected_case_names]
    if not image_cases:
        raise ValueError("--image-cases must select at least one case")
    concurrencies = [
        int(value) for value in args.concurrencies.split(",") if value.strip()
    ]
    if max(concurrencies) > args.max_batch_size:
        raise ValueError("max concurrency exceeds --max-batch-size")
    if max(concurrencies) > args.max_batch_images:
        raise ValueError("max concurrency exceeds --max-batch-images")

    physical_device = physical_device_index()
    wait_for_gpus_idle(
        physical_device,
        args.allow_busy_gpu,
        args.busy_threshold,
        args.idle_memory_threshold_mib,
        True,
        args.idle_seconds,
        args.idle_poll_seconds,
        args.idle_timeout_seconds,
    )

    print(f"loading M3VL from {args.checkpoint}", flush=True)
    mm = build_model(args.checkpoint, args.load_checkpoint_weights)
    cuda_graph_enabled = args.enable_cuda_graph
    mm.set_vision_cuda_graph_enabled(cuda_graph_enabled)

    # Import only after the model/native extension has initialized.
    from rtp_llm.metrics import kmonitor
    from rtp_llm.utils.base_model_datatypes import MMUrlType

    kmonitor.report = lambda *unused_args, **unused_kwargs: None

    metadata = {
        "date": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "branch": git_output("branch", "--show-current"),
        "commit": git_output("rev-parse", "HEAD"),
        "worktree_dirty": bool(git_output("status", "--porcelain")),
        "hostname": platform.node(),
        "gpu": torch.cuda.get_device_name(),
        "physical_device": physical_device,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "checkpoint": args.checkpoint,
        "weights_loaded": args.load_checkpoint_weights,
        "scope": (
            "post-decode CPU RGB -> scheduler -> H2D -> GPU transform -> "
            "ViT -> assembly -> CUDA synchronize"
        ),
        "excluded": ["download", "image decode", "RPC", "cache", "LLM prefill"],
        "one_image_per_request": True,
        "concurrencies": concurrencies,
        "requests_per_point": args.requests_per_point,
        "min_waves": args.min_waves,
        "minimum_point_seconds": args.minimum_point_seconds,
        "repeats": args.repeats,
        "batch_wait_ms": args.batch_wait_ms,
        "max_batch_size": args.max_batch_size,
        "max_batch_images": args.max_batch_images,
        "busy_threshold": args.busy_threshold,
        "idle_memory_threshold_mib": args.idle_memory_threshold_mib,
        "idle_seconds": args.idle_seconds,
        "max_repeat_attempts": args.max_repeat_attempts,
        "allow_busy_gpu": args.allow_busy_gpu,
        "validate_mixed_batch": args.validate_mixed_batch,
        "cuda_graph_enabled": cuda_graph_enabled,
        "image_cases": [asdict(case) for case in image_cases],
    }

    results = []
    repetitions = []
    discarded = []
    for image_case in image_cases:
        data, image_info = make_image_data(mm, image_case)
        print(
            f"{image_case.name}: raw={image_case.width}x{image_case.height} "
            f"target={image_info['target_width']}x"
            f"{image_info['target_height']} "
            f"patches={image_info['input_patches']} "
            f"tokens={image_info['output_tokens']}",
            flush=True,
        )
        for concurrency in concurrencies:
            point_results = []
            for repeat in range(args.repeats):
                result = None
                for attempt in range(1, args.max_repeat_attempts + 1):
                    wait_for_gpus_idle(
                        physical_device,
                        args.allow_busy_gpu,
                        args.busy_threshold,
                        args.idle_memory_threshold_mib,
                        False,
                        args.idle_seconds,
                        args.idle_poll_seconds,
                        args.idle_timeout_seconds,
                    )
                    candidate = run_point(
                        mm,
                        data,
                        image_case,
                        image_info,
                        MMUrlType.IMAGE,
                        concurrency,
                        args,
                        physical_device,
                    )
                    if candidate.external_busy_samples == 0 or args.allow_busy_gpu:
                        result = candidate
                        break
                    discarded.append(candidate)
                    print(
                        f"  c={concurrency:>2} repeat={repeat + 1} "
                        f"attempt={attempt} discarded: "
                        f"external_busy_samples="
                        f"{candidate.external_busy_samples}",
                        flush=True,
                    )
                if result is None:
                    raise RuntimeError(
                        f"failed to collect a clean repetition for "
                        f"{image_case.name} concurrency={concurrency} "
                        f"after {args.max_repeat_attempts} attempts"
                    )
                point_results.append(result)
                repetitions.append(result)
                print(
                    f"  c={concurrency:>2} repeat={repeat + 1} "
                    f"rps={result.request_per_second:>7.2f} "
                    f"p50={result.rt_p50_ms:>7.2f}ms "
                    f"p99={result.rt_p99_ms:>7.2f}ms "
                    f"gpu={result.gpu_util_avg:>5.1f}% "
                    f"mem_delta={result.peak_allocated_delta_mib:>7.0f}MiB "
                    f"batch={result.average_batch_images:.1f}/"
                    f"{result.maximum_batch_images} "
                    f"graph={result.cuda_graph_hits}/"
                    f"{result.cuda_graph_misses}/"
                    f"{result.cuda_graph_captures}/"
                    f"{result.cuda_graph_fallbacks}",
                    flush=True,
                )
            # Preserve a coherent observed run rather than mixing percentiles
            # from different repetitions. The middle-throughput run is the
            # baseline; all repetitions remain in JSON for variance analysis.
            result = sorted(point_results, key=lambda row: row.request_per_second)[
                len(point_results) // 2
            ]
            results.append(result)
            print(
                f"  c={concurrency:>2} selected_median_rps="
                f"{result.request_per_second:.2f}",
                flush=True,
            )

    vision_model = mm.visual.vision_tower.vision_model
    metadata["attention_backends"] = sorted(
        {layer.self_attn.last_backend for layer in vision_model.encoder.layers}
    )
    metadata["attention_backend_errors"] = sorted(
        {
            layer.self_attn.last_backend_error
            for layer in vision_model.encoder.layers
            if layer.self_attn.last_backend_error
        }
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    short_commit = metadata["commit"][:9]
    csv_path = output_dir / f"m3_vit_baseline_{short_commit}.csv"
    json_path = output_dir / f"m3_vit_baseline_{short_commit}.json"
    plot_path = output_dir / f"m3_vit_baseline_{short_commit}.png"
    write_csv(csv_path, results)
    write_json(json_path, metadata, results, repetitions, discarded)
    write_plot(plot_path, results)

    external_busy = sum(result.external_busy_samples for result in repetitions)
    print(f"csv={csv_path}", flush=True)
    print(f"json={json_path}", flush=True)
    print(f"plot={plot_path}", flush=True)
    print(f"discarded_repetitions={len(discarded)}", flush=True)
    print(f"external_busy_samples={external_busy}", flush=True)
    if external_busy and not args.allow_busy_gpu:
        raise RuntimeError(
            "non-target GPUs became busy during the run; discard this baseline"
        )


if __name__ == "__main__":
    main()

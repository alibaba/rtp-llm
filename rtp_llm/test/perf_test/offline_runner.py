"""Offline throughput benchmark runner.

Closed-loop dispatch: maintains concurrency_limit requests in flight for a
configurable duration, then drains all in-flight requests.

Statistics cover ALL completed requests (no warmup/cooldown exclusion).
TPS = total_success_tokens / wall_time (first submit → last completion).
"""

import asyncio
import concurrent.futures
import json
import logging
import os
import random
import threading
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, NamedTuple, Optional, Set, TextIO, Tuple

import aiohttp
import requests

BASE_TEXTS = [
    "The quick brown fox jumps over the lazy dog near the river bank. ",
    "In a galaxy far far away there existed a civilization of extraordinary beings. ",
    "Once upon a time in a land of endless mountains and deep valleys lived a wise old sage. ",
    "The fundamental principles of quantum mechanics govern the behavior of particles at atomic scales. ",
    "Throughout history great empires have risen and fallen leaving behind monuments and legends. ",
    "Modern artificial intelligence systems leverage vast amounts of data to learn complex patterns. ",
    "The ocean depths remain one of the least explored frontiers on our planet today. ",
    "Classical music compositions from the baroque era continue to influence contemporary artists worldwide. ",
    "Advances in biotechnology have enabled unprecedented precision in genetic engineering applications. ",
    "The philosophy of consciousness explores the nature of subjective experience and awareness. ",
]


@dataclass
class OfflineBenchConfig:
    input_len_min: int = 512
    input_len_max: int = 2048
    output_len_min: int = 64
    output_len_max: int = 512
    prefix_groups: int = 1
    prefix_len: int = 0
    num_return_sequences: int = 1

    duration_s: int = 300
    drain_timeout_s: int = 0  # 0 = wait forever
    concurrency_limit: int = 0
    total_requests: int = (
        -1
    )  # -1 = auto (2*concurrency_limit), 0 = duration mode, >0 = fixed
    seed: int = 42
    dump_workload: str = ""

    def validate(self):
        if self.input_len_min <= 0:
            raise ValueError("input_len_min must be > 0")
        if self.input_len_min > self.input_len_max:
            raise ValueError(
                f"input_len_min ({self.input_len_min}) must be <= input_len_max ({self.input_len_max})"
            )
        if self.output_len_min <= 0:
            raise ValueError("output_len_min must be > 0")
        if self.output_len_min > self.output_len_max:
            raise ValueError(
                f"output_len_min ({self.output_len_min}) must be <= output_len_max ({self.output_len_max})"
            )
        if self.prefix_groups < 1:
            raise ValueError("prefix_groups must be >= 1")
        if self.prefix_len < 0:
            raise ValueError("prefix_len must be >= 0")
        if self.prefix_len > self.input_len_min:
            raise ValueError(
                f"prefix_len ({self.prefix_len}) must be <= input_len_min ({self.input_len_min})"
            )
        if self.num_return_sequences < 1:
            raise ValueError("num_return_sequences must be >= 1")
        if self.total_requests <= 0 and self.duration_s <= 0:
            raise ValueError("duration_s must be > 0 in duration mode")
        if self.concurrency_limit <= 0:
            raise ValueError("concurrency_limit must be > 0")


@dataclass
class OfflineMetrics:
    total_wall_time_s: float = 0.0
    output_tps: float = 0.0
    input_tps: float = 0.0
    total_tps: float = 0.0

    total_submitted: int = 0
    success_requests: int = 0
    fail_requests: int = 0
    cancelled_requests: int = 0
    avg_request_latency_s: float = 0.0
    p50_request_latency_s: float = 0.0
    p99_request_latency_s: float = 0.0
    avg_wait_time_ms: float = 0.0
    max_wait_time_ms: float = 0.0

    avg_ttft_ms: float = 0.0
    p50_ttft_ms: float = 0.0
    p99_ttft_ms: float = 0.0
    avg_tpot_ms: float = 0.0
    p50_tpot_ms: float = 0.0
    p99_tpot_ms: float = 0.0

    avg_input_len: float = 0.0
    avg_output_len: float = 0.0
    input_len_range: Tuple[int, int] = (0, 0)
    output_len_range: Tuple[int, int] = (0, 0)

    avg_reuse_len: float = 0.0
    avg_reuse_ratio: float = 0.0
    total_reuse_tokens: int = 0
    total_compute_tokens: int = 0

    kv_cache_total_tokens: int = 0
    kv_cache_total_blocks: int = 0
    kv_cache_block_size: int = 0
    kv_cache_avg_utilization: float = 0.0
    kv_cache_max_utilization: float = 0.0
    kv_cache_min_utilization: float = 0.0
    kv_cache_max_dp_utilization: float = 0.0
    concurrency_avg_utilization: float = 0.0
    concurrency_max_utilization: float = 0.0

    def raise_if_all_requests_failed(self):
        if self.total_submitted == 0:
            raise RuntimeError("Offline benchmark failed: no requests were submitted")
        if self.success_requests == 0:
            raise RuntimeError(
                "Offline benchmark failed: no requests succeeded "
                f"({self.total_submitted} submitted, {self.fail_requests} failed, "
                f"{self.cancelled_requests} cancelled)"
            )

    def print_table(self):
        lines = [
            "================== Offline Throughput Benchmark ==================",
            f"Submitted:             {self.total_submitted}",
            f"Success / Fail / Cancelled: "
            f"{self.success_requests} / {self.fail_requests} / {self.cancelled_requests}",
            f"Wall Time:             {self.total_wall_time_s:.1f} s",
            "--- Throughput ---",
            f"Output TPS:            {self.output_tps:,.1f} tokens/s",
            f"Input TPS:             {self.input_tps:,.1f} tokens/s",
            f"Total TPS:             {self.total_tps:,.1f} tokens/s",
            "--- Request Latency ---",
            f"Avg / P50 / P99:       {self.avg_request_latency_s:.2f} / "
            f"{self.p50_request_latency_s:.2f} / {self.p99_request_latency_s:.2f} s",
            f"Avg Wait Time:         {self.avg_wait_time_ms:.1f} ms "
            f"(max: {self.max_wait_time_ms:.1f} ms)",
            f"TTFT (avg/p50/p99):    {self.avg_ttft_ms:.1f} / "
            f"{self.p50_ttft_ms:.1f} / {self.p99_ttft_ms:.1f} ms",
            f"TPOT (avg/p50/p99):    {self.avg_tpot_ms:.2f} / "
            f"{self.p50_tpot_ms:.2f} / {self.p99_tpot_ms:.2f} ms",
            "--- Input / Output Distribution ---",
            f"Input  Len:            avg={self.avg_input_len:.0f}  "
            f"range=[{self.input_len_range[0]}, {self.input_len_range[1]}]",
            f"Output Len:            avg={self.avg_output_len:.0f}  "
            f"range=[{self.output_len_range[0]}, {self.output_len_range[1]}]",
            "--- Prefix Cache ---",
            f"Avg Reuse Len:         {self.avg_reuse_len:.0f} tokens "
            f"({self.avg_reuse_ratio * 100:.1f}% of input)",
            f"Total Reuse:           {self.total_reuse_tokens:,} tokens",
            f"Actual Prefill:        {self.total_compute_tokens:,} tokens",
            "--- KV Cache ---",
            f"Total Tokens:          {self.kv_cache_total_tokens:,}",
            f"Total Blocks:          {self.kv_cache_total_blocks:,} "
            f"(block_size={self.kv_cache_block_size})",
        ]
        if self.kv_cache_max_utilization > 0:
            lines += [
                "--- Runtime Utilization ---",
                f"KV Cache:              avg={self.kv_cache_avg_utilization * 100:.1f}%  "
                f"max={self.kv_cache_max_utilization * 100:.1f}%  "
                f"min={self.kv_cache_min_utilization * 100:.1f}%",
                f"Max DP KV Cache:       {self.kv_cache_max_dp_utilization * 100:.1f}%",
                f"Concurrency:           avg={self.concurrency_avg_utilization * 100:.1f}%  "
                f"max={self.concurrency_max_utilization * 100:.1f}%",
            ]
        lines += [
            "==================================================================",
        ]
        print("\n".join(lines))

    def save_json(self, result_dir: str):
        os.makedirs(result_dir, exist_ok=True)
        path = os.path.join(result_dir, "offline_benchmark_result.json")
        data = asdict(self)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        logging.info(f"Saved offline benchmark result to {path}")


# ---------------------------------------------------------------------------
# WorkloadGenerator
# ---------------------------------------------------------------------------


@dataclass
class _PrefixGroup:
    group_id: int
    prefix_text: str
    prefix_token_ids: Tuple[int, ...]
    rng: random.Random


@dataclass(frozen=True)
class _WorkloadSpec:
    request_id: int
    group: _PrefixGroup
    input_len: int
    output_len: int


@dataclass(frozen=True)
class _PreparedPrompt:
    request_id: Optional[int]
    prompt: str
    output_len: int
    dump_record: Optional[str] = None


class _PromptGenerationExpired(Exception):
    """Prompt generation was cancelled or crossed its dispatch deadline."""


class WorkloadGenerator:
    """On-demand request generator with independent per-prefix-group RNGs.

    Prompt generation starts from token-id slices for speed, then validates the
    final text with the real tokenizer. Decode/concatenation is not assumed to
    preserve token boundaries: candidates are repaired until they have the
    requested exact length and the group's canonical token prefix.
    """

    _MAX_TOKEN_REPAIR_ATTEMPTS = 64

    def __init__(
        self,
        config: OfflineBenchConfig,
        tokenizer: Any,
        seed: int = 42,
        dump_file: Optional[TextIO] = None,
    ):
        self._config = config
        self._tokenizer = tokenizer
        self._seed = seed
        self._dump_file = dump_file
        self._counter = 0
        self._state_lock = threading.Lock()
        self._dump_lock = threading.Lock()
        self._pending_dump_records: Dict[int, str] = {}
        self._next_dump_id = 0

        # Pre-compute a token buffer large enough for all configured prompt and prefix slices.
        big_text = " ".join(BASE_TEXTS) * 50
        base_tokens = self._tokenizer.encode(big_text)
        if not base_tokens:
            raise ValueError("tokenizer produced an empty token buffer")
        min_buffer_tokens = max(
            config.input_len_max,
            config.prefix_groups * config.prefix_len,
            1,
        )
        repeats = (min_buffer_tokens + len(base_tokens) - 1) // len(base_tokens)
        self._big_tokens = base_tokens * repeats
        self._big_text = big_text

        self._groups: List[_PrefixGroup] = []
        for g in range(config.prefix_groups):
            prefix_text = ""
            prefix_token_ids: Tuple[int, ...] = ()
            if config.prefix_len > 0:
                prefix_text, encoded_prefix = self._find_exact_text(
                    offset=g * config.prefix_len,
                    target_len=config.prefix_len,
                )
                prefix_token_ids = tuple(encoded_prefix)
                if any(
                    prefix_token_ids == group.prefix_token_ids for group in self._groups
                ):
                    # Very small/fake tokenizers can map multiple source slices to
                    # the same text. Search farther in the reservoir so configured
                    # groups still represent distinct cache-prefix populations.
                    for shift in range(1, self._MAX_TOKEN_REPAIR_ATTEMPTS + 1):
                        candidate_text, candidate_ids = self._find_exact_text(
                            offset=(g + shift) * config.prefix_len,
                            target_len=config.prefix_len,
                        )
                        candidate_tuple = tuple(candidate_ids)
                        if all(
                            candidate_tuple != group.prefix_token_ids
                            for group in self._groups
                        ):
                            prefix_text = candidate_text
                            prefix_token_ids = candidate_tuple
                            break
            self._groups.append(
                _PrefixGroup(
                    group_id=g,
                    prefix_text=prefix_text,
                    prefix_token_ids=prefix_token_ids,
                    rng=random.Random(seed + g),
                )
            )

    def _token_slice(self, offset: int, length: int) -> List[int]:
        if length <= 0:
            return []
        buf = self._big_tokens
        start = offset % len(buf)
        end = start + length
        if end <= len(buf):
            return buf[start:end]

        result = buf[start:]
        remaining = length - len(result)
        full_repeats, tail = divmod(remaining, len(buf))
        if full_repeats:
            result.extend(buf * full_repeats)
        if tail:
            result.extend(buf[:tail])
        return result

    def _find_exact_text(
        self,
        offset: int,
        target_len: int,
        prefix_text: str = "",
        prefix_token_ids: Tuple[int, ...] = (),
        cancel_event: Optional[threading.Event] = None,
        deadline: Optional[float] = None,
    ) -> Tuple[str, List[int]]:
        """Decode a reservoir slice and repair it to an exact token length.

        The tokenizer is authoritative. Each candidate is encoded after the
        final string concatenation, so boundary merges/splits are observed.
        ``prefix_token_ids`` makes the first ``prefix_len`` tokens an invariant
        rather than an assumption about string concatenation.
        """
        self._check_generation_active(cancel_event, deadline)
        if target_len < len(prefix_token_ids):
            raise ValueError(
                f"target_len ({target_len}) is shorter than required prefix "
                f"({len(prefix_token_ids)})"
            )
        if target_len == len(prefix_token_ids):
            self._check_generation_active(cancel_event, deadline)
            encoded = list(self._tokenizer.encode(prefix_text))
            self._check_generation_active(cancel_event, deadline)
            if len(encoded) == target_len and tuple(encoded) == prefix_token_ids:
                return prefix_text, encoded

        suffix_len = max(target_len - len(prefix_token_ids), 0)
        last_error = ""
        for shift in range(self._MAX_TOKEN_REPAIR_ATTEMPTS):
            self._check_generation_active(cancel_event, deadline)
            candidate_suffix_len = suffix_len
            shifted_offset = offset + shift
            for _ in range(self._MAX_TOKEN_REPAIR_ATTEMPTS):
                self._check_generation_active(cancel_event, deadline)
                suffix_ids = self._token_slice(shifted_offset, candidate_suffix_len)
                candidate = prefix_text + self._tokenizer.decode(suffix_ids)
                self._check_generation_active(cancel_event, deadline)
                encoded = list(self._tokenizer.encode(candidate))
                self._check_generation_active(cancel_event, deadline)
                prefix_matches = (
                    tuple(encoded[: len(prefix_token_ids)]) == prefix_token_ids
                )
                if len(encoded) == target_len and prefix_matches:
                    return candidate, encoded

                last_error = (
                    f"encoded_len={len(encoded)}, target_len={target_len}, "
                    f"prefix_matches={prefix_matches}"
                )
                length_delta = target_len - len(encoded)
                if length_delta == 0:
                    # The length is right but this suffix re-tokenizes the prefix
                    # boundary. Try a different reservoir offset.
                    break
                repaired_len = candidate_suffix_len + length_delta
                if repaired_len < 0 or repaired_len == candidate_suffix_len:
                    break
                candidate_suffix_len = repaired_len

        raise ValueError(
            "Unable to construct an exact tokenizer-validated prompt after "
            f"{self._MAX_TOKEN_REPAIR_ATTEMPTS} attempts: {last_error}"
        )

    @staticmethod
    def _check_generation_active(
        cancel_event: Optional[threading.Event], deadline: Optional[float]
    ) -> None:
        if cancel_event is not None and cancel_event.is_set():
            raise _PromptGenerationExpired("prompt generation cancelled")
        if deadline is not None and time.perf_counter() >= deadline:
            raise _PromptGenerationExpired("prompt generation deadline exceeded")

    def _reserve_next(self) -> _WorkloadSpec:
        """Reserve deterministic RNG state before parallel prompt construction."""
        with self._state_lock:
            idx = self._counter
            self._counter += 1
            group = self._groups[idx % len(self._groups)]
            cfg = self._config
            input_len = group.rng.randint(cfg.input_len_min, cfg.input_len_max)
            output_len = group.rng.randint(cfg.output_len_min, cfg.output_len_max)
        return _WorkloadSpec(idx, group, input_len, output_len)

    def _generate(
        self,
        spec: _WorkloadSpec,
        cancel_event: Optional[threading.Event] = None,
        deadline: Optional[float] = None,
    ) -> _PreparedPrompt:
        idx = spec.request_id
        group = spec.group
        input_len = spec.input_len
        output_len = spec.output_len
        cfg = self._config

        self._check_generation_active(cancel_event, deadline)
        # Start from a pre-tokenized slice, but treat encode(final_prompt) as the
        # source of truth because decode and string concatenation can merge tokens.
        buf = self._big_tokens
        offset = (idx * 37) % max(1, len(buf) - input_len + 1)
        prompt, prompt_token_ids = self._find_exact_text(
            offset=offset + cfg.prefix_len,
            target_len=input_len,
            prefix_text=group.prefix_text,
            prefix_token_ids=group.prefix_token_ids,
            cancel_event=cancel_event,
            deadline=deadline,
        )
        self._check_generation_active(cancel_event, deadline)
        if len(prompt_token_ids) != input_len:
            raise AssertionError(
                f"prompt token length mismatch: {len(prompt_token_ids)} != {input_len}"
            )
        if tuple(prompt_token_ids[: cfg.prefix_len]) != group.prefix_token_ids:
            raise AssertionError(f"prompt prefix mismatch for group {group.group_id}")

        encoded_record = None
        if self._dump_file is not None:
            record = {
                "id": idx,
                "group_id": group.group_id,
                "input_len": input_len,
                "output_len": output_len,
                "prompt_preview": prompt[:200],
            }
            encoded_record = json.dumps(record, ensure_ascii=False) + "\n"

        return _PreparedPrompt(idx, prompt, output_len, encoded_record)

    def commit(self, prepared: _PreparedPrompt) -> None:
        """Record a generated prompt only once the dispatcher will submit it."""
        if (
            self._dump_file is None
            or prepared.request_id is None
            or prepared.dump_record is None
        ):
            return

        with self._dump_lock:
            self._pending_dump_records[prepared.request_id] = prepared.dump_record
            while self._next_dump_id in self._pending_dump_records:
                self._dump_file.write(
                    self._pending_dump_records.pop(self._next_dump_id)
                )
                self._next_dump_id += 1

    def next(self) -> Tuple[str, int]:
        """Generate one (prompt, output_len) pair."""
        prepared = self._generate(self._reserve_next())
        self.commit(prepared)
        return prepared.prompt, prepared.output_len


class _PromptPrefetcher:
    """Bounded, ordered prompt prefetch backed by a small thread pool."""

    def __init__(
        self,
        gen: WorkloadGenerator,
        concurrency_limit: int,
        total_requests: Optional[int] = None,
        deadline: Optional[float] = None,
    ):
        worker_count = min(4, concurrency_limit, total_requests or concurrency_limit)
        self._gen = gen
        self._total_requests = total_requests
        self._deadline = deadline
        self._cancel_event = threading.Event()
        self._loop = asyncio.get_running_loop()
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=max(worker_count, 1), thread_name_prefix="prompt_gen"
        )
        self._queue: asyncio.Queue[asyncio.Future] = asyncio.Queue(
            maxsize=max(concurrency_limit, 1)
        )
        self._native_futures: Set[concurrent.futures.Future] = set()
        self._native_futures_lock = threading.Lock()
        self._async_futures: Set[asyncio.Future] = set()
        self._closed = False
        self._scheduler_task = asyncio.create_task(self._schedule())

    async def __aenter__(self) -> "_PromptPrefetcher":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.close()

    async def _schedule(self) -> None:
        scheduled = 0
        while True:
            if self._total_requests is not None and scheduled >= self._total_requests:
                return
            if self._deadline is not None and time.perf_counter() >= self._deadline:
                return

            future = self._submit_next()
            try:
                await self._queue.put(future)
            except asyncio.CancelledError:
                future.cancel()
                raise
            scheduled += 1

    def _submit_next(self) -> asyncio.Future:
        if isinstance(self._gen, WorkloadGenerator):
            # Reserve request IDs and RNG draws on the event-loop thread so executor
            # worker start order cannot change a seeded workload's request order.
            spec = self._gen._reserve_next()
            native_future = self._executor.submit(
                self._gen._generate,
                spec,
                self._cancel_event,
                self._deadline,
            )
        else:
            native_future = self._executor.submit(self._generate_generic_prompt)

        with self._native_futures_lock:
            self._native_futures.add(native_future)
        native_future.add_done_callback(self._discard_native_future)

        future = asyncio.wrap_future(native_future, loop=self._loop)
        self._async_futures.add(future)
        future.add_done_callback(self._discard_async_future)
        return future

    def _generate_generic_prompt(self) -> _PreparedPrompt:
        self._check_generation_active()
        prompt, output_len = self._gen.next()
        self._check_generation_active()
        return _PreparedPrompt(None, prompt, output_len)

    def _check_generation_active(self) -> None:
        if self._cancel_event.is_set():
            raise _PromptGenerationExpired("prompt generation cancelled")
        if self._deadline is not None and time.perf_counter() >= self._deadline:
            raise _PromptGenerationExpired("prompt generation deadline exceeded")

    def _discard_native_future(self, future: concurrent.futures.Future) -> None:
        with self._native_futures_lock:
            self._native_futures.discard(future)

    def _discard_async_future(self, future: asyncio.Future) -> None:
        self._async_futures.discard(future)
        if not future.cancelled():
            future.exception()

    async def get(self, timeout: Optional[float] = None) -> _PreparedPrompt:
        async def _get() -> _PreparedPrompt:
            future = await self._queue.get()
            return await future

        if timeout is None:
            return await _get()
        return await asyncio.wait_for(_get(), timeout=timeout)

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._cancel_event.set()
        self._scheduler_task.cancel()
        await asyncio.gather(self._scheduler_task, return_exceptions=True)

        while not self._queue.empty():
            self._queue.get_nowait()
        async_futures = list(self._async_futures)
        for future in async_futures:
            future.cancel()
        if async_futures:
            await asyncio.gather(*async_futures, return_exceptions=True)
        with self._native_futures_lock:
            native_futures = list(self._native_futures)
        for future in native_futures:
            future.cancel()
        self._executor.shutdown(wait=False, cancel_futures=True)


# ---------------------------------------------------------------------------
# OfflineRunner
# ---------------------------------------------------------------------------


@dataclass
class _RequestRecord:
    submit_time: float
    finish_time: float
    response: Dict[str, Any]


class _DispatchResult(NamedTuple):
    records: List[_RequestRecord]
    cancelled_count: int
    submitted_count: int
    first_submit_time: Optional[float]
    terminal_time: float


class OfflineRunner:
    def __init__(
        self,
        port: int,
        config: OfflineBenchConfig,
        tokenizer_path: str,
        checkpoint_path: str = "",
        model_type: str = "",
        result_dir: str = ".",
        seed: int = 42,
        profile: bool = False,
        profile_steps: int = 50,
    ):
        self._port = port
        self._config = config
        self._tokenizer_path = tokenizer_path
        self._checkpoint_path = checkpoint_path
        self._model_type = model_type
        self._result_dir = result_dir
        self._seed = seed
        self._profile = profile
        self._profile_steps = profile_steps
        self._tokenizer = None
        self._status_samples: List[Dict[str, Any]] = []
        self._status_sample_errors: int = 0

    def run(self) -> OfflineMetrics:
        self._config.validate()
        self._load_tokenizer()

        dump_file: Optional[TextIO] = None
        try:
            if self._config.dump_workload:
                dump_path = self._config.dump_workload
                os.makedirs(os.path.dirname(dump_path) or ".", exist_ok=True)
                dump_file = open(dump_path, "w")
                logging.info(f"Dumping workload to {dump_path}")

            gen = WorkloadGenerator(
                self._config,
                self._tokenizer,
                seed=self._config.seed,
                dump_file=dump_file,
            )

            metrics = self._run(gen)
        finally:
            if dump_file is not None:
                dump_file.close()

        metrics.print_table()
        metrics.save_json(self._result_dir)
        self._save_status_samples()
        metrics.raise_if_all_requests_failed()
        return metrics

    def _run(self, gen: WorkloadGenerator) -> OfflineMetrics:
        cfg = self._config

        engine_status = self._query_status()
        logging.info(f"Engine status before: {engine_status}")

        concurrency_limit = cfg.concurrency_limit

        if cfg.total_requests > 0:
            logging.info(
                f"Fixed-workload mode: {cfg.total_requests} requests, "
                f"concurrency_limit={concurrency_limit}, seed={cfg.seed}, "
                f"input_len=[{cfg.input_len_min}, {cfg.input_len_max}], "
                f"output_len=[{cfg.output_len_min}, {cfg.output_len_max}]"
            )
        else:
            logging.info(
                f"Duration mode: {cfg.duration_s}s, concurrency_limit={concurrency_limit}, "
                f"input_len=[{cfg.input_len_min}, {cfg.input_len_max}], "
                f"output_len=[{cfg.output_len_min}, {cfg.output_len_max}], "
                f"prefix_groups={cfg.prefix_groups}, prefix_len={cfg.prefix_len}"
            )

        if cfg.total_requests > 0:
            dispatch_result = asyncio.run(
                self._dispatch_fixed(
                    gen, concurrency_limit, cfg.total_requests, cfg.drain_timeout_s
                )
            )
        else:
            dispatch_result = asyncio.run(
                self._dispatch(
                    gen, concurrency_limit, cfg.duration_s, cfg.drain_timeout_s
                )
            )

        records = dispatch_result.records
        cancelled_count = dispatch_result.cancelled_count
        submitted_count = dispatch_result.submitted_count

        # Include cancelled requests in the throughput window. Their request records do not have a
        # finish time, so dispatch reports the complete request lifecycle explicitly.
        if dispatch_result.first_submit_time is not None:
            wall_time = max(
                dispatch_result.terminal_time - dispatch_result.first_submit_time, 0.0
            )
        else:
            wall_time = 0.0

        logging.info(
            f"Finished: {len(records)} completed, {cancelled_count} cancelled, "
            f"wall_time={wall_time:.1f}s"
        )

        metrics = self._analyze(
            records,
            wall_time,
            engine_status,
            submitted_count,
            cancelled_count,
        )
        return metrics

    async def _dispatch(
        self,
        gen: WorkloadGenerator,
        concurrency_limit: int,
        duration_s: float,
        drain_timeout_s: int = 0,
    ) -> _DispatchResult:
        """Dispatch requests for duration_s, then drain all in-flight.

        If drain_timeout_s > 0, cancel remaining requests after that many
        seconds and report based on whatever has completed so far.

        Returns completed records, counts, the first actual submit time, and the time when every
        submitted request has completed or been cancelled.
        """
        sem = asyncio.Semaphore(concurrency_limit)
        records: List[_RequestRecord] = []
        lock = asyncio.Lock()
        submitted = 0
        first_submit_time: Optional[float] = None
        loop = asyncio.get_running_loop()

        timeout = aiohttp.ClientTimeout(total=3 * 3600)
        connector = aiohttp.TCPConnector(limit=0)
        t_start = time.perf_counter()
        deadline = t_start + duration_s
        prefetcher = _PromptPrefetcher(gen, concurrency_limit, deadline=deadline)
        async with prefetcher, aiohttp.ClientSession(
            timeout=timeout, connector=connector
        ) as session:
            pending_tasks: Set[asyncio.Task] = set()
            profile_futures: Set[asyncio.Future] = set()

            sampler_task = asyncio.create_task(
                self._status_sampler(
                    interval=5.0, t_start=t_start, duration=duration_s + 600
                )
            )

            while time.perf_counter() < deadline:
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    break
                try:
                    prepared = await prefetcher.get(timeout=remaining)
                except (asyncio.TimeoutError, _PromptGenerationExpired):
                    break

                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    break
                try:
                    await asyncio.wait_for(sem.acquire(), timeout=remaining)
                except asyncio.TimeoutError:
                    break

                if isinstance(gen, WorkloadGenerator):
                    gen.commit(prepared)
                submit_time = time.perf_counter()
                if first_submit_time is None:
                    first_submit_time = submit_time
                submitted += 1

                task = asyncio.create_task(
                    self._send_and_release(
                        session,
                        sem,
                        prepared.prompt,
                        prepared.output_len,
                        submit_time,
                        records,
                        lock,
                    )
                )
                pending_tasks.add(task)
                task.add_done_callback(pending_tasks.discard)

                if self._profile and submitted == concurrency_limit:
                    self._schedule_profile_trigger(loop, profile_futures)

                if submitted % 100 == 0:
                    elapsed = time.perf_counter() - t_start
                    inflight = len(pending_tasks)
                    logging.info(
                        f"Submitted {submitted}, elapsed={elapsed:.1f}s/{duration_s:.0f}s, "
                        f"inflight={inflight}/{concurrency_limit}"
                    )

            drain_count = len(pending_tasks)
            drain_deadline_desc = (
                f"{drain_timeout_s}s" if drain_timeout_s > 0 else "unlimited"
            )
            logging.info(
                f"Dispatch done: {submitted} submitted. "
                f"Draining {drain_count} in-flight requests "
                f"(timeout={drain_deadline_desc})..."
            )

            cancelled_count = await self._drain_pending(pending_tasks, drain_timeout_s)
            terminal_time = time.perf_counter()

            sampler_task.cancel()
            try:
                await sampler_task
            except asyncio.CancelledError:
                pass

        if cancelled_count > 0:
            logging.info(
                f"Drain finished: {len(records)} completed, "
                f"{cancelled_count} cancelled"
            )
        else:
            logging.info(f"All requests drained. Total records: {len(records)}")
        return _DispatchResult(
            records, cancelled_count, submitted, first_submit_time, terminal_time
        )

    async def _drain_pending(
        self, pending_tasks: Set[asyncio.Task], drain_timeout_s: int
    ) -> int:
        cancelled_count = 0
        if pending_tasks:
            remaining = set(pending_tasks)
            drain_start = time.perf_counter()
            drain_deadline = (
                drain_start + drain_timeout_s if drain_timeout_s > 0 else float("inf")
            )
            last_log_time = drain_start
            while remaining:
                wait_timeout = 10.0
                if drain_timeout_s > 0:
                    time_left = drain_deadline - time.perf_counter()
                    if time_left <= 0:
                        break
                    wait_timeout = min(wait_timeout, time_left)
                _, remaining = await asyncio.wait(
                    remaining, timeout=wait_timeout, return_when=asyncio.FIRST_COMPLETED
                )
                now = time.perf_counter()
                if now - last_log_time >= 10.0 or not remaining:
                    elapsed_drain = now - drain_start
                    logging.info(
                        f"Draining: {len(remaining)} requests remaining, "
                        f"drain elapsed={elapsed_drain:.1f}s"
                    )
                    last_log_time = now

            if remaining:
                cancelled_count = len(remaining)
                logging.warning(
                    f"Drain timeout ({drain_timeout_s}s) reached, "
                    f"cancelling {cancelled_count} in-flight requests"
                )
                for task in remaining:
                    task.cancel()
                await asyncio.gather(*remaining, return_exceptions=True)
        return cancelled_count

    async def _dispatch_fixed(
        self,
        gen: WorkloadGenerator,
        concurrency_limit: int,
        total_requests: int,
        drain_timeout_s: int = 0,
    ) -> _DispatchResult:
        """Submit exactly total_requests, then drain all in-flight.

        Returns completed records, counts, the first actual submit time, and the time when every
        submitted request has completed or been cancelled.
        """
        sem = asyncio.Semaphore(concurrency_limit)
        records: List[_RequestRecord] = []
        lock = asyncio.Lock()
        submitted = 0
        first_submit_time: Optional[float] = None
        loop = asyncio.get_running_loop()

        timeout = aiohttp.ClientTimeout(total=3 * 3600)
        connector = aiohttp.TCPConnector(limit=0)
        t_start = time.perf_counter()
        prefetcher = _PromptPrefetcher(
            gen, concurrency_limit, total_requests=total_requests
        )
        async with prefetcher, aiohttp.ClientSession(
            timeout=timeout, connector=connector
        ) as session:
            pending_tasks: Set[asyncio.Task] = set()
            profile_futures: Set[asyncio.Future] = set()

            sampler_task = asyncio.create_task(
                self._status_sampler(interval=5.0, t_start=t_start, duration=3 * 3600)
            )

            while submitted < total_requests:
                prepared = await prefetcher.get()

                # drain_timeout starts only after the fixed workload has been fully submitted.
                await sem.acquire()
                if isinstance(gen, WorkloadGenerator):
                    gen.commit(prepared)
                submit_time = time.perf_counter()
                if first_submit_time is None:
                    first_submit_time = submit_time
                submitted += 1

                task = asyncio.create_task(
                    self._send_and_release(
                        session,
                        sem,
                        prepared.prompt,
                        prepared.output_len,
                        submit_time,
                        records,
                        lock,
                    )
                )
                pending_tasks.add(task)
                task.add_done_callback(pending_tasks.discard)

                if self._profile and submitted == concurrency_limit:
                    self._schedule_profile_trigger(loop, profile_futures)

                if submitted % 100 == 0:
                    elapsed = time.perf_counter() - t_start
                    inflight = len(pending_tasks)
                    logging.info(
                        f"Submitted {submitted}/{total_requests}, elapsed={elapsed:.1f}s, "
                        f"inflight={inflight}/{concurrency_limit}"
                    )

            logging.info(
                f"Submission complete: {submitted}/{total_requests} requests submitted. "
                f"Draining in-flight requests..."
            )

            cancelled_count = await self._drain_pending(pending_tasks, drain_timeout_s)
            terminal_time = time.perf_counter()

            sampler_task.cancel()
            try:
                await sampler_task
            except asyncio.CancelledError:
                pass

        if cancelled_count > 0:
            logging.info(
                f"Fixed-workload done: {len(records)} completed, "
                f"{cancelled_count} cancelled"
            )
        else:
            logging.info(f"Fixed-workload done: all {len(records)} completed")
        return _DispatchResult(
            records, cancelled_count, submitted, first_submit_time, terminal_time
        )

    def _schedule_profile_trigger(
        self,
        loop: asyncio.AbstractEventLoop,
        profile_futures: Set[asyncio.Future],
    ) -> None:
        future = loop.run_in_executor(None, self._trigger_profile)
        profile_futures.add(future)

        def _cleanup(done: asyncio.Future) -> None:
            profile_futures.discard(done)
            try:
                done.result()
            except Exception as e:
                logging.warning(f"Profile trigger executor failed: {e}")

        future.add_done_callback(_cleanup)

    def _trigger_profile(self) -> None:
        try:
            resp = requests.post(
                f"http://127.0.0.1:{self._port}/start_profile",
                json={
                    "trace_name": "offline_bench",
                    "start_step": 0,
                    "num_steps": self._profile_steps,
                },
                timeout=60,
            )
            logging.info(
                f"Profile triggered: num_steps={self._profile_steps}, response={resp.status_code}"
            )
        except Exception as e:
            logging.warning(f"Failed to trigger profile: {e}")

    async def _send_and_release(
        self,
        session: aiohttp.ClientSession,
        sem: asyncio.Semaphore,
        prompt: str,
        output_len: int,
        submit_time: float,
        records: List[_RequestRecord],
        lock: asyncio.Lock,
    ) -> None:
        try:
            resp = await self._send_one(session, prompt, output_len)
            finish_time = time.perf_counter()
            async with lock:
                records.append(
                    _RequestRecord(
                        submit_time=submit_time, finish_time=finish_time, response=resp
                    )
                )
        finally:
            sem.release()

    # ---- status sampling ----

    async def _status_sampler(
        self, interval: float, t_start: float, duration: float
    ) -> None:
        deadline = t_start + duration
        while time.perf_counter() < deadline:
            try:
                status = await asyncio.get_running_loop().run_in_executor(
                    None, self._query_status
                )
                status["timestamp"] = time.time()
                status["elapsed_s"] = time.perf_counter() - t_start
                total_kv = status.get("total_kv_cache", 0)
                avail_kv = status.get("available_kv_cache", 0)
                conc_limit = status.get("concurrency_limit", 0)
                avail_conc = status.get("available_concurrency", 0)
                if total_kv > 0:
                    status["kv_utilization"] = 1.0 - avail_kv / total_kv
                if conc_limit > 0:
                    status["concurrency_utilization"] = 1.0 - avail_conc / conc_limit
                self._status_samples.append(status)
            except Exception as e:
                self._status_sample_errors += 1
                if self._status_sample_errors <= 3:
                    logging.warning(
                        f"Status sample failed ({self._status_sample_errors}): {e}"
                    )
            await asyncio.sleep(interval)

    def _save_status_samples(self) -> None:
        if not self._status_samples:
            return
        path = os.path.join(self._result_dir, "status_samples.jsonl")
        with open(path, "w") as f:
            for s in self._status_samples:
                f.write(json.dumps(s) + "\n")
        logging.info(f"Saved {len(self._status_samples)} status samples to {path}")

    # ---- shared HTTP request ----

    async def _send_one(
        self, session: aiohttp.ClientSession, prompt: str, output_len: int
    ) -> Dict[str, Any]:
        generate_config: Dict[str, Any] = {
            "max_new_tokens": output_len,
            "min_new_tokens": output_len,
            "top_k": 1,
        }
        if self._config.num_return_sequences > 1:
            generate_config.update(
                {
                    "num_beams": 1,
                    "num_return_sequences": self._config.num_return_sequences,
                    "do_sample": True,
                }
            )
        pload = {
            "prompt": prompt,
            "generate_config": generate_config,
            "aux_info": True,
        }
        try:
            async with session.post(
                f"http://127.0.0.1:{self._port}", json=pload
            ) as resp:
                return await resp.json()
        except Exception as e:
            return {"error": str(e)}

    # ---- engine status ----

    def _query_status(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        try:
            cache = requests.get(
                f"http://127.0.0.1:{self._port}/cache_status", timeout=60
            ).json()
            worker = requests.get(
                f"http://127.0.0.1:{self._port}/worker_status", timeout=60
            ).json()

            cache_results = cache.get("results") or [cache]
            if cache_results:
                per_dp_kv_cache = []
                for dp_id, shard in enumerate(cache_results):
                    total_kv = int(shard.get("total_kv_cache", 0))
                    available_kv = int(shard.get("available_kv_cache", 0))
                    block_size = max(int(shard.get("block_size", 1)), 1)
                    utilization = 1.0 - available_kv / total_kv if total_kv > 0 else 0.0
                    per_dp_kv_cache.append(
                        {
                            "dp_id": dp_id,
                            "total_kv_cache": total_kv,
                            "available_kv_cache": available_kv,
                            "block_size": block_size,
                            "utilization": utilization,
                        }
                    )

                result["per_dp_kv_cache"] = per_dp_kv_cache
                result["total_kv_cache"] = sum(
                    s["total_kv_cache"] for s in per_dp_kv_cache
                )
                result["available_kv_cache"] = sum(
                    s["available_kv_cache"] for s in per_dp_kv_cache
                )
                result["total_kv_cache_blocks"] = sum(
                    s["total_kv_cache"] // s["block_size"] for s in per_dp_kv_cache
                )
                result["kv_cache_max_dp_utilization"] = max(
                    s["utilization"] for s in per_dp_kv_cache
                )
                block_sizes = {s["block_size"] for s in per_dp_kv_cache}
                result["block_size"] = (
                    next(iter(block_sizes)) if len(block_sizes) == 1 else 0
                )

            result["concurrency_limit"] = int(
                worker.get("frontend_concurrency_limit", 0)
            )
            result["available_concurrency"] = int(
                worker.get("frontend_available_concurrency", 0)
            )

            task_list = worker.get("running_task_info", [])
            result["running_streams"] = sum(
                1 for t in task_list if not t.get("is_waiting", False)
            )
            result["waiting_streams"] = sum(
                1 for t in task_list if t.get("is_waiting", False)
            )
        except Exception as e:
            logging.warning(f"Failed to query engine status: {e}")
        return result

    # ---- tokenizer ----

    def _load_tokenizer(self):
        from rtp_llm.test.perf_test.test_util import _load_tokenizer

        self._tokenizer = _load_tokenizer(
            self._tokenizer_path,
            checkpoint_path=self._checkpoint_path,
            model_type=self._model_type,
        )

    # ---- analysis ----

    def _analyze(
        self,
        records: List[_RequestRecord],
        wall_time: float,
        engine_status: Dict[str, Any],
        submitted_count: int,
        cancelled_count: int,
    ) -> OfflineMetrics:
        metrics = OfflineMetrics()
        metrics.total_wall_time_s = wall_time
        metrics.total_submitted = submitted_count
        metrics.cancelled_requests = cancelled_count

        responses = [r.response for r in records]

        success_aux: List[Dict[str, Any]] = []
        success_sequence_aux: List[Dict[str, Any]] = []
        fail_reasons: Dict[str, int] = {}
        for resp in responses:
            if isinstance(resp, Exception):
                reason = f"exception:{type(resp).__name__}"
                fail_reasons[reason] = fail_reasons.get(reason, 0) + 1
                continue
            if (
                isinstance(resp, dict)
                and "error" not in resp
                and "error_code" not in resp
            ):
                raw_aux = resp.get("aux_info", {})
                aux_list = raw_aux if isinstance(raw_aux, list) else [raw_aux]
                aux_list = [aux for aux in aux_list if isinstance(aux, dict) and aux]
                if len(aux_list) != self._config.num_return_sequences:
                    reason = (
                        "aux_info_count_mismatch: "
                        f"expected={self._config.num_return_sequences}, actual={len(aux_list)}"
                    )
                    fail_reasons[reason] = fail_reasons.get(reason, 0) + 1
                    continue

                # Request-level input is counted once, while every returned sequence
                # contributes generated tokens. This keeps TPS correct for n>1 without
                # multiplying the shared prompt tokens.
                request_aux = dict(aux_list[0])
                request_aux["output_len"] = sum(
                    aux.get("output_len", 0) for aux in aux_list
                )
                for field in (
                    "cost_time",
                    "wait_time",
                    "first_token_cost_time",
                    "first_token_cost_time_us",
                ):
                    request_aux[field] = max(
                        (aux.get(field, 0) for aux in aux_list), default=0
                    )
                success_aux.append(request_aux)
                success_sequence_aux.extend(aux_list)
            else:
                error_code = (
                    resp.get("error_code", "unknown")
                    if isinstance(resp, dict)
                    else "unknown"
                )
                error_msg = (
                    (
                        resp.get("message")
                        or resp.get("error_message")
                        or resp.get("error")
                        or ""
                    )
                    if isinstance(resp, dict)
                    else str(resp)
                )
                reason = f"error_code={error_code}: {str(error_msg)[:120]}"
                fail_reasons[reason] = fail_reasons.get(reason, 0) + 1

        metrics.success_requests = len(success_aux)
        metrics.fail_requests = max(
            submitted_count - metrics.success_requests - cancelled_count, 0
        )
        if cancelled_count > 0:
            if self._config.drain_timeout_s > 0:
                reason = f"cancelled: drain_timeout={self._config.drain_timeout_s}s"
            else:
                reason = "cancelled"
            fail_reasons[reason] = fail_reasons.get(reason, 0) + cancelled_count

        if fail_reasons:
            print(f"\n--- Failed Requests ({metrics.fail_requests} total) ---")
            for reason, count in sorted(fail_reasons.items(), key=lambda x: -x[1]):
                print(f"  [{count:4d}] {reason}")

        if not success_aux:
            return metrics

        input_lens = [a.get("input_len", 0) for a in success_aux]
        output_lens = [a.get("output_len", 0) for a in success_aux]
        cost_times = [a.get("cost_time", 0.0) for a in success_aux]
        wait_times = [a.get("wait_time", 0.0) for a in success_aux]
        reuse_lens = [a.get("reuse_len", 0) for a in success_aux]

        total_input = sum(input_lens)
        total_output = sum(output_lens)

        metrics.output_tps = total_output / wall_time if wall_time > 0 else 0
        metrics.input_tps = total_input / wall_time if wall_time > 0 else 0
        metrics.total_tps = (
            (total_input + total_output) / wall_time if wall_time > 0 else 0
        )

        latencies_s = [t / 1000.0 for t in cost_times]
        latencies_s.sort()
        n = len(latencies_s)
        metrics.avg_request_latency_s = sum(latencies_s) / n
        metrics.p50_request_latency_s = latencies_s[n // 2]
        metrics.p99_request_latency_s = latencies_s[min(int(n * 0.99), n - 1)]

        metrics.avg_wait_time_ms = sum(wait_times) / n
        metrics.max_wait_time_ms = max(wait_times)

        first_token_times = [
            a.get(
                "first_token_cost_time", a.get("first_token_cost_time_us", 0) / 1000.0
            )
            for a in success_aux
        ]
        ttft_ms_list = [t for t in first_token_times if t > 0]
        if ttft_ms_list:
            ttft_ms_list.sort()
            nt = len(ttft_ms_list)
            metrics.avg_ttft_ms = sum(ttft_ms_list) / nt
            metrics.p50_ttft_ms = ttft_ms_list[nt // 2]
            metrics.p99_ttft_ms = ttft_ms_list[min(int(nt * 0.99), nt - 1)]

        tpot_ms_list = []
        for a in success_sequence_aux:
            ct = a.get("cost_time", 0.0)
            ft = a.get(
                "first_token_cost_time", a.get("first_token_cost_time_us", 0) / 1000.0
            )
            ol = a.get("output_len", 0)
            if ol > 1 and ct > ft:
                tpot_ms_list.append((ct - ft) / (ol - 1))
        if tpot_ms_list:
            tpot_ms_list.sort()
            nt = len(tpot_ms_list)
            metrics.avg_tpot_ms = sum(tpot_ms_list) / nt
            metrics.p50_tpot_ms = tpot_ms_list[nt // 2]
            metrics.p99_tpot_ms = tpot_ms_list[min(int(nt * 0.99), nt - 1)]

        metrics.avg_input_len = total_input / n
        metrics.avg_output_len = total_output / n
        metrics.input_len_range = (min(input_lens), max(input_lens))
        metrics.output_len_range = (min(output_lens), max(output_lens))

        metrics.total_reuse_tokens = sum(reuse_lens)
        metrics.total_compute_tokens = sum(
            il - rl for il, rl in zip(input_lens, reuse_lens)
        )
        metrics.avg_reuse_len = metrics.total_reuse_tokens / n
        metrics.avg_reuse_ratio = (
            metrics.avg_reuse_len / metrics.avg_input_len
            if metrics.avg_input_len > 0
            else 0
        )

        # /cache_status reports total_kv_cache in tokens. Convert to blocks only
        # for the block-count field; keeping both fields makes the units explicit.
        metrics.kv_cache_total_tokens = int(engine_status.get("total_kv_cache", 0))
        metrics.kv_cache_block_size = max(int(engine_status.get("block_size", 1)), 1)
        metrics.kv_cache_total_blocks = int(
            engine_status.get(
                "total_kv_cache_blocks",
                metrics.kv_cache_total_tokens // metrics.kv_cache_block_size,
            )
        )
        if self._status_samples:
            kv_utils = [
                s["kv_utilization"]
                for s in self._status_samples
                if "kv_utilization" in s
            ]
            dp_kv_utils = [
                s["kv_cache_max_dp_utilization"]
                for s in self._status_samples
                if "kv_cache_max_dp_utilization" in s
            ]
            conc_utils = [
                s["concurrency_utilization"]
                for s in self._status_samples
                if "concurrency_utilization" in s
            ]
            if kv_utils:
                metrics.kv_cache_avg_utilization = sum(kv_utils) / len(kv_utils)
                metrics.kv_cache_max_utilization = max(kv_utils)
                metrics.kv_cache_min_utilization = min(kv_utils)
            if dp_kv_utils:
                metrics.kv_cache_max_dp_utilization = max(dp_kv_utils)
            if conc_utils:
                metrics.concurrency_avg_utilization = sum(conc_utils) / len(conc_utils)
                metrics.concurrency_max_utilization = max(conc_utils)

        return metrics

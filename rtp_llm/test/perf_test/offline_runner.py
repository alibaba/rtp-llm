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
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, TextIO, Tuple

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

    duration_s: int = 300
    drain_timeout_s: int = 0  # 0 = wait forever
    concurrency_limit: int = 256
    dump_workload: str = ""

    def validate(self):
        assert self.input_len_min > 0, "input_len_min must be > 0"
        assert self.input_len_min <= self.input_len_max, (
            f"input_len_min ({self.input_len_min}) must be <= input_len_max ({self.input_len_max})"
        )
        assert self.output_len_min > 0, "output_len_min must be > 0"
        assert self.output_len_min <= self.output_len_max, (
            f"output_len_min ({self.output_len_min}) must be <= output_len_max ({self.output_len_max})"
        )
        assert self.prefix_groups >= 1, "prefix_groups must be >= 1"
        assert self.prefix_len >= 0, "prefix_len must be >= 0"
        assert self.prefix_len <= self.input_len_min, (
            f"prefix_len ({self.prefix_len}) must be <= input_len_min ({self.input_len_min})"
        )
        assert self.duration_s > 0, "duration_s must be > 0"


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

    kv_cache_total_blocks: int = 0
    kv_cache_block_size: int = 0
    kv_cache_available_before: int = 0
    kv_cache_available_after: int = 0
    kv_cache_peak_usage_ratio: float = 0.0

    concurrency_limit: int = 0
    available_concurrency_before: int = 0

    kv_cache_avg_utilization: float = 0.0
    kv_cache_max_utilization: float = 0.0
    kv_cache_min_utilization: float = 0.0
    concurrency_avg_utilization: float = 0.0
    concurrency_max_utilization: float = 0.0

    def print_table(self):
        lines = [
            "================== Offline Throughput Benchmark ==================",
            f"Submitted:             {self.total_submitted}",
            f"Success / Fail / Cancelled: {self.success_requests} / {self.fail_requests} / {self.cancelled_requests}",
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
            f"Total Blocks:          {self.kv_cache_total_blocks:,} "
            f"(block_size={self.kv_cache_block_size})",
            f"Available Before:      {self.kv_cache_available_before:,}",
            f"Available After:       {self.kv_cache_available_after:,}",
            f"Peak Usage:            {self.kv_cache_peak_usage_ratio * 100:.1f}%",
            "--- Concurrency ---",
            f"Limit:                 {self.concurrency_limit}",
            f"Available Before:      {self.available_concurrency_before}",
        ]
        if self.kv_cache_max_utilization > 0:
            lines += [
                "--- Runtime Utilization ---",
                f"KV Cache:              avg={self.kv_cache_avg_utilization * 100:.1f}%  "
                f"max={self.kv_cache_max_utilization * 100:.1f}%  "
                f"min={self.kv_cache_min_utilization * 100:.1f}%",
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
    rng: random.Random


class WorkloadGenerator:
    """On-demand request generator with independent per-prefix-group RNGs.

    Prompt generation uses token-id-based construction for speed:
    builds text by repeating base sentences, then uses tokenizer only once
    for final length verification and trim.
    """

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

        # Pre-compute a large text buffer and its token count for fast slicing
        big_text = " ".join(BASE_TEXTS) * 50
        self._big_tokens = self._tokenizer.encode(big_text)
        self._big_text = big_text

        self._groups: List[_PrefixGroup] = []
        for g in range(config.prefix_groups):
            prefix_text = ""
            if config.prefix_len > 0:
                prefix_ids = self._big_tokens[:config.prefix_len]
                prefix_text = self._tokenizer.decode(prefix_ids)
            self._groups.append(
                _PrefixGroup(
                    group_id=g,
                    prefix_text=prefix_text,
                    rng=random.Random(seed + g),
                )
            )

    def next(self) -> Tuple[str, int]:
        """Generate one (prompt, output_len) pair."""
        idx = self._counter
        self._counter += 1
        group = self._groups[idx % len(self._groups)]
        cfg = self._config

        input_len = group.rng.randint(cfg.input_len_min, cfg.input_len_max)
        output_len = group.rng.randint(cfg.output_len_min, cfg.output_len_max)

        # Fast prompt construction: slice pre-tokenized buffer with offset for variety
        offset = (idx * 37) % max(1, len(self._big_tokens) - input_len)
        token_ids = self._big_tokens[offset:offset + input_len]
        if len(token_ids) < input_len:
            token_ids = (self._big_tokens * 2)[offset:offset + input_len]

        if cfg.prefix_len > 0:
            prompt = group.prefix_text + self._tokenizer.decode(token_ids[cfg.prefix_len:])
        else:
            prompt = self._tokenizer.decode(token_ids)

        if self._dump_file is not None:
            record = {
                "id": idx,
                "group_id": group.group_id,
                "input_len": input_len,
                "output_len": output_len,
                "prompt_preview": prompt[:200],
            }
            self._dump_file.write(json.dumps(record, ensure_ascii=False) + "\n")

        return prompt, output_len


# ---------------------------------------------------------------------------
# OfflineRunner
# ---------------------------------------------------------------------------


@dataclass
class _RequestRecord:
    submit_time: float
    finish_time: float
    response: Dict[str, Any]


class OfflineRunner:
    def __init__(
        self,
        port: int,
        config: OfflineBenchConfig,
        tokenizer_path: str,
        result_dir: str = ".",
        seed: int = 42,
        profile: bool = False,
        profile_steps: int = 50,
    ):
        self._port = port
        self._config = config
        self._tokenizer_path = tokenizer_path
        self._result_dir = result_dir
        self._seed = seed
        self._profile = profile
        self._profile_steps = profile_steps
        self._tokenizer = None
        self._status_samples: List[Dict[str, Any]] = []

    def run(self) -> OfflineMetrics:
        self._config.validate()
        self._load_tokenizer()

        dump_file: Optional[TextIO] = None
        if self._config.dump_workload:
            dump_path = self._config.dump_workload
            os.makedirs(os.path.dirname(dump_path) or ".", exist_ok=True)
            dump_file = open(dump_path, "w")
            logging.info(f"Dumping workload to {dump_path}")

        gen = WorkloadGenerator(
            self._config, self._tokenizer, seed=self._seed, dump_file=dump_file
        )

        try:
            metrics = self._run(gen)
        finally:
            if dump_file is not None:
                dump_file.close()

        metrics.print_table()
        metrics.save_json(self._result_dir)
        self._save_status_samples()
        return metrics

    def _run(self, gen: WorkloadGenerator) -> OfflineMetrics:
        cfg = self._config

        status_before = self._query_status()
        logging.info(f"Engine status before: {status_before}")

        concurrency_limit = cfg.concurrency_limit

        logging.info(
            f"Running for {cfg.duration_s}s, concurrency_limit={concurrency_limit}, "
            f"input_len=[{cfg.input_len_min}, {cfg.input_len_max}], "
            f"output_len=[{cfg.output_len_min}, {cfg.output_len_max}], "
            f"prefix_groups={cfg.prefix_groups}, prefix_len={cfg.prefix_len}"
        )

        t_start = time.perf_counter()
        records, cancelled_count = asyncio.run(
            self._dispatch(gen, concurrency_limit, cfg.duration_s, cfg.drain_timeout_s)
        )
        t_end = time.perf_counter()

        status_after = self._query_status()

        # Wall time = first submit to last completion
        if records:
            wall_time = max(r.finish_time for r in records) - min(r.submit_time for r in records)
        else:
            wall_time = t_end - t_start

        logging.info(
            f"Finished: {len(records)} completed, {cancelled_count} cancelled, "
            f"wall_time={wall_time:.1f}s"
        )

        metrics = self._analyze(records, wall_time, status_before, status_after)
        metrics.cancelled_requests = cancelled_count
        return metrics

    async def _dispatch(
        self,
        gen: WorkloadGenerator,
        concurrency_limit: int,
        duration_s: float,
        drain_timeout_s: int = 0,
    ) -> Tuple[List[_RequestRecord], int]:
        """Dispatch requests for duration_s, then drain all in-flight.

        If drain_timeout_s > 0, cancel remaining requests after that many
        seconds and report based on whatever has completed so far.

        Returns (records, cancelled_count).
        """
        sem = asyncio.Semaphore(concurrency_limit)
        records: List[_RequestRecord] = []
        lock = asyncio.Lock()
        submitted = 0
        loop = asyncio.get_event_loop()

        executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=4, thread_name_prefix="prompt_gen"
        )

        async def _next_prompt() -> Tuple[str, int]:
            return await loop.run_in_executor(executor, gen.next)

        timeout = aiohttp.ClientTimeout(total=3 * 3600)
        connector = aiohttp.TCPConnector(limit=0)

        async with aiohttp.ClientSession(
            timeout=timeout, connector=connector
        ) as session:
            t_start = time.perf_counter()
            deadline = t_start + duration_s
            pending_tasks: List[asyncio.Task] = []

            sampler_task = asyncio.create_task(
                self._status_sampler(interval=5.0, t_start=t_start, duration=duration_s + 600)
            )

            while time.perf_counter() < deadline:
                await sem.acquire()
                if time.perf_counter() >= deadline:
                    sem.release()
                    break

                prompt, output_len = await _next_prompt()
                submit_time = time.perf_counter()
                submitted += 1

                task = asyncio.create_task(
                    self._send_and_release(
                        session, sem, prompt, output_len, submit_time, records, lock
                    )
                )
                pending_tasks.append(task)

                if self._profile and submitted == concurrency_limit:
                    self._trigger_profile()

                if submitted % 100 == 0:
                    elapsed = time.perf_counter() - t_start
                    inflight = concurrency_limit - sem._value
                    logging.info(
                        f"Submitted {submitted}, elapsed={elapsed:.1f}s/{duration_s:.0f}s, "
                        f"inflight={inflight}/{concurrency_limit}"
                    )

            drain_count = concurrency_limit - sem._value
            drain_deadline_desc = f"{drain_timeout_s}s" if drain_timeout_s > 0 else "unlimited"
            logging.info(
                f"Dispatch done: {submitted} submitted. "
                f"Draining {drain_count} in-flight requests "
                f"(timeout={drain_deadline_desc})..."
            )

            cancelled_count = 0
            if pending_tasks:
                remaining = set(t for t in pending_tasks if not t.done())
                drain_start = time.perf_counter()
                drain_deadline = drain_start + drain_timeout_s if drain_timeout_s > 0 else float("inf")
                last_log_time = drain_start
                while remaining:
                    wait_timeout = 10.0
                    if drain_timeout_s > 0:
                        time_left = drain_deadline - time.perf_counter()
                        if time_left <= 0:
                            break
                        wait_timeout = min(wait_timeout, time_left)
                    done, remaining = await asyncio.wait(
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

            sampler_task.cancel()
            try:
                await sampler_task
            except asyncio.CancelledError:
                pass

        executor.shutdown(wait=False)
        if cancelled_count > 0:
            logging.info(
                f"Drain finished: {len(records)} completed, "
                f"{cancelled_count} cancelled"
            )
        else:
            logging.info(f"All requests drained. Total records: {len(records)}")
        return records, cancelled_count

    def _trigger_profile(self) -> None:
        try:
            resp = requests.post(
                f"http://127.0.0.1:{self._port}/start_profile",
                json={"trace_name": "offline_bench", "start_step": 0, "num_steps": self._profile_steps},
                timeout=10,
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
                records.append(_RequestRecord(
                    submit_time=submit_time, finish_time=finish_time, response=resp
                ))
        finally:
            sem.release()

    # ---- status sampling ----

    async def _status_sampler(self, interval: float, t_start: float, duration: float) -> None:
        deadline = t_start + duration
        while time.perf_counter() < deadline:
            try:
                status = await asyncio.get_event_loop().run_in_executor(
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
            except Exception:
                pass
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
        pload = {
            "prompt": prompt,
            "generate_config": {
                "max_new_tokens": output_len,
                "min_new_tokens": output_len,
                "top_k": 1,
            },
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
                f"http://127.0.0.1:{self._port}/cache_status", timeout=10
            ).json()
            worker = requests.get(
                f"http://127.0.0.1:{self._port}/worker_status", timeout=10
            ).json()

            cache_results = cache.get("results", [cache])
            if cache_results:
                c = cache_results[0]
                result["total_kv_cache"] = int(c.get("total_kv_cache", 0))
                result["available_kv_cache"] = int(c.get("available_kv_cache", 0))
                result["block_size"] = int(c.get("block_size", 1))

            result["concurrency_limit"] = int(
                worker.get("frontend_concurrency_limit", 0)
            )
            result["available_concurrency"] = int(
                worker.get("frontend_available_concurrency", 0)
            )
        except Exception as e:
            logging.warning(f"Failed to query engine status: {e}")
        return result

    # ---- tokenizer ----

    def _load_tokenizer(self):
        from rtp_llm.test.perf_test.test_util import _load_tokenizer

        self._tokenizer = _load_tokenizer(self._tokenizer_path)

    # ---- analysis ----

    def _analyze(
        self,
        records: List[_RequestRecord],
        wall_time: float,
        status_before: Dict[str, Any],
        status_after: Dict[str, Any],
    ) -> OfflineMetrics:
        metrics = OfflineMetrics()
        metrics.total_wall_time_s = wall_time
        metrics.total_submitted = len(records)

        responses = [r.response for r in records]

        success_aux: List[Dict[str, Any]] = []
        fail_reasons: Dict[str, int] = {}
        for resp in responses:
            if isinstance(resp, Exception):
                reason = f"exception:{type(resp).__name__}"
                fail_reasons[reason] = fail_reasons.get(reason, 0) + 1
                continue
            if isinstance(resp, dict) and "error" not in resp and "error_code" not in resp:
                aux = resp.get("aux_info", {})
                if isinstance(aux, list) and aux:
                    aux = aux[0]
                if aux:
                    success_aux.append(aux)
                else:
                    fail_reasons["empty_aux_info"] = fail_reasons.get("empty_aux_info", 0) + 1
            else:
                error_code = resp.get("error_code", "unknown") if isinstance(resp, dict) else "unknown"
                error_msg = (
                    resp.get("message") or resp.get("error_message") or resp.get("error") or ""
                ) if isinstance(resp, dict) else str(resp)
                reason = f"error_code={error_code}: {str(error_msg)[:120]}"
                fail_reasons[reason] = fail_reasons.get(reason, 0) + 1

        metrics.success_requests = len(success_aux)
        metrics.fail_requests = len(records) - metrics.success_requests

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
        metrics.total_tps = (total_input + total_output) / wall_time if wall_time > 0 else 0

        latencies_s = [t / 1000.0 for t in cost_times]
        latencies_s.sort()
        n = len(latencies_s)
        metrics.avg_request_latency_s = sum(latencies_s) / n
        metrics.p50_request_latency_s = latencies_s[n // 2]
        metrics.p99_request_latency_s = latencies_s[min(int(n * 0.99), n - 1)]

        metrics.avg_wait_time_ms = sum(wait_times) / n
        metrics.max_wait_time_ms = max(wait_times)

        first_token_times = [
            a.get("first_token_cost_time", a.get("first_token_cost_time_us", 0) / 1000.0)
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
        for a in success_aux:
            ct = a.get("cost_time", 0.0)
            ft = a.get("first_token_cost_time", a.get("first_token_cost_time_us", 0) / 1000.0)
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

        metrics.kv_cache_total_blocks = status_before.get("total_kv_cache", 0)
        metrics.kv_cache_block_size = status_before.get("block_size", 1)
        metrics.kv_cache_available_before = status_before.get("available_kv_cache", 0)
        metrics.kv_cache_available_after = status_after.get("available_kv_cache", 0)
        if metrics.kv_cache_total_blocks > 0:
            min_available = min(
                metrics.kv_cache_available_before, metrics.kv_cache_available_after
            )
            metrics.kv_cache_peak_usage_ratio = (
                1.0 - min_available / metrics.kv_cache_total_blocks
            )

        metrics.concurrency_limit = status_before.get("concurrency_limit", 0)
        metrics.available_concurrency_before = status_before.get(
            "available_concurrency", 0
        )

        if self._status_samples:
            kv_utils = [s["kv_utilization"] for s in self._status_samples if "kv_utilization" in s]
            conc_utils = [s["concurrency_utilization"] for s in self._status_samples if "concurrency_utilization" in s]
            if kv_utils:
                metrics.kv_cache_avg_utilization = sum(kv_utils) / len(kv_utils)
                metrics.kv_cache_max_utilization = max(kv_utils)
                metrics.kv_cache_min_utilization = min(kv_utils)
            if conc_utils:
                metrics.concurrency_avg_utilization = sum(conc_utils) / len(conc_utils)
                metrics.concurrency_max_utilization = max(conc_utils)

        return metrics

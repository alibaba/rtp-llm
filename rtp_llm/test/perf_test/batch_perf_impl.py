import hashlib
import json
import logging
import math
import os
import time
from concurrent.futures import Future, ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Union

import requests

from rtp_llm.test.perf_test.dataclass import (
    E2ECohortMetrics,
    E2ERequestMetrics,
    ResponseInfo,
    TestResultMetrics,
    analyze_results,
)
from rtp_llm.utils.util import check_with_info


def _effective_profile_steps(is_decode: bool, decode_test_length: int) -> int:
    # Prefill has a single model-forward step; requesting more steps leaves the
    # profiler armed and prevents trace export before the test server exits.
    profile_steps = int(os.environ.get("PERF_PROFILE_NUM_STEPS", "3"))
    return min(decode_test_length, profile_steps) if is_decode else 1


def _curl_server_single_worker(
    i: int,
    base_port: int,
    input_query: str,
    is_decode: bool,
    decode_test_length: int,
    wait_time: int,
    profile: bool = False,
    generate_config: Optional[Dict[str, Any]] = None,
    profile_trace_name: str = "",
) -> ResponseInfo:
    req = {
        "prompt": input_query,
        "generate_config": {
            "max_new_tokens": decode_test_length if is_decode else 1,
            "min_new_tokens": decode_test_length if is_decode else 1,
            "force_sp_accept": True,
        },
    }

    if generate_config is not None:
        req["generate_config"].update(generate_config)
        if "top_k" in generate_config:
            req["top_k"] = generate_config["top_k"]
        if "top_p" in generate_config:
            req["top_p"] = generate_config["top_p"]

    if os.environ.get(
        "PERF_USE_BATCH_DECODE_SCHEDULER", "1"
    ) == "0" and "force_sp_accept" not in (generate_config or {}):
        req["generate_config"]["force_sp_accept"] = False

    if "top_k" not in req:
        req["top_k"] = 1

    profile_step = _effective_profile_steps(is_decode, decode_test_length)
    if profile:
        req["gen_timeline"] = True
        req["generate_config"]["gen_timeline"] = True
        req["profile_step"] = profile_step
        req["generate_config"]["profile_step"] = profile_step
        if profile_trace_name:
            req["profile_trace_name"] = profile_trace_name
            req["generate_config"]["profile_trace_name"] = profile_trace_name
    measure_e2e = os.environ.get("PERF_MEASURE_E2E", "0") == "1"
    started = time.perf_counter() if measure_e2e else 0.0
    result = ResponseInfo({}, False)
    try:
        response = requests.post(
            f"http://127.0.0.1:{base_port}", json=req, timeout=wait_time
        )
        if response.status_code != 200:
            logging.warning(f"request failed: {response.content}")
        else:
            logging.debug(response.text)
            response_data = response.json()
            result = ResponseInfo(response_data)
            if measure_e2e and (
                not isinstance(result.output_len, int)
                or isinstance(result.output_len, bool)
                or result.output_len <= 0
            ):
                raise ValueError(
                    "E2E response must report a positive output token count"
                )
            if measure_e2e and response_data.get("response") is not None:
                result.response_sha256 = hashlib.sha256(
                    json.dumps(
                        response_data["response"],
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode("utf-8")
                ).hexdigest()
    except Exception as e:
        logging.warning(f" request exception: {e}")
        result = ResponseInfo({}, False)
    if measure_e2e:
        result.wall_seconds = time.perf_counter() - started
        result.input_query_sha256 = hashlib.sha256(
            input_query.encode("utf-8")
        ).hexdigest()
    return result


def _curl_server_batch_worker(
    request_indices: List[int],
    base_port: int,
    input_queries: List[str],
    is_decode: bool,
    decode_test_length: int,
    wait_time: int,
    profile: bool = False,
    generate_config: Optional[Dict[str, Any]] = None,
    profile_trace_name: str = "",
) -> List[ResponseInfo]:
    """Concurrently send requests, each with its own query string."""
    with ThreadPoolExecutor(max_workers=len(request_indices)) as executor:
        futures = []
        for idx, i in enumerate(request_indices):
            future = executor.submit(
                _curl_server_single_worker,
                i,
                base_port,
                input_queries[idx],
                is_decode,
                decode_test_length,
                wait_time,
                profile,
                generate_config,
                profile_trace_name,
            )
            futures.append(future)
        return [f.result() for f in futures]


class BatchPerfImpl(object):
    def __init__(
        self,
        base_port: int,
        dp_size: int,
        batch_size: int,
        query: Union[str, List[str]],
        is_decode: bool = True,
        wait_time: int = 100,
        decode_test_length: int = 10,
        profile: bool = True,
        generate_config: Optional[Dict[str, Any]] = None,
        profile_trace_name: str = "",
        warmup_runs: Optional[int] = None,
        measure_runs: Optional[int] = None,
        profile_runs: Optional[int] = None,
        reuse_cache_seed_query: Optional[Union[str, List[str]]] = None,
        query_variants: Optional[List[Union[str, List[str]]]] = None,
        target_reuse_len: int = 0,
        expected_input_len: Optional[int] = None,
    ):
        self.base_port = base_port
        self.dp_size = dp_size
        self.batch_size = batch_size
        self.input_queries = self._normalize_queries(query)
        self.is_decode = is_decode
        self.max_requests_per_process = 128
        self.num_processes = max(
            1,
            (batch_size + self.max_requests_per_process - 1)
            // self.max_requests_per_process,
        )
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        self.executor = ProcessPoolExecutor(max_workers=self.num_processes)
        self.wait_time = wait_time
        self.decode_test_length = decode_test_length
        self.profile = profile
        self.generate_config = generate_config or {}
        self.profile_trace_name = profile_trace_name
        self.validation_errors: List[str] = []
        self.expected_input_len = expected_input_len
        self.query_variants = (
            [self._normalize_queries(q) for q in query_variants]
            if query_variants
            else []
        )
        self.query_variant_index = 0
        self.reuse_cache_seed_queries = (
            self._normalize_seed_queries(reuse_cache_seed_query)
            if reuse_cache_seed_query
            else []
        )
        self.target_reuse_len = target_reuse_len
        self.warmup_runs = (
            int(os.environ.get("PERF_FORMAL_WARMUP_RUNS", "1"))
            if warmup_runs is None
            else int(warmup_runs)
        )
        self.measure_runs = (
            int(os.environ.get("PERF_MEASURE_RUNS", "1"))
            if measure_runs is None
            else int(measure_runs)
        )
        self.profile_runs = (
            int(os.environ.get("PERF_PROFILE_RUNS", "1" if profile else "0"))
            if profile_runs is None
            else int(profile_runs)
        )

    def _normalize_queries(self, query: Union[str, List[str]]) -> List[str]:
        if isinstance(query, str):
            return [query] * self.batch_size
        assert (
            len(query) == self.batch_size
        ), f"query list length {len(query)} != batch_size {self.batch_size}"
        return query

    def _normalize_seed_queries(self, query: Union[str, List[str]]) -> List[str]:
        if isinstance(query, str):
            return [query]
        assert query, "reuse-cache seed query list must not be empty"
        return query

    def _next_input_queries(self) -> List[str]:
        if not self.query_variants:
            return self.input_queries
        variant = self.query_variants[
            min(self.query_variant_index, len(self.query_variants) - 1)
        ]
        self.query_variant_index += 1
        return variant

    def _seed_reuse_cache(self) -> None:
        if not self.reuse_cache_seed_queries:
            return
        seed_config = dict(self.generate_config)
        seed_config.update({"reuse_cache": True, "enable_device_cache": True})
        logging.info(
            "[PERF_REUSE_CACHE_SEED] trace=%s seed_requests=%d target_reuse_len=%d",
            self.profile_trace_name,
            len(self.reuse_cache_seed_queries),
            self.target_reuse_len,
        )
        responses = _curl_server_batch_worker(
            list(range(len(self.reuse_cache_seed_queries))),
            self.base_port,
            self.reuse_cache_seed_queries,
            self.is_decode,
            self.decode_test_length,
            self.wait_time,
            False,
            seed_config,
            "",
        )
        metric = analyze_results(responses)
        check_with_info(
            metric.success_requests == metric.total_requests,
            "reuse-cache seed failed: "
            f"{metric.success_requests}/{metric.total_requests} succeeded",
        )

    def _validate_reuse_metric(self, metric: TestResultMetrics) -> None:
        if self.target_reuse_len <= 0 or metric.success_requests == 0:
            return
        tolerance = max(64.0, self.target_reuse_len * 0.02)
        delta = abs(metric.avg_reuse_len - self.target_reuse_len)
        check_with_info(
            delta <= tolerance,
            "reuse-cache hit length is outside tolerance: "
            f"target={self.target_reuse_len}, actual_avg={metric.avg_reuse_len:.2f}, "
            f"tolerance={tolerance:.2f}, trace={self.profile_trace_name}",
        )

    def _prearm_profile(self) -> None:
        # Pre-arm via /start_profile with enable_all_rank=true so that all
        # TP/DP ranks profile the upcoming request. Controlled by env
        # PERF_PREARM_PROFILE=1.
        if os.environ.get("PERF_PREARM_PROFILE", "0") != "1":
            return
        try:
            num_steps = _effective_profile_steps(
                self.is_decode, self.decode_test_length
            )
            arm_sleep = float(os.environ.get("PERF_PROFILE_ARM_SLEEP", "2"))
            r = requests.post(
                f"http://127.0.0.1:{self.base_port}/start_profile",
                json={
                    "gen_timeline": True,
                    "trace_name": self.profile_trace_name or "perf_prearm",
                    "start_step": int(os.environ.get("PERF_PROFILE_START_STEP", "0")),
                    "num_steps": num_steps,
                    "enable_all_rank": True,
                },
                timeout=60,
            )
            logging.info(
                f"[PERF_PREARM_PROFILE] num_steps={num_steps} arm_sleep={arm_sleep} "
                f"-> {r.status_code} {r.text[:200]}"
            )
            time.sleep(arm_sleep)
        except Exception as e:
            logging.warning(f"[PERF_PREARM_PROFILE] failed: {e}")

    # warmup (JIT compile), measure timing, profile (optional, torch profiler affects accuracy)
    def _record_decode_round(self, phase, index, responses):
        if os.environ.get("PERF_VALIDATE_DECODE") != "1" or not self.is_decode:
            return
        fields = (
            "success",
            "input_len",
            "output_len",
            "wait_time",
            "total_time",
            "prefill_time",
            "decode_time",
            "decode_time_per_token",
            "reuse_len",
        )
        rows = [{key: getattr(r, key) for key in fields} for r in responses]
        valid = len(rows) == self.batch_size and all(
            r["success"]
            and r["output_len"] == self.decode_test_length
            and (
                self.expected_input_len is None
                or r["input_len"] == self.expected_input_len
            )
            and math.isfinite(r["decode_time_per_token"])
            and r["decode_time_per_token"] > 0
            for r in rows
        )
        if not valid:
            self.validation_errors.append(f"{phase} round {index}: invalid responses")
        metrics = asdict(analyze_results(responses))
        # Keep invalid latency evidence in standards-compliant JSON as well.
        for row in rows + [metrics]:
            for key, value in row.items():
                if isinstance(value, float) and not math.isfinite(value):
                    row[key] = str(value)
        directory = os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR", ".")
        os.makedirs(directory, exist_ok=True)
        with open(os.path.join(directory, "decode_rounds.jsonl"), "a") as f:
            f.write(
                json.dumps(
                    dict(
                        trace=self.profile_trace_name,
                        phase=phase,
                        round=index,
                        expected_requests=self.batch_size,
                        valid=valid,
                        responses=rows,
                        metrics=metrics,
                    ),
                    allow_nan=False,
                )
                + "\n"
            )

    def run(self):
        self._set_concurrency()
        for i in range(self.warmup_runs):
            self._seed_reuse_cache()
            logging.info(
                "[PERF_WARMUP_RUN] %d/%d trace=%s",
                i + 1,
                self.warmup_runs,
                self.profile_trace_name,
            )
            responses = self._curl_server_responses(
                input_queries=self._next_input_queries()
            )
            self._record_decode_round("warmup", i + 1, responses)

        all_measure_responses: List[ResponseInfo] = []
        e2e_cohorts: List[E2ECohortMetrics] = []
        measure_e2e = os.environ.get("PERF_MEASURE_E2E", "0") == "1"
        for i in range(self.measure_runs):
            self._seed_reuse_cache()
            input_queries = self._next_input_queries()
            started = time.perf_counter() if measure_e2e else 0.0
            responses = self._curl_server_responses(input_queries=input_queries)
            if measure_e2e:
                wall_seconds = time.perf_counter() - started
                output_tokens = sum(r.output_len for r in responses if r.success)
                cohort = E2ECohortMetrics(
                    wall_seconds=wall_seconds,
                    output_tokens=output_tokens,
                    output_tokens_per_second=output_tokens / wall_seconds,
                    requests=[
                        E2ERequestMetrics(
                            success=r.success,
                            wall_seconds=r.wall_seconds,
                            input_tokens=r.input_len,
                            output_tokens=r.output_len,
                            input_query_sha256=r.input_query_sha256,
                            response_sha256=r.response_sha256,
                            speculative_verify_rounds=r.speculative_verify_rounds,
                            speculative_accepted_token_num=r.speculative_accepted_token_num,
                            speculative_proposed_draft_tokens=r.speculative_proposed_draft_tokens,
                        )
                        for r in responses
                    ],
                )
                e2e_cohorts.append(cohort)
                logging.info("[PERF_E2E_COHORT] %s", cohort)

            metric = analyze_results(responses)
            self._record_decode_round("formal", i + 1, responses)
            self._validate_reuse_metric(metric)
            logging.info(
                "[PERF_MEASURE_RUN] %d/%d trace=%s success=%d/%d "
                "avg_prefill_ms=%.3f avg_total_ms=%.3f avg_wait_ms=%.3f "
                "avg_reuse_len=%.2f avg_reuse_hit_rate=%.6f",
                i + 1,
                self.measure_runs,
                self.profile_trace_name,
                metric.success_requests,
                metric.total_requests,
                metric.avg_prefill_time,
                metric.avg_total_time,
                metric.avg_wait_time,
                metric.avg_reuse_len,
                metric.avg_reuse_hit_rate,
            )
            all_measure_responses.extend(responses)
        results = analyze_results(all_measure_responses)
        results.e2e_cohorts = e2e_cohorts
        self._validate_reuse_metric(results)

        if self.profile and self.profile_runs > 0:
            for i in range(self.profile_runs):
                self._seed_reuse_cache()
                self._prearm_profile()
                logging.info(
                    "[PERF_PROFILE_RUN] %d/%d trace=%s",
                    i + 1,
                    self.profile_runs,
                    self.profile_trace_name,
                )
                _ = self._curl_server(
                    True,
                    input_queries=self._next_input_queries(),
                )
            time.sleep(int(os.environ.get("PERF_PROFILE_FLUSH_SLEEP", "60")))
        return results

    def _set_concurrency(self):
        if os.environ.get("PERF_USE_BATCH_DECODE_SCHEDULER", "1") == "0":
            return
        check_with_info(
            self.batch_size % self.dp_size == 0,
            f"concurrency {self.batch_size} must be divisible by dp_size {self.dp_size}",
        )
        local_batch_size = self.batch_size // self.dp_size
        payload = {
            "batch_size": local_batch_size,
            "mode": "decode" if self.is_decode else "prefill",
        }
        last_error = None
        for attempt in range(1, 21):
            try:
                response = requests.post(
                    f"http://127.0.0.1:{self.base_port}/update_scheduler_info",
                    json=payload,
                    timeout=60,
                )
                if (
                    response.status_code == 200
                    and response.json().get("status", "ok") == "ok"
                ):
                    return
                last_error = f"{response.text}, {response.status_code}"
            except Exception as e:
                last_error = repr(e)
            logging.warning(
                "failed to set concurrency, retrying (%d/20): %s",
                attempt,
                last_error,
            )
            time.sleep(3)
        raise Exception(f"failed to set concurrency after retries: {last_error}")

    def _curl_server_responses(
        self,
        profile: bool = False,
        input_queries: Optional[List[str]] = None,
    ) -> List[ResponseInfo]:
        effective_queries = input_queries or self.input_queries
        request_batches: List[List[int]] = []
        for i in range(0, self.batch_size, self.max_requests_per_process):
            batch_indices = list(
                range(i, min(i + self.max_requests_per_process, self.batch_size))
            )
            request_batches.append(batch_indices)

        futures: List[Future[List[ResponseInfo]]] = []
        for batch_indices in request_batches:
            batch_queries = [effective_queries[i] for i in batch_indices]
            futures.append(
                self.executor.submit(
                    _curl_server_batch_worker,
                    batch_indices,
                    self.base_port,
                    batch_queries,
                    self.is_decode,
                    self.decode_test_length,
                    self.wait_time,
                    profile,
                    self.generate_config,
                    self.profile_trace_name if profile else "",
                )
            )

        all_responses: List[ResponseInfo] = []
        for future in futures:
            all_responses.extend(future.result())

        if (
            os.environ.get("PERF_REQUIRE_ALL_REQUESTS_SUCCESS", "0") == "1"
            or os.environ.get("PERF_USE_BATCH_DECODE_SCHEDULER", "1") == "0"
        ):
            successes = sum(response.success for response in all_responses)
            check_with_info(
                len(all_responses) == self.batch_size and successes == self.batch_size,
                f"perf requests failed: {successes}/{self.batch_size} succeeded, "
                f"profile={profile}, trace={self.profile_trace_name}",
            )
        return all_responses

    def _curl_server(
        self,
        profile: bool = False,
        input_queries: Optional[List[str]] = None,
    ) -> TestResultMetrics:
        return analyze_results(self._curl_server_responses(profile, input_queries))

    def dump_results(self, results: List[Dict[str, Any]]):
        for result in results:
            logging.debug(json.dumps(result))

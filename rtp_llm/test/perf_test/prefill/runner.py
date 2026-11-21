"""Cache-aware prefill request execution and correctness classification."""

from __future__ import annotations

import array
import copy
import hashlib
import json
import math
import statistics
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, List, Mapping, Tuple

from rtp_llm.test.perf_test.errors import error_message
from rtp_llm.test.perf_test.prefill.measurement import PREFILL_MEASUREMENT_CONTRACT
from rtp_llm.test.perf_test.prefill.prompts import (
    PrefixPromptFactory,
    build_grouped_prompts,
)


def _token_digest(ids) -> bytes:
    """Hash complete token sequences as canonical little-endian signed int32."""
    try:
        packed = array.array("i", ids)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("token IDs must be signed 32-bit integers") from error
    if packed.itemsize != 4:
        raise ValueError("native int array must use 4-byte items")
    if sys.byteorder != "little":
        packed.byteswap()
    return hashlib.sha256(packed).digest()


class PrefillRunner:
    def __init__(
        self,
        client: Any,
        tokenizer: Any,
        *,
        commit_tail_tokens: int,
        max_batch_tokens: int,
        workspace_tokens: int | None = None,
        cache_block_size: int | None = None,
        input_ids_only: bool = False,
        monotonic: Callable[[], float] = time.monotonic,
        warmup_deviation_threshold: float = 0.25,
        warmup_sweep_budget_s: float = 1200.0,
        warmup_sweep_profiles: int = 3,
    ):
        if not 0.0 < float(warmup_deviation_threshold) <= 1.0:
            raise ValueError(
                "warmup_deviation_threshold must be in (0, 1], got "
                f"{warmup_deviation_threshold!r}"
            )
        if float(warmup_sweep_budget_s) <= 0:
            raise ValueError("warmup_sweep_budget_s must be positive")
        if int(warmup_sweep_profiles) < 1:
            raise ValueError("warmup_sweep_profiles must be >= 1")
        self.monotonic = monotonic
        self.client = client
        self.factory = PrefixPromptFactory(tokenizer)
        self.commit_tail_tokens = commit_tail_tokens
        self.max_batch_tokens = max_batch_tokens
        self.workspace_tokens = workspace_tokens
        self.cache_block_size = cache_block_size or commit_tail_tokens
        self.input_ids_only = input_ids_only
        self.warmup_deviation_threshold = float(warmup_deviation_threshold)
        self.warmup_sweep_budget_s = float(warmup_sweep_budget_s)
        self.warmup_sweep_profiles = int(warmup_sweep_profiles)
        self._validated_cases: set[str] = set()
        self._warmed_shapes: set[int] = set()

    @staticmethod
    def _identity(case: Dict[str, Any]) -> Dict[str, Any]:
        return {
            key: copy.deepcopy(value) for key, value in case.items() if key != "case_id"
        }

    @staticmethod
    def _failure(case: Dict[str, Any], status: str, message: str) -> Dict[str, Any]:
        return {
            "case_id": case["case_id"],
            "identity": PrefillRunner._identity(case),
            "status": status,
            "runs": case["measure_runs"],
            "errors": [{"code": status, "message": message}],
            "metrics": None,
        }

    @staticmethod
    def _validate(
        response: Dict[str, Any], *, input_len: int, expected_reuse: int
    ) -> Tuple[str, str] | None:
        if not isinstance(response, dict):
            return "request_failed", "request did not complete successfully"
        if not response.get("success"):
            error = response.get("error")
            if isinstance(error, BaseException):
                error = error_message(error)
            message = (
                error
                if isinstance(error, str) and error.strip()
                else "request did not complete successfully"
            )
            return "request_failed", message
        if response.get("input_len") != input_len or response.get("output_len") != 1:
            return "invalid_shape", "response input/output shape does not match plan"
        if response.get("reuse_len") != expected_reuse:
            return (
                "invalid_reuse",
                f"expected reuse {expected_reuse}, got {response.get('reuse_len')}",
            )
        latency = response.get("latency_ms")
        if (
            isinstance(latency, bool)
            or not isinstance(latency, (int, float))
            or not math.isfinite(latency)
            or latency <= 0
        ):
            return "invalid_timing", "latency_ms must be a finite positive number"
        wait = response.get("wait_time_ms")
        if wait is not None and (
            isinstance(wait, bool)
            or not isinstance(wait, (int, float))
            or not math.isfinite(wait)
            or wait < 0
        ):
            return "invalid_timing", "wait_time_ms must be finite and non-negative"
        return None

    def _validation_key(self, case: Dict[str, Any]) -> str:
        data = json.dumps(
            (case, self.commit_tail_tokens, self.cache_block_size, self.input_ids_only),
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        return hashlib.sha256(data).hexdigest()

    def preflight(self, cases):
        """Reject sampled prompts that cannot remain independent in token space."""

        seen = set()
        blocks = {}
        round_blocks = set()
        validated = set()
        for case in cases:
            if "sample_index" not in case:
                continue
            try:
                run_count = int(case["measure_runs"]) + 1
                members = build_grouped_prompts(self.factory, case, run_count)
                for index, member in enumerate(members):
                    owner = (case["case_id"], index)
                    prompts = member["prompts"]
                    cache_len = member["cache_len"]
                    if (
                        len(prompts.prefix_ids) != cache_len
                        or len(prompts.run_ids) != run_count
                        or len(prompts.run_texts) != run_count
                    ):
                        raise ValueError("prompt shape does not match plan")
                    if member["cache_len"]:
                        seed_text, seed_ids = self.factory.build_seed_and_ids(
                            member["prefix_id"],
                            prompts,
                            member["cache_len"],
                            self.commit_tail_tokens,
                        )
                        if (
                            len(seed_ids) != cache_len + self.commit_tail_tokens
                            or seed_ids[:cache_len] != prompts.prefix_ids
                        ):
                            raise ValueError("seed IDs do not preserve planned prefix")
                        if (
                            not self.input_ids_only
                            and list(self.factory.tokenizer.encode(seed_text))
                            != seed_ids
                        ):
                            raise ValueError("seed text and input IDs differ")
                        boundary = member["cache_len"] + self.cache_block_size
                        if len(seed_ids) >= boundary:
                            round_blocks.add(
                                (owner, _token_digest(seed_ids[:boundary]))
                            )
                    for text, ids in zip(prompts.run_texts, prompts.run_ids):
                        if (
                            not self.input_ids_only
                            and list(self.factory.tokenizer.encode(text)) != ids
                        ):
                            raise ValueError("text and input IDs differ")
                        if ids[:cache_len] != prompts.prefix_ids:
                            raise ValueError(
                                "request IDs do not preserve planned prefix"
                            )
                        key = _token_digest(ids)
                        if len(ids) != member["input_len"] or key in seen:
                            raise ValueError("sampled prompts are not independent")
                        seen.add(key)
                        block = self.cache_block_size
                        if len(ids) >= block:
                            first = _token_digest(ids[:block])
                            if first in blocks and blocks[first] != owner:
                                raise ValueError(
                                    "sampled cache blocks are not independent"
                                )
                            blocks[first] = owner
                        boundary = member["cache_len"] + block
                        if len(ids) >= boundary:
                            round_key = (owner, _token_digest(ids[:boundary]))
                            if round_key in round_blocks:
                                raise ValueError("sampled rounds are not independent")
                            round_blocks.add(round_key)
                validated.add(self._validation_key(case))
            except ValueError as error:
                raise ValueError(
                    f"traffic prompt preflight failed for {case['case_id']}: {error}; "
                    "increase min_input_tokens or choose a compatible tokenizer"
                ) from error
        # A failed global check must not authorize partially validated cases.
        self._validated_cases.update(validated)

    def _seed_batches(self, members: List[Dict[str, Any]]) -> List[List[int]]:
        batches = []
        batch = []
        tokens = padded_tokens = 0
        block = self.cache_block_size
        for member, item in enumerate(members):
            if not item["cache_len"]:
                continue
            reservation = item["cache_len"] + self.commit_tail_tokens + 1
            padded = (reservation + block - 1) // block * block
            if reservation > self.max_batch_tokens or (
                self.workspace_tokens is not None and padded > self.workspace_tokens
            ):
                raise ValueError(f"seed member {member} exceeds token budget")
            if batch and (
                len(batch) == 8
                or tokens + reservation > self.max_batch_tokens
                or (
                    self.workspace_tokens is not None
                    and padded_tokens + padded > self.workspace_tokens
                )
            ):
                batches.append(batch)
                batch = []
                tokens = padded_tokens = 0
            batch.append(member)
            tokens += reservation
            padded_tokens += padded
        if batch:
            batches.append(batch)
        return batches

    def _seed_batch(
        self, case: Dict[str, Any], members: List[Dict[str, Any]], batch: List[int]
    ) -> Tuple[str, str] | None:
        try:
            requests = []
            for member in batch:
                item = members[member]
                seed_text, seed_ids = self.factory.build_seed_and_ids(
                    item["prefix_id"],
                    item["prompts"],
                    item["cache_len"],
                    self.commit_tail_tokens,
                )
                requests.append(
                    dict(
                        phase="seed",
                        case_id=case["case_id"],
                        request_index=member,
                        run_index=None,
                        prompt=seed_text,
                        input_ids=seed_ids,
                        output_length=1,
                    )
                )
            self.client.configure_scheduler(len(batch), mode="prefill")
            release = threading.Event()
            aborted = False

            def request_seed(request):
                release.wait()
                if aborted:
                    raise RuntimeError("seed batch setup aborted")
                return self.client.request(**request)

            with ThreadPoolExecutor(max_workers=len(batch)) as executor:
                futures = []
                failures = []
                try:
                    for request in requests:
                        futures.append(executor.submit(request_seed, request))
                except Exception as error:
                    aborted = True
                    failures.append(error_message(error))
                finally:
                    # Submit the full batch before any seed enters the scheduler.
                    release.set()
                responses = []
                for future in futures:
                    try:
                        responses.append(future.result())
                    except Exception as error:
                        failures.append(error_message(error))
                if failures:
                    return "request_failed", "; ".join(failures)
            for member, response in zip(batch, responses):
                invalid = self._validate(
                    response,
                    input_len=members[member]["cache_len"] + self.commit_tail_tokens,
                    expected_reuse=0,
                )
                if invalid is not None:
                    return invalid
        except Exception as error:
            return "request_failed", error_message(error)
        return None

    def run_case(self, case: Dict[str, Any]) -> Dict[str, Any]:
        if (
            "sample_index" in case
            and self._validation_key(case) not in self._validated_cases
        ):
            try:
                self.preflight([case])
            except ValueError as error:
                return self._failure(case, "request_failed", error_message(error))
        batch_size = int(case["batch_size"])
        measure_runs = int(case["measure_runs"])
        # The first prompt is a fixed warmup; formal rounds retain their IDs.
        prompt_count = measure_runs + 1
        include_warmup = self._should_warmup(batch_size, measure_runs)
        input_len = int(case["input_len"])
        cache_len = int(case["cache_len"])
        grouped = "request_groups" in case
        contract = (
            "concurrent_grouped_batch_v1" if grouped else PREFILL_MEASUREMENT_CONTRACT
        )
        members = (
            build_grouped_prompts(self.factory, case, prompt_count)
            if grouped
            else [
                {
                    "input_len": input_len,
                    "cache_len": cache_len,
                    "prefix_id": f"{case['case_id']}_r{member}",
                    "prompts": self.factory.build_case_prompts(
                        f"{case['case_id']}_r{member}",
                        input_len,
                        cache_len,
                        prompt_count,
                    ),
                }
                for member in range(batch_size)
            ]
        )
        try:
            seed_batches = self._seed_batches(members)
        except ValueError as error:
            return self._failure(case, "request_failed", error_message(error))
        for batch in seed_batches:
            invalid = self._seed_batch(case, members, batch)
            if invalid is not None:
                return self._failure(case, *invalid)

        try:
            self.client.configure_scheduler(batch_size, mode="prefill")
        except Exception as error:
            return self._failure(case, "request_failed", error_message(error))

        measurements: List[Dict[str, Any]] = []
        round_rates: List[float] = []
        per_run_elapsed_ms: List[float] = []
        compute_length = (
            sum(item["input_len"] - item["cache_len"] for item in members)
            if grouped
            else int(case["compute_len"])
        )
        with ThreadPoolExecutor(max_workers=batch_size) as executor:
            # Skipping the warmup round still executes the same formal prompts
            # (indices 1..measure_runs), keeping measurements comparable.
            prompt_start = 0 if include_warmup else 1
            for prompt_index in range(prompt_start, prompt_count):
                run_index = prompt_index - 1
                warmup = run_index < 0
                release = threading.Event()
                aborted = False

                def request_member(member: int) -> Dict[str, Any]:
                    release.wait()
                    if aborted:
                        raise RuntimeError("round setup aborted")
                    prompts = members[member]["prompts"]
                    return self.client.request(
                        phase="warmup" if warmup else "measure",
                        case_id=case["case_id"],
                        request_index=member,
                        run_index=None if warmup else run_index,
                        prompt=prompts.run_texts[prompt_index],
                        input_ids=prompts.run_ids[prompt_index],
                        output_length=1,
                    )

                futures = []
                setup_failure = None
                setup_status = "request_failed"
                try:
                    for member in range(batch_size):
                        futures.append(executor.submit(request_member, member))
                    if not warmup:
                        setup_status = "invalid_timing"
                        started = self.monotonic()
                except Exception as error:
                    aborted = True
                    setup_failure = (setup_status, error_message(error))
                finally:
                    # Even partially submitted rounds must unblock before shutdown.
                    release.set()
                responses = []
                failures = []
                # Drain every submitted member before classifying the round.
                for future in futures:
                    try:
                        responses.append(future.result())
                    except Exception as error:
                        failures.append(error_message(error))
                if setup_failure is not None:
                    return self._failure(case, *setup_failure)
                elapsed_ms = 0.0 if warmup else (self.monotonic() - started) * 1000.0
                if failures:
                    return self._failure(case, "request_failed", "; ".join(failures))
                if not warmup and (not math.isfinite(elapsed_ms) or elapsed_ms <= 0):
                    return self._failure(
                        case,
                        "invalid_timing",
                        "round elapsed time must be finite and positive",
                    )
                for member, response in enumerate(responses):
                    invalid = self._validate(
                        response,
                        input_len=members[member]["input_len"],
                        expected_reuse=members[member]["cache_len"],
                    )
                    if invalid is not None:
                        return self._failure(case, *invalid)
                    if warmup:
                        continue
                    measurements.append(
                        {
                            "run": run_index,
                            "request_index": member,
                            "latency_ms": response["latency_ms"],
                            "ttft_ms": response.get("ttft_ms", response["latency_ms"]),
                            "wait_time_ms": response.get("wait_time_ms"),
                            "observed_reuse_length": response["reuse_len"],
                            "round_elapsed_ms": elapsed_ms,
                            "measurement_contract": contract,
                            **(
                                {
                                    "input_length": members[member]["input_len"],
                                    "expected_reuse_length": members[member][
                                        "cache_len"
                                    ],
                                    "compute_length": members[member]["input_len"]
                                    - members[member]["cache_len"],
                                    "group_index": members[member]["group_index"],
                                }
                                if grouped
                                else {}
                            ),
                        }
                    )
                if warmup:
                    continue
                per_run_elapsed_ms.append(elapsed_ms)
                round_rates.append(
                    compute_length
                    * (1 if grouped else batch_size)
                    * 1000.0
                    / elapsed_ms
                )

        if include_warmup:
            self._warmed_shapes.add(batch_size)
        warmup_anomaly = None
        if not include_warmup and len(per_run_elapsed_ms) >= 2:
            first_elapsed = per_run_elapsed_ms[0]
            reference = statistics.median(per_run_elapsed_ms[1:])
            slowest = max(first_elapsed, reference)
            deviation = (first_elapsed - reference) / slowest if slowest > 0 else 0.0
            if deviation > self.warmup_deviation_threshold:
                # The first formal round likely absorbed a lazy-compile cost the
                # sweep missed; drop it so the polluted round cannot skew stats.
                warmup_anomaly = {
                    "deviation_ratio": round(deviation, 6),
                    "dropped_runs": [0],
                }
                measurements = [item for item in measurements if item["run"] != 0]
                round_rates = round_rates[1:]
        if not measurements or not round_rates:
            return self._failure(
                case,
                "invalid_timing",
                "all measured rounds were discarded by the warmup deviation check",
            )
        measured_runs_present = sorted({item["run"] for item in measurements})
        latencies = [float(item["latency_ms"]) for item in measurements]
        median_latency = statistics.median(
            [
                max(item["latency_ms"] for item in measurements if item["run"] == run)
                for run in measured_runs_present
            ]
            if grouped
            else latencies
        )
        waits = [item["wait_time_ms"] for item in measurements]
        median_wait = (
            statistics.median(
                [
                    max(
                        item["wait_time_ms"]
                        for item in measurements
                        if item["run"] == run
                    )
                    for run in measured_runs_present
                ]
                if grouped
                else waits
            )
            if all(wait is not None for wait in waits)
            else None
        )
        if grouped:
            input_len = sum(item["input_len"] for item in members)
            cache_len = sum(item["cache_len"] for item in members)
        throughput = statistics.mean(round_rates)
        return {
            "case_id": case["case_id"],
            "identity": self._identity(case),
            "status": "ok",
            "runs": measure_runs,
            "errors": [],
            "metrics": {
                "input_length": input_len,
                "expected_reuse_length": cache_len,
                "observed_reuse_length": cache_len,
                "compute_length": compute_length,
                "latency_ms": median_latency,
                "ttft_ms": median_latency,
                "wait_time_ms": median_wait,
                "throughput_tokens_per_second": throughput,
                "validation": {"shape": True, "reuse": True, "timing": True},
                "seed": int(case["seed"]),
                "prefix_policy": {"kind": case["prefix_policy"]},
                "measurements": measurements,
                "warmup_used": include_warmup,
                **({"warmup_anomaly": warmup_anomaly} if warmup_anomaly else {}),
            },
        }

    def _should_warmup(self, batch_size: int, measure_runs: int) -> bool:
        if measure_runs < 2:
            # Without a second round there is no deviation sentinel, so keep
            # the per-case warmup round to protect the single measurement.
            return True
        return batch_size not in self._warmed_shapes

    def warmup_sweep(
        self,
        cases: List[Dict[str, Any]],
        *,
        budget_s: float | None = None,
        profiles_per_shape: int | None = None,
    ) -> Dict[str, Any]:
        """Pre-warm representative batch/length combinations once per session.

        Executes synthetic cases built from the shard's own shapes with fresh
        prompt material (distinct case IDs, so token sequences never collide
        with real cases) and discards their results. Shapes warmed here let
        ``run_case`` skip the per-case warmup round; the first-round deviation
        sentinel remains the safety net for anything the sweep misses.
        """
        report: Dict[str, Any] = {
            "schema_version": 1,
            "combos": [],
            "warmed_shapes": 0,
            "failures": [],
            "budget_exhausted": False,
        }
        if len(cases) <= 1:
            # A single case warms itself lazily on its first round; a sweep
            # would only duplicate that case's full cost ahead of it.
            return report
        budget = self.warmup_sweep_budget_s if budget_s is None else float(budget_s)
        profiles = (
            self.warmup_sweep_profiles
            if profiles_per_shape is None
            else int(profiles_per_shape)
        )
        deadline = self.monotonic() + budget
        for combo in self._sweep_combinations(cases, profiles_per_shape=profiles):
            if self.monotonic() > deadline:
                report["budget_exhausted"] = True
                break
            began = self.monotonic()
            result = self.run_case(combo["case"])
            duration_s = self.monotonic() - began
            entry = {
                "case_id": combo["case"]["case_id"],
                "batch_size": combo["batch_size"],
                "profile": combo["profile"],
                "status": result.get("status"),
                "duration_s": round(duration_s, 3),
            }
            report["combos"].append(entry)
            if result.get("status") != "ok":
                # A failed combo only costs warmup coverage; the deviation
                # sentinel still guards the real measurements.
                report["failures"].append(entry)
        report["warmed_shapes"] = len(
            {
                entry["batch_size"]
                for entry in report["combos"]
                if entry["status"] == "ok"
            }
        )
        return report

    @staticmethod
    def _case_length_score(case: Dict[str, Any]) -> int:
        if "request_groups" in case:
            return sum(
                int(group["input_len"]) * int(group.get("count", 1))
                for group in case["request_groups"]
            )
        return int(case["input_len"]) * int(case["batch_size"])

    @staticmethod
    def _profile_indices(count: int, profiles: int) -> List[int]:
        if count <= 0:
            return []
        if profiles <= 1 or count == 1:
            return [0]
        step = (count - 1) / (profiles - 1)
        return sorted(
            {min(count - 1, int(round(index * step))) for index in range(profiles)}
        )

    @classmethod
    def _sweep_combinations(
        cls, cases: List[Dict[str, Any]], *, profiles_per_shape: int
    ) -> List[Dict[str, Any]]:
        by_shape: Dict[int, List[Tuple[int, Dict[str, Any]]]] = {}
        for case in cases:
            by_shape.setdefault(int(case["batch_size"]), []).append(
                (cls._case_length_score(case), case)
            )
        combos: List[Dict[str, Any]] = []
        for batch_size in sorted(by_shape):
            entries = sorted(
                by_shape[batch_size],
                key=lambda pair: (pair[0], str(pair[1].get("case_id", ""))),
            )
            for profile, index in enumerate(
                cls._profile_indices(len(entries), profiles_per_shape)
            ):
                combos.append(
                    {
                        "batch_size": batch_size,
                        "profile": profile,
                        "case": cls._make_sweep_case(entries[index][1], profile),
                    }
                )
        return combos

    @staticmethod
    def _make_sweep_case(case: Dict[str, Any], profile: int) -> Dict[str, Any]:
        sweep: Dict[str, Any] = {
            "case_id": f"warmup-sweep-{case['case_id']}-p{profile}",
            "batch_size": case["batch_size"],
            "measure_runs": 1,
            "seed": case.get("seed", 1),
            "prefix_policy": case["prefix_policy"],
        }
        for key in ("input_len", "cache_len", "compute_len"):
            if key in case:
                sweep[key] = case[key]
        if "request_groups" in case:
            sweep["request_groups"] = copy.deepcopy(case["request_groups"])
        return sweep


def resolve_warmup_settings(workload: Mapping[str, Any] | None) -> Dict[str, Any]:
    """Warmup runner settings from the resolved workload config.

    The startup shape sweep is always enabled; only its tunables are
    configurable (``workload.warmup_deviation_threshold``,
    ``workload.warmup_sweep_budget_s``, ``workload.warmup_sweep_profiles``).
    """
    workload = workload or {}

    def resolve(key: str, default):
        value = workload.get(key)
        return default if value is None else value

    return {
        "warmup_deviation_threshold": float(
            resolve("warmup_deviation_threshold", 0.25)
        ),
        "warmup_sweep_budget_s": float(resolve("warmup_sweep_budget_s", 1200.0)),
        "warmup_sweep_profiles": int(resolve("warmup_sweep_profiles", 3)),
    }

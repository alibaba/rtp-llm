"""Cache-aware prefill request execution and correctness classification."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import statistics
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, List, Tuple

from rtp_llm.test.perf_test.prefill.measurement import PREFILL_MEASUREMENT_CONTRACT
from rtp_llm.test.perf_test.prefill.prompts import (
    PrefixPromptFactory,
    build_grouped_prompts,
)


class PrefillRunner:
    def __init__(
        self,
        client: Any,
        tokenizer: Any,
        *,
        commit_tail_tokens: int,
        cache_block_size: int | None = None,
        monotonic: Callable[[], float] = time.monotonic,
    ):
        self.monotonic = monotonic
        self.client = client
        self.factory = PrefixPromptFactory(tokenizer)
        self.commit_tail_tokens = commit_tail_tokens
        self.cache_block_size = cache_block_size or commit_tail_tokens

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
        if not isinstance(response, dict) or not response.get("success"):
            return "request_failed", "request did not complete successfully"
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
        return None

    def preflight(self, cases):
        """Reject sampled prompts that cannot remain independent in token space."""

        def digest(ids):
            data = json.dumps(ids, separators=(",", ":")).encode()
            return hashlib.sha256(data).digest()

        seen = set()
        blocks = {}
        round_blocks = set()
        for case in cases:
            if "sample_index" not in case:
                continue
            try:
                members = build_grouped_prompts(
                    self.factory, case, int(case["measure_runs"])
                )
                for index, member in enumerate(members):
                    owner = (case["case_id"], index)
                    prompts = member["prompts"]
                    if member["cache_len"]:
                        seed_text, seed_ids = self.factory.make_seed_and_ids(
                            member["prefix_id"],
                            prompts.seed_text,
                            member["cache_len"],
                            self.commit_tail_tokens,
                        )
                        if list(self.factory.tokenizer.encode(seed_text)) != seed_ids:
                            raise ValueError("seed text and input IDs differ")
                        boundary = member["cache_len"] + self.cache_block_size
                        if len(seed_ids) >= boundary:
                            round_blocks.add((owner, digest(seed_ids[:boundary])))
                    for text, ids in zip(prompts.run_texts, prompts.run_ids):
                        if list(self.factory.tokenizer.encode(text)) != ids:
                            raise ValueError("text and input IDs differ")
                        key = digest(ids)
                        if len(ids) != member["input_len"] or key in seen:
                            raise ValueError("sampled prompts are not independent")
                        seen.add(key)
                        block = self.cache_block_size
                        if len(ids) >= block:
                            first = digest(ids[:block])
                            if first in blocks and blocks[first] != owner:
                                raise ValueError(
                                    "sampled cache blocks are not independent"
                                )
                            blocks[first] = owner
                        boundary = member["cache_len"] + block
                        if len(ids) >= boundary:
                            round_key = (owner, digest(ids[:boundary]))
                            if round_key in round_blocks:
                                raise ValueError("sampled rounds are not independent")
                            round_blocks.add(round_key)
            except ValueError as error:
                raise ValueError(
                    f"traffic prompt preflight failed for {case['case_id']}: {error}; "
                    "increase min_input_tokens or choose a compatible tokenizer"
                ) from error

    def run_case(self, case: Dict[str, Any]) -> Dict[str, Any]:
        if "sample_index" in case:
            try:
                self.preflight([case])
            except ValueError as error:
                return self._failure(case, "request_failed", str(error))
        batch_size = int(case["batch_size"])
        measure_runs = int(case["measure_runs"])
        input_len = int(case["input_len"])
        cache_len = int(case["cache_len"])
        grouped = "request_groups" in case
        contract = (
            "concurrent_grouped_batch_v1" if grouped else PREFILL_MEASUREMENT_CONTRACT
        )
        members = (
            build_grouped_prompts(self.factory, case, measure_runs)
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
                        measure_runs,
                    ),
                }
                for member in range(batch_size)
            ]
        )
        if any(item["cache_len"] for item in members):
            try:
                self.client.configure_scheduler(1, mode="prefill")
            except Exception as error:
                return self._failure(case, "request_failed", str(error))
            for member, item in enumerate(members):
                if not item["cache_len"]:
                    continue
                prompts = item["prompts"]
                seed_len = item["cache_len"] + self.commit_tail_tokens
                seed_text, seed_ids = self.factory.make_seed_and_ids(
                    item["prefix_id"],
                    prompts.seed_text,
                    item["cache_len"],
                    self.commit_tail_tokens,
                )
                try:
                    response = self.client.request(
                        phase="seed",
                        case_id=case["case_id"],
                        request_index=member,
                        run_index=None,
                        prompt=seed_text,
                        input_ids=seed_ids,
                        output_length=1,
                    )
                except Exception as error:
                    return self._failure(case, "request_failed", str(error))
                invalid = self._validate(response, input_len=seed_len, expected_reuse=0)
                if invalid is not None:
                    return self._failure(case, *invalid)

        try:
            self.client.configure_scheduler(batch_size, mode="prefill")
        except Exception as error:
            return self._failure(case, "request_failed", str(error))

        measurements: List[Dict[str, Any]] = []
        round_rates: List[float] = []
        compute_length = (
            sum(item["input_len"] - item["cache_len"] for item in members)
            if grouped
            else int(case["compute_len"])
        )
        with ThreadPoolExecutor(max_workers=batch_size) as executor:
            for run_index in range(measure_runs):
                release = threading.Event()
                aborted = False

                def request_member(member: int) -> Dict[str, Any]:
                    release.wait()
                    if aborted:
                        raise RuntimeError("round setup aborted")
                    prompts = members[member]["prompts"]
                    return self.client.request(
                        phase="measure",
                        case_id=case["case_id"],
                        request_index=member,
                        run_index=run_index,
                        prompt=prompts.run_texts[run_index],
                        input_ids=prompts.run_ids[run_index],
                        output_length=1,
                    )

                futures = []
                setup_failure = None
                setup_status = "request_failed"
                try:
                    for member in range(batch_size):
                        futures.append(executor.submit(request_member, member))
                    setup_status = "invalid_timing"
                    started = self.monotonic()
                except Exception as error:
                    aborted = True
                    setup_failure = (setup_status, str(error))
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
                        failures.append(str(error))
                if setup_failure is not None:
                    return self._failure(case, *setup_failure)
                elapsed_ms = (self.monotonic() - started) * 1000.0
                if failures:
                    return self._failure(case, "request_failed", "; ".join(failures))
                if not math.isfinite(elapsed_ms) or elapsed_ms <= 0:
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
                    measurements.append(
                        {
                            "run": run_index,
                            "request_index": member,
                            "latency_ms": response["latency_ms"],
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
                round_rates.append(
                    compute_length
                    * (1 if grouped else batch_size)
                    * 1000.0
                    / elapsed_ms
                )

        latencies = [float(item["latency_ms"]) for item in measurements]
        median_latency = statistics.median(
            [
                max(item["latency_ms"] for item in measurements if item["run"] == run)
                for run in range(measure_runs)
            ]
            if grouped
            else latencies
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
                "throughput_tokens_per_second": throughput,
                "validation": {"shape": True, "reuse": True, "timing": True},
                "seed": int(case["seed"]),
                "prefix_policy": {"kind": case["prefix_policy"]},
                "measurements": measurements,
            },
        }

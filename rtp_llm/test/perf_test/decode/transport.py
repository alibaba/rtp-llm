"""Service-facing HTTP operations for decode measurements and profiling."""

from __future__ import annotations

import copy
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, Mapping, Sequence, Tuple

import requests


class DecodeHttpTransport:
    def __init__(
        self,
        endpoint: str,
        *,
        prompt_for_length: Callable[[int], str],
        post: Callable[..., Any] = requests.post,
        sleep: Callable[[float], None] = time.sleep,
        timeout_seconds: float = 300,
        scheduler_attempts: int = 20,
    ):
        self._endpoint = endpoint.rstrip("/")
        self._prompt_for_length = prompt_for_length
        self._post = post
        self._sleep = sleep
        self._timeout_seconds = timeout_seconds
        self._scheduler_attempts = scheduler_attempts

    def configure_scheduler(
        self, worker_allocations: Sequence[Mapping[str, Any]]
    ) -> None:
        local_batch = max(int(item["local_batch_size"]) for item in worker_allocations)
        last_error = "unknown scheduler error"
        for attempt in range(self._scheduler_attempts):
            try:
                response = self._post(
                    f"{self._endpoint}/update_scheduler_info",
                    json={"batch_size": local_batch, "mode": "decode"},
                    timeout=60,
                )
                body = response.json() if response.status_code == 200 else {}
                if response.status_code == 200 and body.get("status") == "ok":
                    return
                last_error = f"HTTP {response.status_code}: {response.text[:500]}"
            except Exception as error:
                last_error = str(error)
            if attempt + 1 < self._scheduler_attempts:
                self._sleep(3)
        raise RuntimeError(
            f"failed to configure decode scheduler after "
            f"{self._scheduler_attempts} attempts: {last_error}"
        )

    def _request(
        self,
        request_index: int,
        input_length: int,
        output_length: int,
        generate_config: Mapping[str, Any],
    ) -> Dict[str, Any]:
        config = copy.deepcopy(dict(generate_config))
        config.update(
            {
                "max_new_tokens": output_length,
                "min_new_tokens": output_length,
                "force_sp_accept": True,
                "top_k": config.get("top_k", 1),
            }
        )
        try:
            response = self._post(
                self._endpoint,
                json={
                    "prompt": self._prompt_for_length(input_length),
                    "generate_config": config,
                },
                headers={"X-DashScope-LoggingConsent": "none"},
                timeout=self._timeout_seconds,
            )
            if response.status_code != 200:
                raise RuntimeError(
                    f"HTTP {response.status_code}: {response.text[:500]}"
                )
            aux = response.json().get("aux_info") or {}
            actual_input = int(aux.get("input_len", 0))
            actual_output = int(aux.get("output_len", 0))
            wait = float(aux.get("wait_time", 0.0))
            first_token = float(aux.get("first_token_cost_time", 0.0))
            cost = float(aux.get("cost_time", 0.0))
            decode = cost - first_token
            return {
                "request_index": request_index,
                "success": True,
                "input_length": actual_input,
                "output_length": actual_output,
                "latency_ms": {
                    "wait": wait,
                    "prefill": first_token - wait,
                    "decode": decode,
                    "tpot": decode / (actual_output - 1) if actual_output > 1 else 0.0,
                    "total": cost - wait,
                },
            }
        except Exception as error:
            return {
                "request_index": request_index,
                "success": False,
                "input_length": input_length,
                "output_length": output_length,
                "error": str(error),
            }

    def run_round(
        self,
        *,
        input_lengths: Sequence[int],
        output_length: int,
        generate_config: Mapping[str, Any],
    ) -> Tuple[Dict[str, Any], ...]:
        lengths = tuple(input_lengths)
        with ThreadPoolExecutor(max_workers=len(lengths)) as pool:
            futures = [
                pool.submit(
                    self._request,
                    index,
                    input_length,
                    output_length,
                    generate_config,
                )
                for index, input_length in enumerate(lengths)
            ]
            return tuple(future.result() for future in futures)

    def start_profile(
        self, *, trace_name: str, steps: int, enable_all_ranks: bool
    ) -> None:
        response = self._post(
            f"{self._endpoint}/start_profile",
            json={
                "gen_timeline": True,
                "trace_name": trace_name,
                "start_step": 0,
                "num_steps": steps,
                "enable_all_rank": enable_all_ranks,
            },
            timeout=60,
        )
        body = response.json() if response.status_code == 200 else {}
        if response.status_code != 200 or body.get("status") != "ok":
            raise RuntimeError(
                f"failed to start decode profile: HTTP {response.status_code}: "
                f"{response.text[:500]}"
            )

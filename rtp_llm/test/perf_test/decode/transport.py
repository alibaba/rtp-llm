"""Service-facing HTTP operations for decode measurements and profiling."""

from __future__ import annotations

import copy
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple

import requests

from rtp_llm.test.perf_test.decode.scheduler import scheduler_local_batch_size
from rtp_llm.test.perf_test.errors import (
    EngineLifecycleError,
    PlanningError,
    error_message,
)


class DecodeHttpTransport:
    def __init__(
        self,
        endpoint: str,
        *,
        prompt_for_length: Callable[[int], str],
        post: Optional[Callable[..., Any]] = None,
        sleep: Callable[[float], None] = time.sleep,
        timeout_seconds: float = 300,
        scheduler_attempts: int = 20,
        health_check: Optional[Callable[[], None]] = None,
    ):
        self._endpoint = endpoint.rstrip("/")
        self._prompt_for_length = prompt_for_length
        self._post = post
        self._sleep = sleep
        self._timeout_seconds = timeout_seconds
        self._scheduler_attempts = scheduler_attempts
        self._health_check = health_check or (lambda: None)
        self._pool = None
        self._pool_size = 0
        self._worker_state = threading.local()
        self._sessions = []
        self._session_lock = threading.Lock()

    def _sender(self):
        if self._post is not None:
            return self._post
        session = getattr(self._worker_state, "session", None)
        if session is None:
            session = requests.Session()
            self._worker_state.session = session
            with self._session_lock:
                self._sessions.append(session)
        return session.post

    def close(self) -> None:
        if self._pool is not None:
            self._pool.shutdown(wait=True)
            self._pool = None
            self._pool_size = 0
        for session in self._sessions:
            session.close()
        self._sessions.clear()
        self._worker_state = threading.local()

    def _prepare_pool(self, size: int) -> None:
        if size <= 0:
            raise PlanningError("decode round requires at least one request")
        if self._pool_size == size:
            return
        self.close()
        self._pool = ThreadPoolExecutor(max_workers=size)
        release = threading.Event()
        ready = [threading.Event() for _ in range(size)]

        def prepare_worker(event):
            try:
                self._sender()
            finally:
                event.set()
            release.wait()

        futures = []
        try:
            try:
                for event in ready:
                    futures.append(self._pool.submit(prepare_worker, event))
                for event in ready:
                    if not event.wait(timeout=5):
                        raise RuntimeError("decode client workers did not become ready")
            finally:
                release.set()
            for future in futures:
                future.result()
        except BaseException:
            self.close()
            raise
        self._pool_size = size

    def configure_scheduler(
        self, worker_allocations: Sequence[Mapping[str, Any]]
    ) -> None:
        local_batch = scheduler_local_batch_size(
            tuple(item["local_batch_size"] for item in worker_allocations)
        )
        # HTTP requests are assigned by internal request ID, not these planned
        # contiguous ranges. Mixed lengths are safe only when each frontend
        # owns one DP worker (the cluster transport explicitly selects nodes).
        lengths = {
            length
            for allocation in worker_allocations
            for length in allocation.get("input_lengths", ())
        }
        if len(worker_allocations) > 1 and len(lengths) > 1:
            raise PlanningError(
                "mixed decode input lengths with multiple DP workers on one "
                "frontend require explicit DP routing; use one DP worker per "
                "node or uniform input lengths"
            )
        # Thread creation and Session setup precede DecodeBatchExecutor's clock.
        self._prepare_pool(sum(item["local_batch_size"] for item in worker_allocations))
        last_error = "unknown scheduler error"
        for attempt in range(self._scheduler_attempts):
            self._health_check()
            try:
                response = self._sender()(
                    f"{self._endpoint}/update_scheduler_info",
                    json={"batch_size": local_batch, "mode": "decode"},
                    timeout=60,
                )
                body = response.json() if response.status_code == 200 else {}
                if response.status_code == 200 and body.get("status") == "ok":
                    return
                last_error = f"HTTP {response.status_code}: {response.text[:500]}"
            except EngineLifecycleError:
                raise
            except Exception as error:
                last_error = error_message(error)
            self._health_check()
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
        self._health_check()
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
            response = self._sender()(
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
            self._health_check()
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
        except EngineLifecycleError:
            raise
        except Exception as error:
            self._health_check()
            return {
                "request_index": request_index,
                "success": False,
                "input_length": input_length,
                "output_length": output_length,
                "error": error_message(error),
            }

    def run_round(
        self,
        *,
        input_lengths: Sequence[int],
        output_length: int,
        generate_config: Mapping[str, Any],
    ) -> Tuple[Dict[str, Any], ...]:
        lengths = tuple(input_lengths)
        self._prepare_pool(len(lengths))
        futures = [
            self._pool.submit(
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
        self._health_check()
        response = self._sender()(
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

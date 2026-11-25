"""Concrete engine launch and prefill request adapters for unified local runs."""

from __future__ import annotations

import json
import os
import shlex
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Tuple
from urllib.parse import urlsplit

import requests

from rtp_llm.test.perf_test.config_schema import (
    DEFAULT_PREFILL_REQUEST_TIMEOUT_SECONDS,
    DEFAULT_PREFILL_TRANSPORT,
)
from rtp_llm.test.perf_test.errors import EngineLifecycleError, PlanningError
from rtp_llm.test.perf_test.execution.devices import (
    GPU_LOCKED_DEVICES_ENV,
    resolve_allocated_devices,
)
from rtp_llm.test.perf_test.execution.launch_boundary import (
    FORCED_ENGINE_ENVIRONMENT,
    server_environment_exclusions,
    validate_engine_overrides,
)
from rtp_llm.test.perf_test.plan import (
    AOneResourcePlan,
    ExecutionPlan,
    LocalResourcePlan,
    SshResourcePlan,
    thaw,
)


@dataclass(frozen=True)
class EngineLaunchSpec:
    argv: Tuple[str, ...]
    environment: Dict[str, str]
    device_ids: Tuple[int, ...]
    checkpoint: str
    model_type: str
    tokenizer: str


def _option_value(value: Any) -> str:
    if type(value) is bool:
        return "true" if value else "false"
    if value is None:
        return "null"
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    return str(value)


def build_engine_launch_spec(plan: ExecutionPlan) -> EngineLaunchSpec:
    """Project an immutable local plan onto the model server launch contract."""
    if isinstance(plan.resource_plan, LocalResourcePlan):
        device_ids = plan.resource_plan.devices
    elif isinstance(plan.resource_plan, AOneResourcePlan):
        device_ids = tuple(range(plan.resource_plan.gpus_per_shard))
    elif isinstance(plan.resource_plan, SshResourcePlan):
        device_ids = tuple(range(plan.resource_plan.local_world_size))
    else:
        raise PlanningError("engine launch requires an executable resource plan")
    config = thaw(plan.resolved_config)
    model = config["model"]
    engine = config["engine"]
    topology = model["topology"]
    if topology["pp"] != 1:
        raise PlanningError("perf engine launch requires model.topology.pp=1")
    reserved = {
        "model_type": model["type"],
        "checkpoint_path": model["checkpoint"],
        "tokenizer_path": model["tokenizer"],
        "tp_size": topology["tp"],
        "dp_size": topology["dp"],
        "pp_size": topology["pp"],
        "ep_size": topology["ep"],
        "world_size": topology["tp"] * topology["dp"] * topology["pp"],
        "max_seq_len": engine["max_seq_len"],
        "max_batch_tokens_size": engine["max_batch_tokens"],
        "concurrency_limit": engine["concurrency_limit"],
    }
    validate_engine_overrides(engine["args"], engine["env"], reserved)
    options = {**reserved, **engine["args"]}
    argv = tuple(f"--{name}={_option_value(value)}" for name, value in options.items())
    environment = {name: str(value) for name, value in engine["env"].items()}
    environment.update(
        {
            "WORLD_SIZE": str(reserved["world_size"]),
            **FORCED_ENGINE_ENVIRONMENT,
        }
    )
    return EngineLaunchSpec(
        argv=argv,
        environment=environment,
        device_ids=device_ids,
        checkpoint=model["checkpoint"],
        model_type=model["type"],
        tokenizer=model["tokenizer"],
    )


class HttpPrefillClient:
    """Translate model-server HTTP responses to the prefill runner contract."""

    def __init__(
        self,
        endpoint: str,
        *,
        post: Callable[..., Any] = requests.post,
        timeout_seconds: float = DEFAULT_PREFILL_REQUEST_TIMEOUT_SECONDS,
        sleep: Callable[[float], None] = time.sleep,
        scheduler_attempts: int = 20,
        health_check: Optional[Callable[[], None]] = None,
    ):
        if type(scheduler_attempts) is not int or scheduler_attempts <= 0:
            raise ValueError("scheduler_attempts must be a positive integer")
        self._sleep = sleep
        self._scheduler_attempts = scheduler_attempts
        self.endpoint = endpoint.rstrip("/")
        self._post = post
        self._timeout_seconds = timeout_seconds
        self._health_check = health_check or (lambda: None)

    def configure_scheduler(self, batch_size: int, mode: str = "prefill") -> None:
        """Require an explicit acknowledgement before issuing prefill requests."""
        if type(batch_size) is not int or batch_size <= 0:
            raise ValueError("scheduler batch_size must be a positive integer")
        if mode != "prefill":
            raise ValueError(f"unsupported prefill scheduler mode: {mode}")
        url = f"{self.endpoint}/update_scheduler_info"
        payload = {"batch_size": batch_size, "mode": mode}
        last_error = "missing scheduler acknowledgement"
        for attempt in range(self._scheduler_attempts):
            self._health_check()
            try:
                response = self._post(
                    url, json=payload, timeout=min(self._timeout_seconds, 300)
                )
                self._health_check()
                if response.status_code != 200:
                    raise RuntimeError(
                        f"HTTP {response.status_code}: {response.text[:500]}"
                    )
                acknowledgement = response.json()
                if (
                    not isinstance(acknowledgement, dict)
                    or acknowledgement.get("status") != "ok"
                    or "error" in acknowledgement
                ):
                    raise RuntimeError(
                        f"invalid scheduler acknowledgement: {acknowledgement!r}"
                    )
                return
            except EngineLifecycleError:
                raise
            except Exception as error:
                self._health_check()
                last_error = str(error)
            if attempt + 1 < self._scheduler_attempts:
                self._sleep(3)
        raise RuntimeError(
            f"failed to configure prefill scheduler at {url} with {payload} "
            f"after {self._scheduler_attempts} attempts: {last_error}"
        )

    @property
    def port(self) -> Optional[int]:
        """Endpoint port for callers that compose loopback service URLs."""
        return urlsplit(self.endpoint).port

    def request(
        self,
        *,
        phase: str,
        case_id: str,
        request_index: int,
        run_index: Optional[int],
        prompt: str,
        input_ids: Any,
        output_length: int,
    ) -> Dict[str, Any]:
        del phase, input_ids
        body = {
            "prompt": prompt,
            "generate_config": {
                "max_new_tokens": output_length,
                "min_new_tokens": output_length,
                "force_sp_accept": True,
                "top_k": 1,
            },
        }
        self._health_check()
        try:
            response = self._post(
                self.endpoint,
                json=body,
                headers={"X-DashScope-LoggingConsent": "none"},
                timeout=self._timeout_seconds,
            )
            self._health_check()
            if response.status_code != 200:
                return {
                    "success": False,
                    "error": f"HTTP {response.status_code}: {response.text[:500]}",
                }
            payload = response.json()
            aux = payload.get("aux_info") or {}
            return {
                "success": True,
                "input_len": int(aux.get("input_len", 0)),
                "output_len": int(aux.get("output_len", 0)),
                "reuse_len": int(aux.get("reuse_len", 0)),
                "latency_ms": float(aux.get("first_token_cost_time", 0.0)),
                "ttft_ms": float(aux.get("first_token_cost_time", 0.0)),
                "wait_time_ms": (
                    float(aux["wait_time"]) if "wait_time" in aux else None
                ),
            }
        except EngineLifecycleError:
            raise
        except Exception as error:
            self._health_check()
            return {
                "success": False,
                "error": (
                    f"request {case_id}/{request_index}/{run_index} failed: {error}"
                ),
            }


class MagaEngineHandle:
    """Adapt MagaServerManager to the generic EngineSession handle protocol."""

    def __init__(
        self,
        manager: Any,
        *,
        checkpoint: str,
        model_type: str,
        tokenizer: str,
        transport: str = DEFAULT_PREFILL_TRANSPORT,
        timeout_seconds: float = DEFAULT_PREFILL_REQUEST_TIMEOUT_SECONDS,
    ):
        self._manager = manager
        self._checkpoint = checkpoint
        self._model_type = model_type
        self._tokenizer = tokenizer
        self._stopped = False
        if transport not in ("http_text", "dashsc_input_ids"):
            raise ValueError(f"unsupported prefill transport: {transport}")
        self.client = HttpPrefillClient(
            f"http://127.0.0.1:{manager.port}",
            health_check=self._assert_healthy,
            timeout_seconds=timeout_seconds,
        )
        if transport == "dashsc_input_ids":
            from rtp_llm.test.perf_test.prefill.dashsc_transport import (
                DashScPrefillClient,
            )

            self.client = DashScPrefillClient(
                self.client, timeout_seconds=timeout_seconds
            )

    @property
    def log_path(self) -> Optional[str]:
        return self._manager.log_file_path

    @property
    def exit_code(self) -> Optional[int]:
        return self._manager.exit_code

    def wait_ready(self, timeout_seconds: int) -> bool:
        ready = bool(
            self._manager.start_server(
                model_path=self._checkpoint,
                model_type=self._model_type,
                tokenizer_path=self._tokenizer,
                timeout=timeout_seconds,
            )
        )

        if ready and hasattr(self.client, "wait_ready"):
            self.client.wait_ready(timeout_seconds)
        return ready

    def is_alive(self) -> bool:
        return not self._stopped and self._manager.is_alive()

    def _assert_healthy(self) -> None:
        if not self.is_alive():
            raise EngineLifecycleError(
                f"engine exited during run (exit code {self.exit_code}, "
                f"log: {self.log_path})"
            )

    def stop(self) -> None:
        if self._stopped:
            return
        self._stopped = True
        try:
            if hasattr(self.client, "close"):
                self.client.close()
        finally:
            self._manager.stop_server()


def create_maga_handle(
    plan: ExecutionPlan, *, profile_directory=None, nsys_session=None
) -> MagaEngineHandle:
    """Create, but do not start, the real local model server handle."""
    from rtp_llm.test.utils.maga_server_manager import MagaServerManager

    if profile_directory is None and plan.mode == "decode":
        profile_directory = Path(plan.result_directory).resolve() / "engine_traces"
        profile_directory.mkdir(parents=True, exist_ok=True)
    spec = build_engine_launch_spec(plan)
    device_ids = resolve_allocated_devices(spec.device_ids)
    if GPU_LOCKED_DEVICES_ENV in os.environ and "HIP_VISIBLE_DEVICES" in os.environ:
        spec.environment["HIP_VISIBLE_DEVICES"] = ",".join(map(str, device_ids))
    # This launcher owns the entire local world. A submit host may retain rank
    # and gang settings from an earlier SSH run; they must not change this run.
    topology = thaw(plan.resolved_config)["model"]["topology"]
    spec.environment.update(
        {
            "WORLD_RANK": "0",
            "LOCAL_RANK": "0",
            "LOCAL_WORLD_SIZE": spec.environment["WORLD_SIZE"],
            "GANG_CONFIG_STRING": "",
            **{f"{name.upper()}_SIZE": str(value) for name, value in topology.items()},
        }
    )
    extra = {}
    if profile_directory is not None:
        spec.environment["TORCH_CUDA_PROFILER_DIR"] = str(profile_directory)
        # Explicit argv wins over environment in the server argument parser.
        spec = replace(
            spec,
            argv=tuple(
                arg
                for arg in spec.argv
                if not arg.startswith("--torch_cuda_profiler_dir=")
            )
            + ("--torch_cuda_profiler_dir=" + str(profile_directory),),
        )
        settings = thaw(plan.resolved_config)["workload"].get("profiling", {})
        if settings.get("backend") == "nsys":
            extra["process_prefix"] = [
                settings.get("nsys_path") or "nsys",
                "launch",
                "--session-new=" + nsys_session,
                "--show-output=true",
                "--trace=cuda,nvtx,osrt",
                "--wait=all",
            ]
    manager = MagaServerManager(
        **extra,
        env_args=spec.environment,
        device_ids=list(device_ids),
        process_file_name="process.log",
        smoke_args_str=shlex.join(spec.argv),
        inherited_env_exclude=(
            *server_environment_exclusions(),
            "TORCH_CUDA_PROFILER_DIR",
        ),
    )
    return MagaEngineHandle(
        manager,
        checkpoint=spec.checkpoint,
        model_type=spec.model_type,
        tokenizer=spec.tokenizer,
        transport=thaw(plan.resolved_config)["workload"].get(
            "transport", DEFAULT_PREFILL_TRANSPORT
        ),
        timeout_seconds=thaw(plan.resolved_config)["workload"].get(
            "request_timeout_seconds", DEFAULT_PREFILL_REQUEST_TIMEOUT_SECONDS
        ),
    )

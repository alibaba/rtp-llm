"""Warm the real inference path before publishing the startup health gate."""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple


class WarmupError(RuntimeError):
    pass


@dataclass(frozen=True)
class WarmupCase:
    target_tokens: int
    batch_size: int

    @property
    def label(self) -> str:
        return f"{self.target_tokens}x{self.batch_size}"


@dataclass(frozen=True)
class CaseResult:
    max_ttft_ms: float
    input_lengths: Tuple[int, ...]


def parse_cases(value: str) -> List[WarmupCase]:
    cases: List[WarmupCase] = []
    seen = set()
    for raw_item in value.split(","):
        item = raw_item.strip().lower()
        if not item:
            continue
        try:
            target, batch = (int(part.strip()) for part in item.split("x", 1))
        except (TypeError, ValueError) as error:
            raise WarmupError(
                f"invalid warmup case {raw_item!r}; expected TOKENSxBATCH"
            ) from error
        if target <= 0 or batch <= 0:
            raise WarmupError(f"warmup case values must be positive: {raw_item!r}")
        case = WarmupCase(target, batch)
        if case not in seen:
            cases.append(case)
            seen.add(case)
    if not cases:
        raise WarmupError("at least one warmup case is required")
    return cases


class HttpJsonClient:
    def __init__(self, base_url: str, timeout_s: float):
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s

    def get_status(self, path: str, timeout_s: float = 5.0) -> int:
        request = urllib.request.Request(f"{self.base_url}{path}", method="GET")
        try:
            with urllib.request.urlopen(request, timeout=timeout_s) as response:
                return response.status
        except urllib.error.HTTPError as error:
            return error.code
        except (OSError, urllib.error.URLError):
            return 0

    def post_json(self, path: str, payload: Mapping[str, Any]) -> Dict[str, Any]:
        request = urllib.request.Request(
            f"{self.base_url}{path}",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                status = response.status
                body = response.read()
        except urllib.error.HTTPError as error:
            raise WarmupError(f"{path} returned HTTP {error.code}") from error
        except (OSError, urllib.error.URLError) as error:
            raise WarmupError(
                f"{path} request failed: {type(error).__name__}"
            ) from error
        if status != 200:
            raise WarmupError(f"{path} returned HTTP {status}")
        try:
            result = json.loads(body)
        except json.JSONDecodeError as error:
            raise WarmupError(f"{path} returned invalid JSON") from error
        if not isinstance(result, dict):
            raise WarmupError(f"{path} returned a non-object response")
        return result


class PromptBuilder:
    _FILLERS = (
        " atlas",
        " cedar",
        " harbor",
        " lantern",
        " meadow",
        " quartz",
        " summit",
        " willow",
    )

    def __init__(self, client: HttpJsonClient, model: str):
        self.client = client
        self.model = model
        self._token_count_cache: Dict[str, int] = {}

    def _token_count(self, content: str) -> int:
        cached = self._token_count_cache.get(content)
        if cached is not None:
            return cached
        response = self.client.post_json(
            "/tokenize",
            {
                "model": self.model,
                "messages": [{"role": "user", "content": content}],
                "max_tokens": 2,
                "stream": False,
                "chat_template_kwargs": {"enable_thinking": False},
            },
        )
        token_ids = response.get("token_ids")
        if not isinstance(token_ids, list):
            raise WarmupError("/tokenize response has no token_ids list")
        count = len(token_ids)
        self._token_count_cache[content] = count
        return count

    def closest_content(
        self, target_tokens: int, family: int, variant: int, prefix: str = ""
    ) -> Tuple[str, int]:
        filler = self._FILLERS[(family + variant) % len(self._FILLERS)]
        tag = f"[startup-warmup-{family}-{variant}]"
        base = f"{prefix} {tag}" if prefix else tag

        low = 0
        high = max(target_tokens * 2, 32)
        best_content = base
        best_count = self._token_count(best_content)
        while (
            self._token_count(base + filler * high) < target_tokens
            and high < target_tokens * 32
        ):
            high *= 2

        while low <= high:
            middle = (low + high) // 2
            candidate = base + filler * middle
            count = self._token_count(candidate)
            if abs(count - target_tokens) < abs(best_count - target_tokens):
                best_content, best_count = candidate, count
            if count < target_tokens:
                low = middle + 1
            elif count > target_tokens:
                high = middle - 1
            else:
                return candidate, count
        return best_content, best_count


class ServingPathWarmup:
    def __init__(
        self,
        client: HttpJsonClient,
        model: str,
        regular_cases: Sequence[WarmupCase],
        prefix_cases: Sequence[WarmupCase],
    ):
        self.client = client
        self.model = model
        self.regular_cases = regular_cases
        self.prefix_cases = prefix_cases
        self.prompts = PromptBuilder(client, model)

    def wait_until_backend_is_ready(self, deadline_s: float) -> None:
        deadline = time.monotonic() + deadline_s
        while time.monotonic() < deadline:
            # The root route performs the normal backend health check but is not
            # blocked by the startup gate, so the canary cannot race model load.
            if self.client.get_status("/") == 200:
                return
            time.sleep(1)
        raise WarmupError("backend did not become ready before the warmup deadline")

    def run_round(self, round_index: int) -> Dict[str, CaseResult]:
        results: Dict[str, CaseResult] = {}
        family_base = round_index * 1000
        for case_index, case in enumerate(self.regular_cases):
            contents = [
                self.prompts.closest_content(
                    case.target_tokens,
                    family_base + case_index,
                    variant,
                )[0]
                for variant in range(case.batch_size)
            ]
            results[f"no-prefix:{case.label}"] = self._infer(
                contents, reuse_cache=False
            )

        for case_index, case in enumerate(self.prefix_cases):
            family = family_base + 500 + case_index
            prefix_target = max(32, case.target_tokens * 3 // 4)
            prefix, _ = self.prompts.closest_content(prefix_target, family, 0)
            self._infer([prefix], reuse_cache=True)
            contents = [
                self.prompts.closest_content(
                    case.target_tokens,
                    family,
                    variant + 1,
                    prefix=prefix,
                )[0]
                for variant in range(case.batch_size)
            ]
            results[f"high-prefix:{case.label}"] = self._infer(
                contents, reuse_cache=True
            )
        return results

    def _request(self, content: str, reuse_cache: bool, seed: int) -> Dict[str, Any]:
        return {
            "model": self.model,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": 2,
            "temperature": 0.0,
            "seed": seed,
            "stream": False,
            "chat_template_kwargs": {"enable_thinking": False},
            "extra_configs": {
                "top_k": 1,
                "min_new_tokens": 2,
                "ignore_eos": True,
                "reuse_cache": reuse_cache,
            },
        }

    def _infer(self, contents: Sequence[str], reuse_cache: bool) -> CaseResult:
        requests = [
            self._request(content, reuse_cache, index + 1)
            for index, content in enumerate(contents)
        ]
        if len(requests) == 1:
            responses = [self.client.post_json("/v1/chat/completions", requests[0])]
        else:
            # BatchGenerateCall executes locally on the Prefill server in PD
            # mode. Concurrent ordinary requests exercise the production
            # HTTP -> Prefill -> Decode path and create the intended batch.
            with ThreadPoolExecutor(max_workers=len(requests)) as executor:
                futures = [
                    executor.submit(
                        self.client.post_json, "/v1/chat/completions", request
                    )
                    for request in requests
                ]
                responses = [future.result() for future in futures]

        ttfts: List[float] = []
        input_lengths: List[int] = []
        for response in responses:
            if not isinstance(response, dict) or not response.get("choices"):
                raise WarmupError("warmup inference response has no choices")
            aux_info = response.get("aux_info")
            if not isinstance(aux_info, dict):
                raise WarmupError("warmup inference response has no aux_info")
            ttft = aux_info.get("first_token_cost_time")
            input_len = aux_info.get("input_len")
            if not isinstance(ttft, (int, float)) or not isinstance(input_len, int):
                raise WarmupError("warmup aux_info is missing TTFT or input length")
            ttfts.append(float(ttft))
            input_lengths.append(input_len)
        return CaseResult(max(ttfts), tuple(input_lengths))


def jit_cache_directories(env: Mapping[str, str]) -> List[Path]:
    candidates = [
        env.get("DG_JIT_CACHE_DIR"),
        env.get("TRITON_CACHE_DIR"),
        env.get("TORCHINDUCTOR_CACHE_DIR"),
        str(Path.home() / ".deep_gemm"),
        str(Path.home() / ".triton" / "cache"),
        str(Path.home() / ".cache" / "torch" / "inductor"),
    ]
    result: List[Path] = []
    seen = set()
    for value in candidates:
        if not value:
            continue
        path = Path(value).expanduser()
        if path not in seen:
            result.append(path)
            seen.add(path)
    return result


def snapshot_jit_artifacts(directories: Iterable[Path]) -> Set[str]:
    artifacts: Set[str] = set()
    for directory in directories:
        if not directory.is_dir():
            continue
        for path in directory.rglob("*"):
            if path.is_file():
                stat = path.stat()
                artifacts.add(
                    f"{directory}:{path.relative_to(directory)}:{stat.st_size}:{stat.st_mtime_ns}"
                )
    return artifacts


def validate_second_round(
    first: Mapping[str, CaseResult],
    second: Mapping[str, CaseResult],
    max_ratio: float,
    slack_ms: float,
    max_ttft_ms: float,
) -> None:
    if first.keys() != second.keys():
        raise WarmupError("warmup rounds did not execute the same case matrix")
    for name, second_result in second.items():
        first_result = first[name]
        if second_result.max_ttft_ms > max_ttft_ms:
            raise WarmupError(
                f"second-round TTFT is too high for {name}: "
                f"{second_result.max_ttft_ms:.1f}ms > {max_ttft_ms:.1f}ms"
            )
        limit_ms = max(
            first_result.max_ttft_ms * max_ratio, first_result.max_ttft_ms + slack_ms
        )
        if second_result.max_ttft_ms > limit_ms:
            raise WarmupError(
                f"second-round TTFT regressed for {name}: "
                f"{second_result.max_ttft_ms:.1f}ms > {limit_ms:.1f}ms"
            )


def publish_gate(gate_file: Path, summary: Mapping[str, Any]) -> None:
    gate_file.parent.mkdir(parents=True, exist_ok=True)
    temporary = gate_file.with_name(f".{gate_file.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(summary, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, gate_file)


def publish_phase(phase_file: Optional[Path], phase: str) -> None:
    if phase_file is None:
        return
    phase_file.parent.mkdir(parents=True, exist_ok=True)
    temporary = phase_file.with_name(f".{phase_file.name}.{os.getpid()}.tmp")
    temporary.write_text(phase + "\n", encoding="utf-8")
    os.replace(temporary, phase_file)


def compile_event_count(event_file: Optional[Path]) -> int:
    if event_file is None or not event_file.is_file():
        return 0
    with event_file.open(encoding="utf-8") as stream:
        return sum(1 for line in stream if line.strip())


def _log_round(phase: str, round_index: int, results: Mapping[str, CaseResult]) -> None:
    for name, result in results.items():
        lengths = ",".join(str(value) for value in result.input_lengths)
        print(
            f"STARTUP_WARMUP phase={phase} round={round_index} case={name} "
            f"input_lengths={lengths} max_ttft_ms={result.max_ttft_ms:.1f}",
            flush=True,
        )


def main(env: Optional[Mapping[str, str]] = None) -> int:
    config = dict(os.environ if env is None else env)
    gate_value = config.get("RTP_LLM_STARTUP_WARMUP_HEALTH_GATE_FILE", "").strip()
    if not gate_value:
        print("STARTUP_WARMUP phase=DISABLED reason=no_gate_file", flush=True)
        return 0

    phase_value = config.get("RTP_LLM_TRITON_COMPILE_PHASE_FILE", "").strip()
    event_value = config.get("RTP_LLM_TRITON_COMPILE_EVENT_FILE", "").strip()
    phase_file = Path(phase_value) if phase_value else None
    event_file = Path(event_value) if event_value else None

    try:
        regular_cases = parse_cases(
            config.get(
                "RTP_LLM_STARTUP_WARMUP_CASES",
                "64x1,128x1,256x1,1024x1,2048x1,256x4,256x8,2048x4,2048x8",
            )
        )
        prefix_cases = parse_cases(
            config.get(
                "RTP_LLM_STARTUP_WARMUP_PREFIX_CASES", "1024x1,256x8,2048x1,2048x8"
            )
        )
        concurrency_limit = int(config.get("CONCURRENCY_LIMIT", "8"))
        largest_batch = max(case.batch_size for case in regular_cases + prefix_cases)
        if largest_batch > concurrency_limit:
            raise WarmupError(
                f"warmup batch {largest_batch} exceeds CONCURRENCY_LIMIT={concurrency_limit}"
            )

        port = int(config.get("START_PORT", "12233"))
        request_timeout_s = float(
            config.get("RTP_LLM_STARTUP_WARMUP_REQUEST_TIMEOUT_S", "1800")
        )
        startup_timeout_s = float(
            config.get("RTP_LLM_STARTUP_WARMUP_START_TIMEOUT_S", "1800")
        )
        max_ratio = float(config.get("RTP_LLM_STARTUP_WARMUP_MAX_TTFT_RATIO", "1.5"))
        slack_ms = float(config.get("RTP_LLM_STARTUP_WARMUP_TTFT_SLACK_MS", "2000"))
        max_ttft_ms = float(
            config.get("RTP_LLM_STARTUP_WARMUP_MAX_SECOND_ROUND_TTFT_MS", "10000")
        )
        fail_on_new_jit = (
            config.get("RTP_LLM_STARTUP_WARMUP_FAIL_ON_NEW_JIT", "1") == "1"
        )
        model = config.get("RTP_LLM_STARTUP_WARMUP_MODEL", "default")

        client = HttpJsonClient(f"http://127.0.0.1:{port}", request_timeout_s)
        warmup = ServingPathWarmup(client, model, regular_cases, prefix_cases)
        print("STARTUP_WARMUP phase=WAITING", flush=True)
        warmup.wait_until_backend_is_ready(startup_timeout_s)

        cache_dirs = jit_cache_directories(config)
        publish_phase(phase_file, "WARMUP")
        print("STARTUP_WARMUP phase=WARMUP round=1", flush=True)
        first = warmup.run_round(1)
        _log_round("WARMUP", 1, first)
        after_first = snapshot_jit_artifacts(cache_dirs)
        compile_events_after_first = compile_event_count(event_file)

        publish_phase(phase_file, "CANARY")
        print("STARTUP_WARMUP phase=CANARY round=2", flush=True)
        second = warmup.run_round(2)
        _log_round("CANARY", 2, second)
        after_second = snapshot_jit_artifacts(cache_dirs)
        changed_jit_artifacts = after_second - after_first
        second_round_compile_events = max(
            0, compile_event_count(event_file) - compile_events_after_first
        )
        if fail_on_new_jit and (changed_jit_artifacts or second_round_compile_events):
            raise WarmupError(
                "second round produced "
                f"{second_round_compile_events} Triton compile event(s) and "
                f"{len(changed_jit_artifacts)} changed local JIT artifact(s)"
            )
        validate_second_round(first, second, max_ratio, slack_ms, max_ttft_ms)

        gate_file = Path(gate_value)
        publish_phase(phase_file, "SERVING")
        publish_gate(
            gate_file,
            {
                "completed_at": int(time.time()),
                "regular_cases": [case.label for case in regular_cases],
                "prefix_cases": [case.label for case in prefix_cases],
                "second_round_compile_events": second_round_compile_events,
                "second_round_changed_jit_artifacts": len(changed_jit_artifacts),
            },
        )
        print(f"STARTUP_WARMUP phase=SERVING gate={gate_file}", flush=True)
        return 0
    except Exception as error:
        try:
            publish_phase(phase_file, "FAILED")
        except OSError:
            pass
        print(f"STARTUP_WARMUP phase=FAILED error={error}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

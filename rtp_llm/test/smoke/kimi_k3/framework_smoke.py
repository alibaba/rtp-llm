"""Bounded, black-box framework checks shared by K3 and K3.1 PD services."""

import json
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

from text_smoke import Runner, SmokeFailure

SUITE = "kimi-pd-compact-v2"
MAX_INFLIGHT = 4
CHUNK_INPUT_TOKENS = 2 * 65536 + 7
PRESSURE_REQUESTS = 3
VALUE_QUERY = (
    "\n检索 key={key} 的 value，逐字复制。只输出一个 JSON 对象，"
    "仅包含字符串字段 value，不要代码块或解释。"
)
STAGES = {
    "page_before",
    "checkpoint_after",
    "chunked_cold",
    "cache_device",
    "cross_owner_reuse",
    "partial_prefix",
    "cache_pressure",
    "cache_host",
    "mixed_cache",
    "decode_refill",
    "cancel_recovery",
}
CASES = (STAGES - {"cache_pressure", "mixed_cache", "decode_refill"}) | {
    *(f"pressure_{i}" for i in range(PRESSURE_REQUESTS)),
    *(f"decode_{i}" for i in range(MAX_INFLIGHT)),
    "mixed_cached",
    "mixed_partial",
    "mixed_recent",
    "mixed_miss",
}


class FrameworkRunner(Runner):
    def validate(self, case, response, *args, **kwargs):
        row = super().validate(case, response, *args, **kwargs)
        counters = response["aux_info"].get("speculative_accepted_tokens_per_pos")
        if (
            not isinstance(counters, list)
            or len(counters) != 3
            or any(type(value) is not int or value < 0 for value in counters)
            or counters != sorted(counters, reverse=True)
        ):
            raise SmokeFailure(
                f"{case.name}: missing or invalid native MTP acceptance counters"
            )
        row.update(
            mtp_accepted_tokens=sum(counters),
            mtp_accepted_tokens_per_pos=counters,
            mtp_acceptance_source="per-position-counters",
        )
        if case.require_mtp and not sum(counters):
            raise SmokeFailure(
                f"{case.name}: no accepted draft token in native MTP counters"
            )
        return row

    def request_cases(
        self, cases, concurrent, admission_wave_size=None, admission_gap_s=0
    ):
        if not concurrent:
            return [self.request(case) for case in cases]
        # Ordinary HTTP concurrency, without a barrier or grouped admission.
        with ThreadPoolExecutor(max_workers=MAX_INFLIGHT) as pool:
            futures = [pool.submit(self.request, case) for case in cases]
            return [future.result() for future in futures]

    def fixture(self, name, tokens, *, owner=0, value=None):
        value = value or name.upper()
        prompt, ids = self.fit_prompt(
            f"Archive {self.args.namespace}/{name}.\n"
            f"key=cedar; value={value}\nkey=birch; value=MAPLE\n",
            VALUE_QUERY.format(key="cedar"),
            tokens,
            enable_thinking=False,
        )
        return self._orthogonal_answer(
            name,
            prompt,
            value,
            "miss",
            expected_input_len=len(ids),
            decode_owner_rank=owner,
            require_chunk=tokens > self.args.chunk_tokens,
        )

    def stage(self, name, cases, *, concurrent=False):
        if concurrent and len(cases) > MAX_INFLIGHT:
            raise ValueError("compact smoke permits at most four in-flight requests")
        self._required_stage(name, cases, concurrent=concurrent)

    @staticmethod
    def partial_prefix(cached):
        original = VALUE_QUERY.format(key="cedar")
        if not cached.prompt.endswith(original):
            raise ValueError(
                "Cache fixture must end with its single retrieval question"
            )
        return replace(
            cached,
            name="partial_prefix",
            reuse="partial",
            expected_input_len=None,
            prompt=cached.prompt[: -len(original)] + VALUE_QUERY.format(key="birch"),
            expected_json={"value": "MAPLE"},
            decode_owner_rank=1,
        )

    def run_framework(self):
        # Cross both the physical page and CP checkpoint boundaries. The first
        # answer also appends Decode KV across a physical-page edge.
        edge = self.fixture("page_before", self.args.block_size - 1)
        self.stage(
            "page_before", [replace(edge, decode_crossings=(self.args.block_size,))]
        )
        self.stage(
            "checkpoint_after",
            [
                self.fixture(
                    "checkpoint_after",
                    self.reuse_unit_tokens + 1,
                    owner=1,
                )
            ],
        )

        # Three real scheduler chunks with a non-aligned tail exercise Last-5
        # checkpoint borrowing without modifying the runtime or its operators.
        seed = self.fixture("chunked_cold", CHUNK_INPUT_TOKENS, value="CEDAR")
        self.stage("chunked_cold", [seed])
        cached = replace(
            seed,
            name="cache_device",
            reuse="hit",
            require_chunk=False,
            allow_long_history=True,
        )
        self.stage("cache_device", [cached])
        if self.records[-1]["prefill_cache_tiers"]["device"] <= 0:
            raise SmokeFailure("immediate Device cache reuse was not observed")
        self.stage(
            "cross_owner_reuse",
            [
                replace(
                    cached,
                    name="cross_owner_reuse",
                    decode_owner_rank=1,
                )
            ],
        )
        partial = self.partial_prefix(cached)
        self.stage("partial_prefix", [partial])

        # Admit four CP blocks at a time, leaving room for the unaligned
        # tail in the 31 usable Device slots. A whole long prompt would force
        # synchronous DROP before the watermark can demote cached KV to Host.
        # These are sequential requests, not a prescribed scheduler batch.
        pressure = [
            self.fixture(f"pressure_{i}", 4 * self.reuse_unit_tokens + 7, owner=i % 2)
            for i in range(PRESSURE_REQUESTS)
        ]
        self.stage("cache_pressure", pressure)
        self.stage("cache_host", [replace(cached, name="cache_host")])
        if self.records[-1]["prefill_cache_tiers"]["memory"] <= 0:
            raise SmokeFailure(
                "Host cache reuse was not observed after bounded pressure"
            )
        self.stage(
            "mixed_cache",
            [
                replace(cached, name="mixed_cached"),
                replace(partial, name="mixed_partial"),
                replace(
                    pressure[-1],
                    name="mixed_recent",
                    reuse="hit",
                    require_chunk=False,
                    allow_long_history=True,
                ),
                self.fixture("mixed_miss", self.reuse_unit_tokens + 7, owner=1),
            ],
            concurrent=True,
        )

        # Unequal output lengths naturally shrink/refill Decode batches. Do
        # not require simultaneous admission or any exact runtime batch size.
        copies = [
            replace(
                self.record_case(f"decode_{i}", 0, words=2 + i * 2),
                thinking_disabled=True,
                require_mtp_draft=True,
                require_mtp=True,
            )
            for i in range(MAX_INFLIGHT)
        ]
        self.stage("decode_refill", copies, concurrent=True)
        self.cancel_stream()
        self.stage(
            "cancel_recovery",
            [
                replace(
                    copies[0],
                    name="cancel_recovery",
                    reuse="any",
                    decode_owner_rank=1,
                )
            ],
        )

    def cancel_stream(self):
        payload = {
            "model": self.args.model_type,
            "messages": [
                {
                    "role": "user",
                    "content": f"{self.args.namespace}/cancel。按顺序输出从0到999的整数，不要省略。",
                }
            ],
            "temperature": 0,
            "top_k": 1,
            "top_p": 0.95,
            "enable_thinking": False,
            "max_tokens": 1024,
            "stream": True,
            "extra_configs": {"role_addrs": [self.decode_role_addrs[0]]},
        }
        request = urllib.request.Request(
            self.endpoint,
            data=json.dumps(payload, ensure_ascii=False).encode(),
            headers={"Content-Type": "application/json"},
        )
        observed = None
        with self.opener.open(request, timeout=min(self.args.timeout, 300)) as response:
            for line in response:
                if not line.startswith(b"data: ") or line.strip() == b"data: [DONE]":
                    continue
                event = json.loads(line[6:])
                if event.get("error") or event.get("error_code"):
                    raise SmokeFailure(f"stream cancellation request failed: {event}")
                for choice in event.get("choices", []):
                    if choice.get("finish_reason") is not None:
                        raise SmokeFailure(
                            "cancellation request finished before disconnect"
                        )
                    if (choice.get("delta") or {}).get("content"):
                        observed = event
                        break
                if observed is not None:
                    break
        if observed is None:
            raise SmokeFailure("cancellation request produced no streaming content")
        self.cancellation = dict(
            request=payload, first_content_event=observed, client_disconnected=True
        )
        (self.args.output.parent / "cancellation.json").write_text(
            json.dumps(self.cancellation, ensure_ascii=False, indent=2) + "\n"
        )

    def save(self, passed, error=None):
        result = dict(
            suite=SUITE,
            model_type=self.args.model_type,
            passed=passed,
            error=error,
            namespace=self.args.namespace,
            max_inflight=MAX_INFLIGHT,
            block_size=self.args.block_size,
            reuse_unit_tokens=self.reuse_unit_tokens,
            chunk_tokens=self.args.chunk_tokens,
            decode_role_addrs=self.decode_role_addrs,
            cases=self.records,
            stages=self.stages,
            failures=self.failures,
            skipped_cases=self.skipped_cases,
            cancellation=getattr(self, "cancellation", None),
            protocol_requests=getattr(self, "protocol_results", []),
            formal_request_retries=0,
            kernel_paths_verified=False,
            elapsed_s=round(time.time() - self.started_at, 3),
        )
        self.args.output.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n"
        )

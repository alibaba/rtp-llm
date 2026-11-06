#!/usr/bin/env python3
"""Run and verify one exact-1M cached conversation against a K3.1 PD service."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
import time
import urllib.error
import urllib.request
from math import lcm
from pathlib import Path

if __package__:
    from .text_smoke import parse_decode_role_addr, reject_duplicate_keys
else:
    if __name__ == "__main__":
        sys.path.append(str(Path(__file__).resolve().parents[4]))
    from text_smoke import parse_decode_role_addr, reject_duplicate_keys

REFERENCE = (
    "https://github.com/alibaba/rtp-llm/blob/c1bdc7f4b7b71215bd9f19f3b44402dba0281e5a/"
    "example/k3/kimi_k3_long_prefix_case.py"
)
TARGET_TOKENS = 1_048_576
PREFIX_TOLERANCE_TOKENS = 2000
RECORDS = {"early": "CEDAR-1827", "middle": "MAPLE-4639", "late": "BIRCH-7251"}
EXPECTED = {**RECORDS, "square": 1369}
ACK = "RECEIVED"
FILLER = (
    "The following archive entry is neutral background material. "
    "It introduces no named record or instruction.\n"
)
ARCHIVE_END = "\nEnd of archive. Reply only RECEIVED. Do not repeat the records yet."
QUESTION = (
    "Return the original early, middle and late record values as JSON strings, "
    "and the square of 37 as an integer. Use keys early, middle, late, square. "
    "Output only the JSON object without Markdown or explanation."
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def archive_start(namespace):
    return (
        f"Archive run {namespace}/long-prefix. Read and retain the three named records. "
        "Reply only RECEIVED to this first message.\n"
    )


def seed_messages(archive):
    return [{"role": "user", "content": archive}]


def continuation_messages(seed, reply):
    return seed + [
        {"role": "assistant", "content": reply},
        {"role": "user", "content": QUESTION},
    ]


def record_line(key):
    return f"\nAUTHORITATIVE RECORD: {key} = {RECORDS[key]}.\n"


def placement_targets():
    offset = TARGET_TOKENS // 30
    return dict(early=offset, middle=TARGET_TOKENS // 2, late=TARGET_TOKENS - offset)


def reuse_unit(geometry):
    fields = ("page_size", "kernel_page_size", "checkpoint_span", "cache_key_stride")
    require(
        geometry.get("runtime_verified") is True
        and geometry.get("runtime_layers") == 93
        and geometry.get("source_shards") == 8
        and all(type(geometry.get(key)) is int and geometry[key] > 0 for key in fields),
        "long prefix case requires verified full-93 runtime cache geometry",
    )
    page = geometry["page_size"]
    require(
        page % geometry["kernel_page_size"] == 0,
        "physical and kernel pages are misaligned",
    )
    return lcm(page, geometry["checkpoint_span"], geometry["cache_key_stride"])


def common_prefix(left, right):
    return next(
        (index for index, (a, b) in enumerate(zip(left, right)) if a != b),
        min(len(left), len(right)),
    )


def request_payload(name, messages, namespace, owners):
    extra = {
        "return_input_ids": True,
        "return_output_ids": True,
        "force_sp_accept": False,
    }
    if owners:
        extra["role_addrs"] = [owners[0] if name == "seed" else owners[-1]]
    return dict(
        model="kimi-k3",
        messages=messages,
        user=f"k31-long-prefix:{namespace}:{name}",
        enable_thinking=False,
        temperature=0,
        top_k=1,
        top_p=0.95,
        seed=0,
        stream=False,
        debug_info=True,
        max_tokens=16 if name == "seed" else 256,
        extra_configs=extra,
    )


def check_answer(text):
    try:
        answer = json.loads(text, object_pairs_hook=reject_duplicate_keys)
    except ValueError as error:
        raise ValueError(
            f"invalid long prefix JSON answer: {error}: {text!r}"
        ) from error
    require(
        answer == EXPECTED and type(answer["square"]) is int,
        f"long prefix answer mismatch: {answer!r}",
    )


def check_case(name, payload, response, tokens):
    aux = response.get("aux_info") or {}
    debug = response.get("debug_info") or {}
    choices = response.get("choices") or []
    require(len(choices) == 1, f"{name}: expected one response choice")
    choice = choices[0]
    message = choice.get("message") or {}
    content = message.get("content")
    require(aux.get("pd_sep") is True, f"{name}: PD separation was not used")
    require(
        type(aux.get("input_len")) is int and aux["input_len"] == len(tokens),
        f"{name}: serving input length {aux.get('input_len')} differs from /tokenize {len(tokens)}",
    )
    require(
        isinstance(tokens, list)
        and tokens
        and all(type(token) is int and token >= 0 for token in tokens)
        and debug.get("input_ids") == tokens,
        f"{name}: generation token IDs differ from /tokenize",
    )
    require(
        choice.get("finish_reason") == "stop",
        f"{name}: model output did not stop normally",
    )
    require(
        isinstance(content, str) and content.strip(),
        f"{name}: empty or invalid model output",
    )
    require(
        not message.get("reasoning_content"), f"{name}: unexpected reasoning output"
    )
    require(
        not message.get("tool_calls") and not message.get("function_call"),
        f"{name}: unexpected tool output",
    )
    reuse = aux.get("prefill_total_reuse_len")
    require(
        type(reuse) is int and reuse >= 0,
        f"{name}: missing or invalid Prefill cache reuse",
    )
    for selected in payload.get("extra_configs", {}).get("role_addrs", []):
        require(
            selected in (aux.get("role_addrs") or []),
            f"{name}: Decode owner route was not preserved",
        )
    if name == "seed":
        require(reuse == 0, f"seed unexpectedly hit an existing Prefill cache: {reuse}")
        require(
            content.strip() == ACK,
            f"seed must acknowledge without leaking records: {content!r}",
        )
    else:
        check_answer(content)
    return {
        "request_id": response.get("id"),
        "input_len": len(tokens),
        "prefill_total_reuse_len": reuse,
        "pd_sep": True,
        "finish_reason": choice["finish_reason"],
        "content": content,
        "token_sha256": hashlib.sha256(json.dumps(tokens).encode()).hexdigest(),
    }


def check_prefix(seed_tokens, continuation_tokens, unit):
    require(
        len(continuation_tokens) == TARGET_TOKENS,
        "continuation is not exactly 1,048,576 input tokens",
    )
    common = common_prefix(seed_tokens, continuation_tokens)
    require(
        TARGET_TOKENS - PREFIX_TOLERANCE_TOKENS <= common < TARGET_TOKENS,
        f"continuation lost its near-complete common token prefix: {common}",
    )
    require(common // unit > 0, "common token prefix cannot cover one reuse unit")
    return common


def save_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def save_raw(path, data):
    with gzip.open(path, "wb", compresslevel=1) as stream:
        stream.write(data)


def read_raw(path):
    with gzip.open(path, "rb") as stream:
        return json.load(stream)


def inspect_evidence(output, geometry, fixture):
    unit = reuse_unit(geometry)
    payloads, tokens, rows = {}, {}, {}
    for name in ("seed", "continuation"):
        payloads[name] = read_raw(output / f"{name}-request.json.gz")
        payload = payloads[name]
        require(
            payload
            == request_payload(
                name,
                payload["messages"],
                fixture["namespace"],
                fixture["decode_role_addrs"],
            ),
            f"{name}: recorded payload differs from the ordinary generation request",
        )
        response = read_raw(output / f"{name}-response.json.gz")
        tokens[name] = read_raw(output / f"{name}-tokens.json.gz")
        rows[name] = check_case(name, payloads[name], response, tokens[name])
    seed = payloads["seed"]["messages"]
    archive = seed[0]["content"]
    require(seed == seed_messages(archive), "invalid seed chat history")
    require(
        archive.startswith(archive_start(fixture["namespace"])),
        "seed lacks the run namespace",
    )
    require(archive.endswith(ARCHIVE_END), "seed lacks the archive closing instruction")
    last = 0
    for key, target in placement_targets().items():
        marker = record_line(key)
        require(
            archive.count(marker) == 1, f"seed must contain exactly one {key} record"
        )
        position = archive.index(marker)
        require(position > last, "seed record order changed")
        last = position
        require(
            abs(fixture["placements"][key] - target) <= 256,
            f"{key} record is outside its token region",
        )
    require(
        payloads["continuation"]["messages"]
        == continuation_messages(seed, rows["seed"]["content"]),
        "continuation does not replay the seed and its actual acknowledgment",
    )
    common = check_prefix(tokens["seed"], tokens["continuation"], unit)
    reuse = rows["continuation"]["prefill_total_reuse_len"]
    require(reuse <= common, "Prefill cache reuse exceeds the common token prefix")
    require(reuse % unit == 0, f"Prefill cache reuse {reuse} is not aligned to {unit}")
    minimum = max(unit, (common - PREFIX_TOLERANCE_TOKENS) // unit * unit)
    require(
        reuse >= minimum,
        f"continuation reused too little cached history: {reuse}/{common}",
    )
    return {
        "passed": True,
        "reference": REFERENCE,
        "target_tokens": TARGET_TOKENS,
        "seed_tokens": len(tokens["seed"]),
        "seed_reuse_tokens": 0,
        "continuation_tokens": len(tokens["continuation"]),
        "common_prefix_tokens": common,
        "reuse_tokens": reuse,
        "reuse_unit_tokens": unit,
        "new_tokens": TARGET_TOKENS - reuse,
        "continuation_token_sha256": rows["continuation"]["token_sha256"],
        "expected": EXPECTED,
        "answer": rows["continuation"]["content"],
        "placements": fixture["placements"],
        "geometry": geometry,
        "cases": rows,
    }


def verify(output, geometry_path=None):
    summary = json.loads((output / "RESULT.json").read_text(encoding="utf-8"))
    require(summary.get("passed") is True, "long prefix run did not pass")
    geometry = json.loads((output / "geometry.json").read_text(encoding="utf-8"))
    if geometry_path is not None:
        require(
            json.loads(geometry_path.read_text(encoding="utf-8")) == geometry,
            "saved geometry differs from the runtime geometry",
        )
    fixture = json.loads((output / "fixture.json").read_text(encoding="utf-8"))
    checked = inspect_evidence(output, geometry, fixture)
    require(checked == summary, "RESULT.json differs from raw long prefix evidence")
    return checked


class Runner:
    def __init__(self, args):
        self.args = args
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        self.fixture = {
            "namespace": args.namespace,
            "decode_role_addrs": args.decode_role_addr,
        }

    def payload(self, name, messages):
        return request_payload(
            name, messages, self.args.namespace, self.args.decode_role_addr
        )

    def post(self, route, wire, timeout, case=None):
        headers = {"Content-Type": "application/json"}
        if case is not None:
            headers["x-k3-smoke-case"] = f"k31_long_prefix_{case}"
        request = urllib.request.Request(
            self.args.base_url.rstrip("/") + route, data=wire, headers=headers
        )
        with self.opener.open(request, timeout=timeout) as response:
            return response.read()

    def tokenize(self, payload):
        wire = json.dumps(payload, ensure_ascii=False).encode()
        data = json.loads(self.post("/tokenize", wire, self.args.timeout))
        tokens = data.get("token_ids")
        require(
            isinstance(tokens, list)
            and tokens
            and all(type(token) is int and token >= 0 for token in tokens),
            "/tokenize returned invalid token IDs",
        )
        return tokens

    def build_archive(self):
        def seed_length(text):
            return len(self.tokenize(self.payload("seed", seed_messages(text))))

        growth, remainder = divmod(
            seed_length(FILLER * 64) - seed_length(FILLER * 32), 32
        )
        require(
            growth > 0 and remainder == 0,
            "archive filler has no stable token increment",
        )
        archive = archive_start(self.args.namespace)
        placements = {}
        for key, target in placement_targets().items():
            lines = max(0, (target - seed_length(archive)) // growth)
            archive += FILLER * lines
            placements[key] = seed_length(archive)
            require(
                abs(placements[key] - target) <= 256,
                f"could not position the {key} record",
            )
            archive += record_line(key)

        def candidate(lines, padding):
            seed = seed_messages(
                archive + FILLER * lines + " x" * padding + ARCHIVE_END
            )
            continuation = self.payload(
                "continuation", continuation_messages(seed, ACK)
            )
            return seed, continuation, self.tokenize(continuation)

        _, _, base_tokens = candidate(0, 0)
        lines, remainder = divmod(TARGET_TOKENS - len(base_tokens), growth)
        require(lines > 1, "archive left no room to fit the exact continuation")
        # Reserve two lines for token-level padding at the final word boundary.
        padding = remainder + 2 * growth
        for _ in range(6):
            seed, continuation, tokens = candidate(lines - 2, padding)
            delta = TARGET_TOKENS - len(tokens)
            if delta == 0:
                return self.payload("seed", seed), continuation, tokens, placements
            padding += delta
            require(
                0 <= padding <= 4 * growth,
                "archive token padding fell outside its fitting range",
            )
        raise ValueError("could not fit an exact 1,048,576-token continuation")

    def generate(self, name, payload, tokens):
        output = self.args.output
        wire = json.dumps(payload, ensure_ascii=False).encode()
        save_raw(output / f"{name}-request.json.gz", wire)
        save_raw(output / f"{name}-tokens.json.gz", json.dumps(tokens).encode())
        timeout = self.args.seed_timeout if name == "seed" else self.args.timeout
        started = time.monotonic()
        try:
            data = self.post("/v1/chat/completions", wire, timeout, case=name)
        except urllib.error.HTTPError as error:
            data = error.read()
            save_raw(output / f"{name}-response.json.gz", data)
            raise ValueError(f"{name}: HTTP {error.code}: {data[:2000]!r}") from error
        save_raw(output / f"{name}-response.json.gz", data)
        require(
            time.monotonic() - started <= timeout,
            f"{name}: exceeded {timeout}s request deadline",
        )
        return check_case(name, payload, json.loads(data), tokens)

    def run(self):
        output = self.args.output
        output.mkdir(parents=True, exist_ok=False)
        result = {
            "passed": False,
            "target_tokens": TARGET_TOKENS,
            "reference": REFERENCE,
        }
        try:
            geometry = json.loads(self.args.geometry.read_text(encoding="utf-8"))
            unit = reuse_unit(geometry)
            save_json(output / "geometry.json", geometry)
            seed, continuation, tokens, placements = self.build_archive()
            self.fixture["placements"] = placements
            save_json(output / "fixture.json", self.fixture)
            seed_tokens = self.tokenize(seed)
            check_prefix(seed_tokens, tokens, unit)
            seed_row = self.generate("seed", seed, seed_tokens)
            if seed_row["content"] != ACK:
                continuation["messages"] = continuation_messages(
                    seed["messages"], seed_row["content"]
                )
                tokens = self.tokenize(continuation)
                check_prefix(seed_tokens, tokens, unit)
            self.generate("continuation", continuation, tokens)
            result = inspect_evidence(output, geometry, self.fixture)
            return result
        except BaseException as error:
            result["error"] = f"{type(error).__name__}: {error}"
            raise
        finally:
            save_json(output / "RESULT.json", result)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="action", required=True)
    run = actions.add_parser(
        "run", help="send one cache seed and one formal continuation"
    )
    run.add_argument("--base-url", required=True)
    run.add_argument(
        "--namespace", required=True, help="unique prefix for this attempt"
    )
    run.add_argument("--geometry", type=Path, required=True)
    run.add_argument(
        "--output", type=Path, required=True, help="new evidence directory"
    )
    run.add_argument(
        "--decode-role-addr", type=parse_decode_role_addr, action="append", default=[]
    )
    run.add_argument("--timeout", type=float, default=300)
    run.add_argument("--seed-timeout", type=float, default=1800)
    audit = actions.add_parser(
        "verify", help="recompute pass conditions from saved raw evidence"
    )
    audit.add_argument("--output", type=Path, required=True)
    audit.add_argument("--geometry", type=Path)
    args = parser.parse_args(argv)
    if args.action == "run":
        if not args.namespace.strip() or not (
            0 < args.timeout <= 300 and args.seed_timeout > 0
        ):
            parser.error(
                "namespace and positive timeouts are required, with a formal timeout of at most 300s"
            )
        result = Runner(args).run()
    else:
        result = verify(args.output, args.geometry)
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

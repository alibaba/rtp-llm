"""Validate decode-to-prefill KV cache writeback against live PD services.

Supports four acceptance modes via subcommands:
  - v4-cp: asymmetric V4 CP-sharded PD writeback & reuse equivalence
  - asymmetric-tp: BF16 asymmetric TP opt-out/opt-in & logit handoff equivalence
  - multiturn: consecutive PD conversation turns & first-use accounting
  - billing: concurrent OpenAI chat usage / streaming cached_tokens accounting
"""

import argparse
import csv
import json
import math
import shlex
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


# ---------------------------------------------------------------------------
# Shared validation, cache-status, and HTTP runner helpers
# ---------------------------------------------------------------------------


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def token_row(response, name):
    rows = response.get(name)
    require(isinstance(rows, list) and len(rows) == 1, f"Missing single {name} row")
    require(isinstance(rows[0], list) and rows[0], f"Empty {name}")
    require(all(type(token) is int for token in rows[0]), f"Invalid {name}")
    return rows[0]


def cached_keys(status):
    if "cache_key_list" in status:
        keys = status["cache_key_list"]
        require(isinstance(keys, list), "Invalid cache_key_list")
        # The RPC prefers a nonempty list and otherwise falls back to the map.
        if keys or "cache_keys" not in status:
            return {int(key) for key in keys}
    if "cache_keys" in status:
        return {int(key) for key, present in status["cache_keys"].items() if present}
    require(isinstance(status.get("cached_keys"), list), "Cache status omitted keys")
    return {int(key) for key in status["cached_keys"]}


def cache_block_size(status):
    value = status.get("block_size")
    require(
        type(value) is int
        or (isinstance(value, str) and value.isascii() and value.isdecimal()),
        f"Invalid cache block_size: {value!r}",
    )
    return int(value)


def generation_query(prompt, count, reuse):
    return {
        "prompt": prompt,
        "generate_config": {
            "max_new_tokens": count,
            "min_new_tokens": count,
            "ignore_eos": True,
            "temperature": 0.0,
            "top_k": 1,
            "top_p": 0,
            "reuse_cache": reuse,
            "enable_device_cache": reuse,
            "enable_memory_cache": False,
            "enable_disk_cache": False,
            "enable_remote_cache": False,
            "return_input_ids": True,
            "return_output_ids": True,
            "skip_special_tokens": False,
        },
    }


def validate_generation(response, input_ids, count, pd_sep):
    require(response.get("finished") is True, "Generation did not finish")
    require(token_row(response, "input_ids") == input_ids, "Input tokenization drift")
    require(
        len(token_row(response, "output_ids")) == count, "Unexpected output token count"
    )
    require(isinstance(response.get("response"), str), "Missing response text")
    aux = response.get("aux_info", {})
    require(aux.get("input_len") == len(input_ids), "Incorrect input_len")
    require(aux.get("output_len") == count, "Incorrect output_len")
    require(aux.get("pd_sep") is pd_sep, "PD separation did not match requested path")


def validate_reuse(response, expected, pd_sep, *, expected_first_use=None):
    aux = response.get("aux_info", {})
    fields = (
        ("prefill_total_reuse_len", "prefill_local_reuse_len")
        if pd_sep
        else ("actual_reuse_len", "local_reuse_len")
    )
    for field in fields:
        require(
            aux.get(field) == expected,
            f"{field}: expected {expected}, got {aux.get(field)}",
        )
    if expected == 0 and pd_sep:
        for field in ("decode_total_reuse_len", "decode_local_reuse_len"):
            require(aux.get(field) == 0, f"Cold control reused D cache: {field}")

    require(
        aux.get("actual_reuse_len") == expected,
        f"actual_reuse_len: expected {expected}, got {aux.get('actual_reuse_len')}",
    )
    first_use = aux.get("writeback_first_use_len")
    require(
        type(first_use) is int and 0 <= first_use <= expected,
        f"Invalid writeback_first_use_len: {first_use!r}",
    )
    require(
        aux.get("reuse_len") == expected - first_use,
        f"reuse_len: expected billed reuse {expected - first_use}, got {aux.get('reuse_len')}",
    )
    if expected_first_use is not None:
        require(
            first_use == expected_first_use,
            f"writeback_first_use_len: expected {expected_first_use}, got {first_use}",
        )


def compare_outputs(warm, control):
    require(
        token_row(warm, "output_ids") == token_row(control, "output_ids"),
        "Warm/cold output token IDs differ",
    )
    require(warm["response"] == control["response"], "Warm/cold response text differs")


def wait_for_publication(
    post, cp_unit, timeout, clock=time.monotonic, sleep=time.sleep
):
    require(math.isfinite(timeout) and timeout > 0, "Invalid cache wait timeout")
    deadline = clock() + timeout
    last_status = None
    while clock() < deadline:
        remaining = deadline - clock()
        if remaining <= 0:
            break
        try:
            last_status = post(min(2.0, remaining))
            require(
                cache_block_size(last_status) == cp_unit,
                "P cache block size is not canonical CP unit",
            )
            if cached_keys(last_status):
                return last_status
        except (urllib.error.URLError, TimeoutError) as error:
            last_status = str(error)
        remaining = deadline - clock()
        if remaining > 0:
            sleep(min(0.1, remaining))
    raise AssertionError(f"P did not publish a complete CP unit: {last_status}")


# ---------------------------------------------------------------------------
# Mode 1: V4 CP-sharded PD writeback acceptance
# ---------------------------------------------------------------------------


def validate_topology(live_info, block_size, prefill_cp_size):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--cp_rotate_method", type=str.upper)
    parser.add_argument("--prefill_cp_size", type=int, default=0)
    for name in (
        "tp_size",
        "seq_size_per_block",
        "prefill_cp_kv_cache_sharded",
        "p2p_writeback_enable",
        "load_cache_timeout_ms",
        "reuse_cache",
        "enable_device_cache",
        "enable_memory_cache",
        "enable_disk_cache",
        "enable_remote_cache",
    ):
        parser.add_argument("--" + name, type=int)
    smoke_args = live_info.get("smoke_args")
    require(isinstance(smoke_args, dict), "Missing service launch arguments")
    for role, tp_size in (("prefill", prefill_cp_size), ("decode", 1)):
        require(isinstance(smoke_args.get(role), str), f"Missing {role} arguments")
        arguments, _ = parser.parse_known_args(shlex.split(smoke_args[role]))
        if role == "prefill":
            require(
                arguments.cp_rotate_method
                in {"ALL_GATHER", "ALL_GATHER_WITH_OVERLAP", "ALLTOALL"},
                "Prefill must enable context parallelism",
            )
        else:
            require(
                arguments.cp_rotate_method == "PREFILL_CP",
                "Decode must use PREFILL_CP peer layout",
            )
            require(
                arguments.prefill_cp_size == prefill_cp_size,
                f"Decode must configure --prefill_cp_size {prefill_cp_size}",
            )
        expected = {
            "tp_size": tp_size,
            "seq_size_per_block": block_size,
            "prefill_cp_kv_cache_sharded": 1,
            "p2p_writeback_enable": 1,
            "reuse_cache": 1,
            "enable_device_cache": 1,
            "enable_memory_cache": 0,
            "enable_disk_cache": 0,
            "enable_remote_cache": 0,
        }
        for name, value in expected.items():
            require(
                getattr(arguments, name) == value,
                f"Incorrect {role} {name}: expected {value}",
            )
        require(
            arguments.load_cache_timeout_ms is not None
            and arguments.load_cache_timeout_ms > 0,
            f"Expected positive {role} load_cache_timeout_ms",
        )


def build_source(tokenizer, cp_unit):
    require(cp_unit > 128, "CP unit is too small for the V4 prompt")
    prefix = "<\uff5cbegin\u2581of\u2581sentence\uff5c><\uff5cUser\uff5c>"
    suffix = (
        "\nIgnore the preceding padding. Write a detailed, continuous essay about "
        "how a database processes transactions. Explain each step in complete "
        "sentences and continue for at least one thousand words."
        "<\uff5cAssistant\uff5c></think>"
    )
    target_length = cp_unit - 32
    lower, upper = 0, cp_unit * 2
    require(
        len(tokenizer.encode(prefix + suffix)) <= target_length,
        "Prompt exceeds CP unit",
    )
    while lower < upper:
        middle = (lower + upper + 1) // 2
        prompt = prefix + " cache" * middle + suffix
        if len(tokenizer.encode(prompt)) <= target_length:
            lower = middle
        else:
            upper = middle - 1
    prompt = prefix + " cache" * lower + suffix
    input_ids = tokenizer.encode(prompt)
    output_count = cp_unit + 1 - len(input_ids)
    require(2 <= output_count <= 128, "Cannot reach CP boundary within output budget")
    return prompt, input_ids, output_count


class AcceptanceRun:
    def __init__(self, live_info, artifacts, request_timeout=300):
        self.servers = live_info["servers"]
        self.artifacts = Path(artifacts)
        self.artifacts.mkdir(parents=True, exist_ok=True)
        require(
            math.isfinite(request_timeout) and request_timeout > 0,
            "Invalid request timeout",
        )
        self.request_timeout = request_timeout

    def save(self, name, value):
        (self.artifacts / f"{name}.json").write_text(json.dumps(value, indent=2) + "\n")

    def post(self, role, endpoint, query, timeout=None):
        port = int(self.servers[role]["port"])
        require(0 < port < 65536, "Invalid service port")
        request = urllib.request.Request(
            f"http://127.0.0.1:{port}{endpoint}",
            data=json.dumps(query).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(
            request, timeout=timeout or self.request_timeout
        ) as response:
            result = json.loads(response.read())
        require(not result.get("error_code"), f"Server returned an error: {result}")
        return result

    def generate(self, name, role, prompt, input_ids, count, reuse, pd_sep):
        query = generation_query(prompt, count, reuse)
        self.save(name + "-request", query)
        response = self.post(role, "/", query)
        self.save(name, response)
        validate_generation(response, input_ids, count, pd_sep)
        return response

    def run(self, tokenizer, cp_unit):
        status_query = {"latest_cache_version": -1, "need_cache_keys": True}
        initial = self.post("prefill", "/rtp_llm/cache_status", status_query)
        self.save("initial-cache", initial)
        require(cache_block_size(initial) == cp_unit, "Incorrect P CP configuration")
        require(
            not cached_keys(initial), "Acceptance requires an isolated, empty P cache"
        )
        prompt, input_ids, output_count = build_source(tokenizer, cp_unit)
        source = self.generate(
            "source", "decode", prompt, input_ids, output_count, True, True
        )
        validate_reuse(source, 0, True)
        complete_ids = input_ids + token_row(source, "output_ids")
        require(
            len(input_ids) < cp_unit == len(complete_ids) - 1,
            "Source must finish one sampled token past the new KV-ready CP unit",
        )
        publication = wait_for_publication(
            lambda timeout: self.post(
                "prefill", "/rtp_llm/cache_status", status_query, timeout
            ),
            cp_unit,
            35,
        )
        self.save("published-cache", publication)
        continuation = tokenizer.decode(
            complete_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
        )
        require(
            tokenizer.encode(continuation) == complete_ids,
            "Continuation token round-trip changed",
        )

        warm_p = self.generate(
            "warm-prefill", "prefill", continuation, complete_ids, 1, True, False
        )
        validate_reuse(warm_p, cp_unit, False, expected_first_use=cp_unit)
        warm_pd = self.generate(
            "warm-pd", "decode", continuation, complete_ids, 8, True, True
        )
        validate_reuse(warm_pd, cp_unit, True, expected_first_use=0)
        cold_p = self.generate(
            "cold-prefill", "prefill", continuation, complete_ids, 1, False, False
        )
        validate_reuse(cold_p, 0, False)
        compare_outputs(warm_p, cold_p)
        cold_pd = self.generate(
            "cold-pd", "decode", continuation, complete_ids, 8, False, True
        )
        validate_reuse(cold_pd, 0, True)
        compare_outputs(warm_pd, cold_pd)
        return {
            "passed": True,
            "source_input_tokens": len(input_ids),
            "source_output_tokens": output_count,
            "prefill_reused_tokens": cp_unit,
            "generated_tokens_reused": cp_unit - len(input_ids),
            "checks": [
                "P publication",
                "P-only reuse",
                "PD reuse",
                "P-only cold control",
                "PD cold control",
            ],
        }


# ---------------------------------------------------------------------------
# Mode 2: BF16 asymmetric TP opt-out/opt-in & logit handoff equivalence
# ---------------------------------------------------------------------------


def probabilities(response):
    rows = response.get("logits")
    require(isinstance(rows, list) and len(rows) == 1, "Missing single logits row")
    logits = rows[0]
    require(isinstance(logits, list) and len(logits) > 1, "Missing vocabulary logits")
    require(
        all(type(value) in (int, float) and math.isfinite(value) for value in logits),
        "Non-finite or invalid logits",
    )
    maximum = max(logits)
    weights = [math.exp(value - maximum) for value in logits]
    total = math.fsum(weights)
    output = token_row(response, "output_ids")
    require(len(output) == 2, "Expected two tokens for a real PD decode step")
    require(
        0 <= output[-1] < len(logits) and logits[output[-1]] == maximum,
        "Returned logits do not describe the final greedy token",
    )
    return [weight / total for weight in weights]


def compare_handoff(warm, cold):
    """Compare the first D step only when both paths have identical conditioning.

    BF16 prefill and incremental decode can exchange near-tied greedy winners.
    Apply the smoke comparer's atol=rtol=1e-2 to token probabilities, plus <= 2%
    total variation over the entire vocabulary. Never compare later free-running
    divergent histories.
    """
    require(
        token_row(warm, "input_ids") == token_row(cold, "input_ids"), "Different inputs"
    )
    require(
        token_row(warm, "output_ids")[:-1] == token_row(cold, "output_ids")[:-1],
        "Different conditioning tokens before the compared decode step",
    )
    warm_probs, cold_probs = probabilities(warm), probabilities(cold)
    require(len(warm_probs) == len(cold_probs), "Different vocabulary sizes")
    errors = [
        abs(warm_value - cold_value)
        for warm_value, cold_value in zip(warm_probs, cold_probs)
    ]
    maximum_error = max(errors)
    total_variation = math.fsum(errors) / 2
    maximum_ratio = max(
        error / (0.01 + 0.01 * probability)
        for error, probability in zip(errors, cold_probs)
    )
    require(
        maximum_ratio <= 1,
        f"Token probability error exceeds atol=rtol=0.01: {maximum_ratio}",
    )
    require(total_variation <= 0.02, f"Total variation exceeds 0.02: {total_variation}")
    return {
        "max_probability_error": maximum_error,
        "total_variation": total_variation,
        "max_tolerance_ratio": maximum_ratio,
        "same_greedy_output": warm["output_ids"] == cold["output_ids"],
    }


def generate_probe(run, name, role, prompt, input_ids, reuse):
    pd_sep = role == "decode"
    query = generation_query(prompt, 2, reuse)
    query["generate_config"].update(
        can_use_pd_separation=pd_sep,
        return_logits=True,
        enable_p2p_writeback=False,
    )
    run.save(name + "-request", query)
    response = run.post(role, "/", query)
    run.save(name, response)
    validate_generation(response, input_ids, 2, pd_sep)
    return response


def generate_source(run, name, prompt, input_ids, count, enable_p2p_writeback):
    query = generation_query(prompt, count, True)
    query["generate_config"]["enable_p2p_writeback"] = enable_p2p_writeback
    run.save(name + "-request", query)
    response = run.post("decode", "/", query)
    run.save(name, response)
    validate_generation(response, input_ids, count, True)
    return response


def assert_no_publication(
    post, block_size, timeout, clock=time.monotonic, sleep=time.sleep
):
    require(math.isfinite(timeout) and timeout > 0, "Invalid cache wait timeout")
    deadline = clock() + timeout
    last_status = None
    while clock() < deadline:
        remaining = deadline - clock()
        if remaining <= 0:
            break
        last_status = post(min(2.0, remaining))
        require(
            cache_block_size(last_status) == block_size,
            "P cache block size changed during opt-out check",
        )
        require(
            not cached_keys(last_status),
            "Request-level writeback opt-out still published a P cache block",
        )
        remaining = deadline - clock()
        if remaining > 0:
            sleep(min(0.1, remaining))
    return last_status


def validate_writeback(run, tokenizer, block_size=16):
    status_query = {"latest_cache_version": -1, "need_cache_keys": True}
    initial = run.post("prefill", "/rtp_llm/cache_status", status_query)
    run.save("initial-cache", initial)
    require(cache_block_size(initial) == block_size, "Unexpected cache block size")
    require(not cached_keys(initial), "Acceptance requires an empty P cache")
    opt_out_prompt = "Explain how databases isolate concurrent transactions."
    opt_out_ids = tokenizer.encode(opt_out_prompt)
    require(
        0 < len(opt_out_ids) < block_size,
        "Opt-out source prompt must not publish a P block",
    )
    opt_out_source = generate_source(
        run,
        "writeback-opt-out-source",
        opt_out_prompt,
        opt_out_ids,
        block_size * 3,
        False,
    )
    validate_reuse(opt_out_source, 0, True)
    opt_out_status = assert_no_publication(
        lambda timeout: run.post(
            "prefill", "/rtp_llm/cache_status", status_query, timeout
        ),
        block_size,
        35,
    )
    run.save("writeback-opt-out-cache", opt_out_status)

    prompt = "Write a detailed explanation of database transactions."
    input_ids = tokenizer.encode(prompt)
    require(
        0 < len(input_ids) < block_size, "Source prompt must not publish a P block"
    )
    source = generate_source(run, "source", prompt, input_ids, block_size * 3, True)
    validate_reuse(source, 0, True)
    publication = wait_for_publication(
        lambda timeout: run.post(
            "prefill", "/rtp_llm/cache_status", status_query, timeout
        ),
        block_size,
        35,
    )
    run.save("published-cache", publication)
    complete_ids = input_ids + token_row(source, "output_ids")
    comparisons = []
    for prefix_blocks in (1, 2, 3):
        prefix = complete_ids[: prefix_blocks * block_size + 1]
        require(
            len(prefix) == prefix_blocks * block_size + 1,
            "Source continuation is too short",
        )
        continuation = tokenizer.decode(
            prefix, skip_special_tokens=False, clean_up_tokenization_spaces=False
        )
        require(
            tokenizer.encode(continuation) == prefix,
            "Continuation tokenization changed",
        )
        for role in ("prefill", "decode"):
            label = f"blocks-{prefix_blocks}-{role}"
            cold = generate_probe(
                run, label + "-cold", role, continuation, prefix, False
            )
            validate_reuse(cold, 0, role == "decode")
            previous = None
            for repeat in range(2):
                warm = generate_probe(
                    run, label + f"-warm-{repeat}", role, continuation, prefix, True
                )
                validate_reuse(
                    warm,
                    prefix_blocks * block_size,
                    role == "decode",
                    expected_first_use=(
                        block_size if role == "prefill" and repeat == 0 else 0
                    ),
                )
                metrics = compare_handoff(warm, cold)
                if previous is not None:
                    compare_outputs(warm, previous)
                previous = warm
                comparisons.append({"case": label, "repeat": repeat, **metrics})
    return {
        "passed": True,
        "published_keys": len(cached_keys(publication)),
        "checks": [
            "request-level writeback opt-out suppresses P publication",
            "request-level writeback opt-in publishes P cache",
            "warm/cold whole-vocabulary logits remain equivalent",
        ],
        "comparisons": comparisons,
    }


# ---------------------------------------------------------------------------
# Mode 3: Multi-turn conversation writeback & reuse acceptance
# ---------------------------------------------------------------------------


def complete_prefix(input_length, output_length, cp_unit):
    require(cp_unit > 0 and output_length > 0, "Invalid generation or CP unit")
    return (input_length + output_length - 1) // cp_unit * cp_unit


def expected_reuse(input_length, published_prefix, cp_unit, min_replay_tokens):
    require(cp_unit > 0 and min_replay_tokens > 0, "Invalid cache replay policy")
    replay_limit = max(0, input_length - min_replay_tokens) // cp_unit * cp_unit
    return min(published_prefix, replay_limit)


def common_prefix_length(previous, current):
    return next(
        (
            index
            for index, pair in enumerate(zip(previous, current))
            if pair[0] != pair[1]
        ),
        min(len(previous), len(current)),
    )


def validate_metrics(aux, expected, expected_first_use=None, pd_sep=True):
    fields = (
        "reuse_len",
        "actual_reuse_len",
        "writeback_first_use_len",
        "prefill_total_reuse_len",
        "prefill_local_reuse_len",
        "decode_total_reuse_len",
    )
    for field in fields:
        require(type(aux.get(field)) is int, f"Missing integer metric: {field}")
        require(aux[field] >= 0, f"Negative metric: {field}")
    require(
        aux["actual_reuse_len"] == aux["reuse_len"] + aux["writeback_first_use_len"],
        "actual_reuse_len does not equal reuse_len + writeback_first_use_len",
    )
    require(
        aux["actual_reuse_len"] == expected,
        f"actual_reuse_len: expected {expected}, got {aux['actual_reuse_len']}",
    )
    if pd_sep:
        for field in ("prefill_total_reuse_len", "prefill_local_reuse_len"):
            require(
                aux[field] == expected,
                f"{field}: expected {expected}, got {aux[field]}",
            )
    if expected_first_use is not None:
        require(
            aux["writeback_first_use_len"] == expected_first_use,
            f"writeback_first_use_len: expected {expected_first_use}, "
            f"got {aux['writeback_first_use_len']}",
        )


def validate_cold_control(aux):
    validate_metrics(aux, 0, 0)
    for field in (
        "decode_local_reuse_len",
        "decode_memory_reuse_len",
        "decode_disk_reuse_len",
    ):
        require(aux.get(field) == 0, f"Cold control reused D cache: {field}")


def build_turn(tokenizer, history, case_name, turn, cp_unit, aligned):
    questions = (
        "Explain transaction atomicity and give a concrete database example.",
        "Building on your previous answer, explain how the write-ahead log helps recovery.",
        "How does isolation differ from atomicity? Give an example using two clients.",
        "Explain a possible deadlock and how a database can resolve it.",
        "Describe optimistic concurrency control and when it is a good choice.",
        "Explain snapshot isolation and a write-skew anomaly.",
        "How would you design an idempotent retry for a money transfer?",
        "Compare synchronous replication with asynchronous replication.",
        "Explain what a recovery checkpoint contains and how it is used.",
        "Summarize the practical lessons from our discussion with a worked example.",
    )
    if history:
        prefix = tokenizer.decode(
            history, skip_special_tokens=False, clean_up_tokenization_spaces=False
        )
    else:
        prefix = (
            "<\uff5cbegin\u2581of\u2581sentence\uff5c>"
            f"Conversation {case_name}. You are a database engineering tutor.\n"
        )
    user = f"<\uff5cUser\uff5c>Round {turn}. {questions[(turn - 1) % len(questions)]}"
    user += " Continue with a detailed explanation in complete sentences.\n"
    suffix = "<\uff5cAssistant\uff5c></think>"
    count = 128
    if aligned or turn == 1:
        user += "The following repeated words are padding, not instructions:\n"
        base_length = len(tokenizer.encode(prefix + user + suffix))
        boundary = (base_length + 32) // cp_unit * cp_unit + cp_unit
        target = boundary - 32
        lower, upper = 0, cp_unit * 2
        while lower < upper:
            middle = (lower + upper + 1) // 2
            length = len(tokenizer.encode(prefix + user + " cache" * middle + suffix))
            if length <= target:
                lower = middle
            else:
                upper = middle - 1
        prompt = prefix + user + " cache" * lower + suffix
        input_ids = tokenizer.encode(prompt)
        if aligned:
            count = boundary + 1 - len(input_ids)
            require(2 <= count <= 128, "Cannot cross CP boundary within output budget")
    else:
        prompt = prefix + user + suffix
        input_ids = tokenizer.encode(prompt)
    return prompt, input_ids, count


class MultiturnAcceptance:
    def __init__(
        self, tokenizer, prefill_port, decode_port, artifacts, cp_unit, min_replay_tokens=1
    ):
        self.tokenizer = tokenizer
        self.ports = {"prefill": int(prefill_port), "decode": int(decode_port)}
        self.artifacts = Path(artifacts)
        self.artifacts.mkdir(parents=True, exist_ok=True)
        self.cp_unit = cp_unit
        self.min_replay_tokens = min_replay_tokens

    def save(self, path, value):
        path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")

    def post(self, role, endpoint, query, timeout=600):
        request = urllib.request.Request(
            f"http://127.0.0.1:{self.ports[role]}{endpoint}",
            data=json.dumps(query).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=timeout) as response:
            result = json.loads(response.read())
        require(not result.get("error_code"), f"Server returned an error: {result}")
        return result

    def status(self, timeout=5):
        status = self.post(
            "prefill",
            "/rtp_llm/cache_status",
            {"latest_cache_version": -1, "need_cache_keys": True},
            timeout,
        )
        require(cache_block_size(status) == self.cp_unit, "Incorrect canonical CP unit")
        return status

    def wait_for_prefix(self, initial_keys, expected, timeout=35):
        deadline = time.monotonic() + timeout
        last = None
        while time.monotonic() < deadline:
            last = self.status(timeout=min(5, max(0.1, deadline - time.monotonic())))
            if len(cached_keys(last) - initial_keys) >= expected // self.cp_unit:
                return last
            time.sleep(min(0.1, max(0, deadline - time.monotonic())))
        raise AssertionError(f"P did not publish prefix {expected}: {last}")

    def generate(self, directory, label, prompt, input_ids, count, reuse, pd_sep=True):
        require(
            not pd_sep or count >= 2,
            "PD probe requires at least two generated tokens",
        )
        query = generation_query(prompt, count, reuse)
        query["generate_config"].update(
            can_use_pd_separation=pd_sep,
            aux_info=True,
            random_seed=42,
        )
        self.save(directory / f"{label}-request.json", query)
        response = self.post("decode" if pd_sep else "prefill", "/", query)
        self.save(directory / f"{label}-response.json", response)
        validate_generation(response, input_ids, count, pd_sep)
        return response

    def persist_rows(self, directory, rows):
        self.save(directory / "turns.json", rows)
        if rows:
            with (directory / "turns.csv").open("w", newline="") as output:
                writer = csv.DictWriter(output, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)

    def run_case(self, name, turns, aligned):
        require(turns >= 2, "A conversation needs at least two turns")
        directory = self.artifacts / name
        directory.mkdir(parents=True, exist_ok=True)
        initial = self.status()
        initial_keys = cached_keys(initial)
        self.save(directory / "initial-cache.json", initial)
        history = []
        published_prefix = 0
        previous_prompt_prefix = 0
        rows = []
        for turn in range(1, turns + 1):
            label = f"turn-{turn:02d}"
            prompt, input_ids, count = build_turn(
                self.tokenizer, history, name, turn, self.cp_unit, aligned
            )
            expected = expected_reuse(
                len(input_ids), published_prefix, self.cp_unit, self.min_replay_tokens
            )
            expected_first_use = max(0, expected - previous_prompt_prefix)
            preserved = common_prefix_length(history, input_ids)
            require(preserved >= expected, f"{label}: cached conversation prefix changed")
            response = self.generate(directory, label, prompt, input_ids, count, True)
            aux = response["aux_info"]
            ready = len(input_ids) + len(token_row(response, "output_ids")) - 1
            published = complete_prefix(len(input_ids), count, self.cp_unit)
            row = {
                "turn": turn,
                "input_len": len(input_ids),
                "output_len": count,
                "reuse_len": aux.get("reuse_len"),
                "actual_reuse_len": aux.get("actual_reuse_len"),
                "writeback_first_use_len": aux.get("writeback_first_use_len"),
                "prefill_total_reuse_len": aux.get("prefill_total_reuse_len"),
                "decode_total_reuse_len": aux.get("decode_total_reuse_len"),
                "expected_actual_reuse_len": expected,
                "expected_writeback_first_use_len": expected_first_use,
                "replay_limited": expected < published_prefix,
                "kv_ready_token_count": ready,
                "expected_published_prefix": published,
                "writeback_eligible": published > len(input_ids) // self.cp_unit * self.cp_unit,
                "preserved_history_tokens": preserved,
            }
            rows.append(row)
            self.persist_rows(directory, rows)
            print(json.dumps({"case": name, **row}), flush=True)
            validate_metrics(aux, expected, expected_first_use)
            status = self.wait_for_prefix(initial_keys, published)
            self.save(directory / f"{label}-published-cache.json", status)
            history = input_ids + token_row(response, "output_ids")
            published_prefix = published
            previous_prompt_prefix = len(input_ids) // self.cp_unit * self.cp_unit

        prefix = (
            self.tokenizer.decode(
                history, skip_special_tokens=False, clean_up_tokenization_spaces=False
            )
            + "<\uff5cUser\uff5c>Summarize this conversation in one sentence."
        )
        suffix = "<\uff5cAssistant\uff5c></think>"
        prompt = prefix + suffix
        input_ids = self.tokenizer.encode(prompt)
        while len(input_ids) < published_prefix + self.min_replay_tokens:
            prefix += " cache"
            prompt = prefix + suffix
            input_ids = self.tokenizer.encode(prompt)
        expected = published_prefix
        expected_first_use = max(0, expected - previous_prompt_prefix)
        require(
            common_prefix_length(history, input_ids) >= expected,
            "Final probe changed cached prefix",
        )
        warm = self.generate(directory, "final-probe", prompt, input_ids, 2, True)
        validate_metrics(warm["aux_info"], expected, expected_first_use)
        direct = self.generate(
            directory, "prefill-only-probe", prompt, input_ids, 1, True, False
        )
        require(
            direct["aux_info"].get("actual_reuse_len", -1) >= expected,
            "P-only probe lost published prefix",
        )
        validate_metrics(
            direct["aux_info"], direct["aux_info"]["actual_reuse_len"], 0, pd_sep=False
        )
        cold = self.generate(directory, "cold-control", prompt, input_ids, 2, False)
        validate_cold_control(cold["aux_info"])
        compare_outputs(warm, cold)
        result = {
            "case": name,
            "turns": turns,
            "cp_unit": self.cp_unit,
            "min_replay_tokens": self.min_replay_tokens,
            "boundary_aligned": aligned,
            "passed": True,
            "positive_first_use_turns": sum(row["writeback_first_use_len"] > 0 for row in rows),
            "eligible_writeback_turns": sum(row["writeback_eligible"] for row in rows),
            "replay_limited_turns": sum(row["replay_limited"] for row in rows),
            "final_probe_aux": warm["aux_info"],
            "prefill_only_probe_aux": direct["aux_info"],
            "cold_control_aux": cold["aux_info"],
            "warm_cold_next_token_equal": token_row(warm, "output_ids") == token_row(cold, "output_ids"),
        }
        self.save(directory / "result.json", result)
        print(json.dumps({"case": name, "result": "passed", "turns": turns}), flush=True)
        return result


# ---------------------------------------------------------------------------
# Mode 4: Concurrent OpenAI chat usage / billing accounting acceptance
# ---------------------------------------------------------------------------


def cached_usage(response):
    details = response["usage"].get("prompt_tokens_details") or {}
    return details.get("cached_tokens") or 0


def validate_stream_usage(frames, input_length, output_length, expected_cached):
    previous_output = 0
    observed = 0
    for frame in frames:
        usage = frame.get("usage")
        if usage is None:
            continue
        observed += 1
        completion = usage["completion_tokens"]
        require(usage["prompt_tokens"] == input_length, "Stream accumulated prompt tokens")
        require(
            previous_output <= completion <= output_length,
            "Stream completion usage is not cumulative",
        )
        require(
            usage["total_tokens"] == input_length + completion,
            "Stream total usage is inconsistent",
        )
        require(
            cached_usage(frame) == expected_cached,
            "Stream cached usage changed or accumulated",
        )
        previous_output = completion
    require(observed > 0, "Stream omitted usage")
    require(previous_output == output_length, "Stream omitted final completion usage")


def validate_billing(response, input_ids, count, expected, allowed_first_use):
    aux = response["aux_info"]
    validate_metrics(aux, expected)
    require(aux.get("pd_sep") is True, "Chat request bypassed PD")
    require(
        aux["writeback_first_use_len"] in allowed_first_use,
        f"Unexpected first-use snapshot: {aux['writeback_first_use_len']}",
    )
    require(aux["input_len"] == len(input_ids), "Incorrect chat input length")
    require(aux["output_len"] == count, "Incorrect chat output length")
    usage = response["usage"]
    require(usage["prompt_tokens"] == len(input_ids), "Incorrect billed prompt tokens")
    require(usage["completion_tokens"] == count, "Incorrect billed completion tokens")
    require(
        usage["total_tokens"] == len(input_ids) + count,
        "Incorrect billed total tokens",
    )
    require(cached_usage(response) == aux["reuse_len"], "Usage differs from reuse_len")
    extra = response.get("extra_outputs") or {}
    debug = response.get("debug_info") or {}
    returned_inputs = extra.get("input_ids") or debug.get("input_ids")
    if returned_inputs:
        if isinstance(returned_inputs[0], list):
            returned_inputs = returned_inputs[0]
        require(returned_inputs == input_ids, "Chat tokenization changed the prompt")


def max_overlap(records):
    events = []
    for record in records:
        events.extend([(record["start"], 1), (record["end"], -1)])
    active = 0
    peak = 0
    for timestamp, delta in sorted(events):
        active += delta
        peak = max(peak, active)
    return peak


class ConcurrentBillingAcceptance(MultiturnAcceptance):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rows = []
        self.waves = []

    def seed(self, name):
        directory = self.artifacts / "sources" / name
        directory.mkdir(parents=True)
        initial_keys = cached_keys(self.status())
        prompt, inputs, count = build_turn(
            self.tokenizer, [], name, 1, self.cp_unit, True
        )
        response = self.generate(directory, "source", prompt, inputs, count, True)
        validate_metrics(response["aux_info"], 0, 0)
        outputs = token_row(response, "output_ids")
        published = complete_prefix(len(inputs), len(outputs), self.cp_unit)
        require(published > len(inputs), "Source did not publish generated KV")
        status = self.wait_for_prefix(initial_keys, published)
        self.save(directory / "published-cache.json", status)
        history = inputs + outputs
        prefix = (
            self.tokenizer.decode(
                history, skip_special_tokens=False, clean_up_tokenization_spaces=False
            )
            + "<\uff5cUser\uff5c>Explain the next step in this example."
        )
        suffix = "<\uff5cAssistant\uff5c></think>"
        while True:
            prompt = prefix + suffix
            probe_ids = self.tokenizer.encode(prompt)
            if len(probe_ids) >= published + self.min_replay_tokens + 16:
                break
            prefix += " cache"
        require(
            common_prefix_length(history, probe_ids) >= published,
            "Continuation changed the published token IDs",
        )
        require(
            len(probe_ids) + 4 < published + self.cp_unit,
            "Probe would introduce another complete cache unit",
        )
        return {"prompt": prompt, "input_ids": probe_ids, "published": published}

    def chat(self, directory, label, seed, reuse, stream):
        config = {
            "max_new_tokens": 4,
            "min_new_tokens": 4,
            "ignore_eos": True,
            "reuse_cache": reuse,
            "enable_device_cache": reuse,
            "enable_memory_cache": False,
            "enable_disk_cache": False,
            "enable_remote_cache": False,
            "can_use_pd_separation": True,
            "return_input_ids": True,
            "return_output_ids": True,
            "aux_info": True,
            "random_seed": 42,
        }
        query = {
            "messages": [{"role": "user", "content": seed["prompt"]}],
            "user_template": "{{ messages[0]['content'] }}",
            "max_tokens": 4,
            "temperature": 0.0,
            "top_k": 1,
            "top_p": 1.0,
            "stream": stream,
            "debug_info": True,
            "aux_info": True,
            "enable_thinking": False,
            "extra_configs": config,
        }
        self.save(directory / f"{label}-request.json", query)
        if not stream:
            response = self.post("decode", "/v1/chat/completions", query)
        else:
            request = urllib.request.Request(
                f"http://127.0.0.1:{self.ports['decode']}/v1/chat/completions",
                data=json.dumps(query).encode(),
                headers={"Content-Type": "application/json"},
            )
            frames = []
            response = {}
            with urllib.request.urlopen(request, timeout=600) as transport:
                for raw in transport:
                    line = raw.decode().strip()
                    if not line.startswith("data:"):
                        continue
                    payload = line[5:].strip()
                    if payload == "[DONE]":
                        break
                    frame = json.loads(payload)
                    frames.append(frame)
                    for key in ("usage", "aux_info", "extra_outputs", "debug_info"):
                        if frame.get(key) is not None:
                            response[key] = frame[key]
            self.save(directory / f"{label}-frames.json", frames)
            validate_stream_usage(
                frames, len(seed["input_ids"]), 4, cached_usage(response)
            )
        self.save(directory / f"{label}-response.json", response)
        return response

    def wave(self, name, seeds, reuse, allowed_first_use, stream=False):
        directory = self.artifacts / name
        directory.mkdir(parents=True)
        barrier = threading.Barrier(len(seeds))

        def request(index):
            label = f"request-{index:02d}"
            barrier.wait(timeout=30)
            start = time.monotonic()
            response = self.chat(directory, label, seeds[index], reuse[index], stream)
            end = time.monotonic()
            expected = seeds[index]["published"] if reuse[index] else 0
            validate_billing(
                response,
                seeds[index]["input_ids"],
                4,
                expected,
                allowed_first_use[index],
            )
            if not reuse[index]:
                validate_cold_control(response["aux_info"])
            aux = response["aux_info"]
            row = {
                "wave": name,
                "request": index,
                "stream": stream,
                "reuse_enabled": reuse[index],
                "start": start,
                "end": end,
                "input_len": len(seeds[index]["input_ids"]),
                "reuse_len": aux["reuse_len"],
                "actual_reuse_len": aux["actual_reuse_len"],
                "writeback_first_use_len": aux["writeback_first_use_len"],
                "cached_tokens": cached_usage(response),
                "prompt_tokens": response["usage"]["prompt_tokens"],
                "completion_tokens": response["usage"]["completion_tokens"],
            }
            print(json.dumps(row), flush=True)
            return row

        with ThreadPoolExecutor(max_workers=len(seeds)) as executor:
            rows = list(executor.map(request, range(len(seeds))))
        peak = max_overlap(rows)
        require(peak == len(seeds), f"Wave did not overlap all requests: {peak}")
        self.rows.extend(rows)
        summary = {
            "name": name,
            "requests": len(rows),
            "max_inflight": peak,
            "first_use_lengths": [row["writeback_first_use_len"] for row in rows],
            "cached_tokens_sum": sum(row["cached_tokens"] for row in rows),
        }
        self.waves.append(summary)
        self.save(self.artifacts / "waves.json", self.waves)
        self.save(self.artifacts / "requests.json", self.rows)
        with (self.artifacts / "requests.csv").open("w", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=list(self.rows[0]))
            writer.writeheader()
            writer.writerows(self.rows)
        return rows

    def run(self):
        for concurrency in (2, 4):
            shared = self.seed(f"shared_{concurrency}")
            unit = shared["published"]
            fresh = self.wave(
                f"shared_fresh_{concurrency}",
                [shared] * concurrency,
                [True] * concurrency,
                [{0, unit}] * concurrency,
            )
            require(
                any(row["writeback_first_use_len"] == unit for row in fresh),
                "Fresh shared writeback was already accounted before all probes",
            )
            self.wave(
                f"shared_accounted_{concurrency}",
                [shared] * concurrency,
                [True] * concurrency,
                [{0}] * concurrency,
            )
            if concurrency == 4:
                self.wave(
                    "shared_accounted_stream_4",
                    [shared] * concurrency,
                    [True] * concurrency,
                    [{0}] * concurrency,
                    stream=True,
                )
        distinct = [self.seed(f"distinct_{index}") for index in range(4)]
        self.wave(
            "distinct_fresh_4",
            distinct,
            [True] * 4,
            [{seed["published"]} for seed in distinct],
        )
        self.wave(
            "mixed_warm_cold_4",
            distinct,
            [True, False, True, False],
            [{0}] * 4,
        )
        return {
            "passed": True,
            "requests": len(self.rows),
            "waves": self.waves,
            "usage_mismatches": 0,
            "scope": "Actual OpenAI-compatible chat token usage, not monetary billing",
        }


# ---------------------------------------------------------------------------
# Unified CLI entrypoint
# ---------------------------------------------------------------------------


def _run_v4_cp(arguments):
    require(
        arguments.block_size > 0 and arguments.prefill_cp_size > 1,
        "Expected asymmetric CP",
    )
    live_info = json.loads(arguments.live_info.read_text())
    run = AcceptanceRun(live_info, arguments.artifacts)
    result = {"passed": False, "started_at": time.time()}
    try:
        run.save("live-info", live_info)
        validate_topology(live_info, arguments.block_size, arguments.prefill_cp_size)
        from rtp_llm.frontend.tokenizer_factory.tokenizer_factory import (
            TokenizerFactory,
        )

        tokenizer = TokenizerFactory.create(
            arguments.checkpoint, arguments.checkpoint, "deepseek_v4"
        )
        result.update(
            run.run(tokenizer, arguments.block_size * arguments.prefill_cp_size)
        )
    except Exception as error:
        result["error"] = str(error)
        raise
    finally:
        result["finished_at"] = time.time()
        run.save("result", result)
        print(json.dumps(result), flush=True)


def _run_asymmetric_tp(arguments):
    run = AcceptanceRun(json.loads(arguments.live_info.read_text()), arguments.artifacts)
    result = {"passed": False}
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            arguments.checkpoint, trust_remote_code=True
        )
        result.update(validate_writeback(run, tokenizer, block_size=arguments.block_size))
    except Exception as error:
        result["error"] = str(error)
        raise
    finally:
        run.save("result", result)
        print(json.dumps(result, indent=2), flush=True)


def _load_tokenizer(checkpoint, model_type):
    if model_type:
        from rtp_llm.frontend.tokenizer_factory.tokenizer_factory import (
            TokenizerFactory,
        )

        return TokenizerFactory.create(checkpoint, checkpoint, model_type)
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)


def _run_multiturn(arguments):
    tokenizer = _load_tokenizer(arguments.checkpoint, arguments.model_type)
    runner = MultiturnAcceptance(
        tokenizer,
        arguments.prefill_port,
        arguments.decode_port,
        arguments.artifacts,
        arguments.cp_unit,
        min_replay_tokens=arguments.min_replay_tokens,
    )
    runner.run_case(arguments.case, arguments.turns, arguments.aligned)


def _run_billing(arguments):
    tokenizer = _load_tokenizer(arguments.checkpoint, arguments.model_type)
    runner = ConcurrentBillingAcceptance(
        tokenizer,
        arguments.prefill_port,
        arguments.decode_port,
        arguments.artifacts,
        arguments.cp_unit,
        min_replay_tokens=arguments.min_replay_tokens,
    )
    result = runner.run()
    runner.save(Path(arguments.artifacts) / "result.json", result)
    print(json.dumps(result, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)

    v4_parser = subparsers.add_parser("v4-cp")
    v4_parser.add_argument("--live-info", type=Path, required=True)
    v4_parser.add_argument("--checkpoint", required=True)
    v4_parser.add_argument("--artifacts", type=Path, required=True)
    v4_parser.add_argument("--block-size", type=int, default=1024)
    v4_parser.add_argument("--prefill-cp-size", type=int, default=2)
    v4_parser.set_defaults(handler=_run_v4_cp)

    asym_parser = subparsers.add_parser("asymmetric-tp")
    asym_parser.add_argument("--live-info", type=Path, required=True)
    asym_parser.add_argument("--checkpoint", required=True)
    asym_parser.add_argument("--artifacts", type=Path, required=True)
    asym_parser.add_argument("--block-size", type=int, default=16)
    asym_parser.set_defaults(handler=_run_asymmetric_tp)

    for mode, handler in (("multiturn", _run_multiturn), ("billing", _run_billing)):
        sub = subparsers.add_parser(mode)
        sub.add_argument("--prefill-port", type=int, required=True)
        sub.add_argument("--decode-port", type=int, required=True)
        sub.add_argument("--checkpoint", required=True)
        sub.add_argument("--artifacts", type=Path, required=True)
        sub.add_argument("--cp-unit", type=int, required=True)
        sub.add_argument("--min-replay-tokens", type=int, default=1)
        sub.add_argument("--model-type", default="deepseek_v4")
        if mode == "multiturn":
            sub.add_argument("--case", default="multiturn")
            sub.add_argument("--turns", type=int, default=4)
            sub.add_argument("--aligned", action="store_true")
        sub.set_defaults(handler=handler)

    arguments = parser.parse_args()
    arguments.handler(arguments)


if __name__ == "__main__":
    main()

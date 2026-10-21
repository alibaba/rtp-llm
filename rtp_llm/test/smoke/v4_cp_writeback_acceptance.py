"""Validate decode writeback against isolated, already-running V4 PD services."""

import argparse
import json
import math
import shlex
import time
import urllib.error
import urllib.request
from pathlib import Path


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


def validate_topology(live_info, block_size, prefill_cp_size):
    parser = argparse.ArgumentParser(add_help=False)
    for name in (
        "tp_size",
        "seq_size_per_block",
        "prefill_cp_kv_cache_sharded",
        "p2p_writeback_enable",
        "p2p_writeback_timeout_ms",
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
            arguments.p2p_writeback_timeout_ms is not None
            and 0 < arguments.p2p_writeback_timeout_ms <= 30000,
            f"Expected bounded {role} writeback timeout <= 30000 ms",
        )
        require(
            arguments.load_cache_timeout_ms is not None
            and arguments.load_cache_timeout_ms >= arguments.p2p_writeback_timeout_ms,
            f"{role} load timeout truncates writeback budget",
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


def validate_reuse(response, expected, pd_sep):
    aux = response.get("aux_info", {})
    fields = (
        ("prefill_total_reuse_len", "prefill_local_reuse_len")
        if pd_sep
        else ("reuse_len", "local_reuse_len")
    )
    for field in fields:
        require(
            aux.get(field) == expected,
            f"{field}: expected {expected}, got {aux.get(field)}",
        )
    if expected == 0 and pd_sep:
        for field in ("decode_total_reuse_len", "decode_local_reuse_len"):
            require(aux.get(field) == 0, f"Cold control reused D cache: {field}")


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
        validate_reuse(warm_p, cp_unit, False)
        warm_pd = self.generate(
            "warm-pd", "decode", continuation, complete_ids, 8, True, True
        )
        validate_reuse(warm_pd, cp_unit, True)
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live-info", type=Path, required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--block-size", type=int, default=1024)
    parser.add_argument("--prefill-cp-size", type=int, default=2)
    arguments = parser.parse_args()
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


if __name__ == "__main__":
    main()

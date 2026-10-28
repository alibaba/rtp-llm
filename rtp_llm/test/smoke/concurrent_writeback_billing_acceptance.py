"""Validate request-local writeback accounting through the chat usage endpoint."""

import csv
import json
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

from v4_cp_writeback_acceptance import cached_keys, require, token_row
from writeback_multiturn_acceptance import (
    MultiturnAcceptance,
    build_turn,
    common_prefix_length,
    complete_prefix,
    validate_cold_control,
    validate_metrics,
)


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
        prefix = self.tokenizer.decode(
            history, skip_special_tokens=False, clean_up_tokenization_spaces=False
        ) + "<｜User｜>Explain the next step in this example."
        suffix = "<｜Assistant｜></think>"
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
            "max_new_tokens": 4, "min_new_tokens": 4, "ignore_eos": True,
            "reuse_cache": reuse, "enable_device_cache": reuse,
            "enable_memory_cache": False, "enable_disk_cache": False,
            "enable_remote_cache": False, "can_use_pd_separation": True,
            "return_input_ids": True, "return_output_ids": True,
            "aux_info": True, "random_seed": 42,
        }
        query = {
            "messages": [{"role": "user", "content": seed["prompt"]}],
            "user_template": "{{ messages[0]['content'] }}",
            "max_tokens": 4, "temperature": 0.0, "top_k": 1, "top_p": 1.0,
            "stream": stream, "debug_info": True, "aux_info": True,
            "enable_thinking": False, "extra_configs": config,
        }
        self.save(directory / f"{label}-request.json", query)
        if not stream:
            response = self.post("prefill", "/v1/chat/completions", query)
        else:
            request = urllib.request.Request(
                f"http://127.0.0.1:{self.ports['prefill']}/v1/chat/completions",
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
                response, seeds[index]["input_ids"], 4, expected,
                allowed_first_use[index],
            )
            if not reuse[index]:
                validate_cold_control(response["aux_info"])
            aux = response["aux_info"]
            row = {
                "wave": name, "request": index, "stream": stream,
                "reuse_enabled": reuse[index], "start": start, "end": end,
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
            "name": name, "requests": len(rows), "max_inflight": peak,
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
                f"shared_fresh_{concurrency}", [shared] * concurrency,
                [True] * concurrency, [{0, unit}] * concurrency,
            )
            require(
                any(row["writeback_first_use_len"] == unit for row in fresh),
                "Fresh shared writeback was already accounted before all probes",
            )
            self.wave(
                f"shared_accounted_{concurrency}", [shared] * concurrency,
                [True] * concurrency, [{0}] * concurrency,
            )
            if concurrency == 4:
                self.wave(
                    "shared_accounted_stream_4", [shared] * concurrency,
                    [True] * concurrency, [{0}] * concurrency, stream=True,
                )
        distinct = [self.seed(f"distinct_{index}") for index in range(4)]
        self.wave(
            "distinct_fresh_4", distinct, [True] * 4,
            [{seed["published"]} for seed in distinct],
        )
        self.wave(
            "mixed_warm_cold_4", distinct, [True, False, True, False], [{0}] * 4,
        )
        return {
            "passed": True, "requests": len(self.rows), "waves": self.waves,
            "usage_mismatches": 0,
            "scope": "Actual OpenAI-compatible chat token usage, not monetary billing",
        }

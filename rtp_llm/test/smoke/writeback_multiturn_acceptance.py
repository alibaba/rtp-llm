"""Measure published decode KV reuse across consecutive PD conversation turns."""

import csv
import json
import time
import urllib.request
from pathlib import Path

from v4_cp_writeback_acceptance import (
    cache_block_size,
    cached_keys,
    compare_outputs,
    generation_query,
    require,
    token_row,
    validate_generation,
)


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
            "<｜begin▁of▁sentence｜>"
            f"Conversation {case_name}. You are a database engineering tutor.\n"
        )
    user = f"<｜User｜>Round {turn}. {questions[(turn - 1) % len(questions)]}"
    user += " Continue with a detailed explanation in complete sentences.\n"
    suffix = "<｜Assistant｜></think>"
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
            "prefill", "/rtp_llm/cache_status",
            {"latest_cache_version": -1, "need_cache_keys": True}, timeout,
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
            can_use_pd_separation=pd_sep, aux_info=True, random_seed=42,
        )
        self.save(directory / f"{label}-request.json", query)
        response = self.post("prefill", "/", query)
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

        prefix = self.tokenizer.decode(
            history, skip_special_tokens=False, clean_up_tokenization_spaces=False
        ) + "<｜User｜>Summarize this conversation in one sentence."
        suffix = "<｜Assistant｜></think>"
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

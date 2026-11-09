"""Validate BF16 asymmetric TP writeback without requiring bit-exact recomputation."""

import argparse
import json
import math
import time
from pathlib import Path

from v4_cp_writeback_acceptance import (
    AcceptanceRun,
    cache_block_size,
    cached_keys,
    compare_outputs,
    generation_query,
    require,
    token_row,
    validate_generation,
    validate_reuse,
    wait_for_publication,
)


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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live-info", type=Path, required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--artifacts", type=Path, required=True)
    arguments = parser.parse_args()
    run = AcceptanceRun(json.loads(arguments.live_info.read_text()), arguments.artifacts)
    result = {"passed": False}
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            arguments.checkpoint, trust_remote_code=True
        )
        result.update(validate_writeback(run, tokenizer))
    except Exception as error:
        result["error"] = str(error)
        raise
    finally:
        run.save("result", result)
        print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()

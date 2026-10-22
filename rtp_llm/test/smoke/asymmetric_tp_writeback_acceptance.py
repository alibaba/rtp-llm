"""Validate BF16 asymmetric TP writeback without requiring bit-exact recomputation."""

import argparse
import json
import math
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
    query["generate_config"].update(can_use_pd_separation=pd_sep, return_logits=True)
    run.save(name + "-request", query)
    response = run.post(role, "/", query)
    run.save(name, response)
    validate_generation(response, input_ids, 2, pd_sep)
    return response


def validate_writeback(run, tokenizer, block_size=16):
    status_query = {"latest_cache_version": -1, "need_cache_keys": True}
    initial = run.post("prefill", "/rtp_llm/cache_status", status_query)
    run.save("initial-cache", initial)
    require(cache_block_size(initial) == block_size, "Unexpected cache block size")
    require(not cached_keys(initial), "Acceptance requires an empty P cache")
    prompt = "Write a detailed explanation of database transactions."
    input_ids = tokenizer.encode(prompt)
    require(
        0 < len(input_ids) < block_size, "Source prompt must not publish a P block"
    )
    source = run.generate(
        "source", "decode", prompt, input_ids, block_size * 3, True, True
    )
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
                validate_reuse(warm, prefix_blocks * block_size, role == "decode")
                metrics = compare_handoff(warm, cold)
                if previous is not None:
                    compare_outputs(warm, previous)
                previous = warm
                comparisons.append({"case": label, "repeat": repeat, **metrics})
    return {
        "passed": True,
        "published_keys": len(cached_keys(publication)),
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

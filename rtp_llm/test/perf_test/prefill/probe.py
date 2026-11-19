"""One-block cache reuse admission probe, separate from formal measurements."""

from typing import Any, Mapping
from uuid import uuid4

from rtp_llm.test.perf_test.prefill.block_size import resolve_cache_block_size
from rtp_llm.test.perf_test.prefill.prompts import PrefixPromptFactory
from rtp_llm.test.perf_test.prefill.runner import PrefillRunner


def resolve_probe_block_size(config: Mapping[str, Any]) -> int:
    workload = config["workload"]
    engine = config["engine"]
    return resolve_cache_block_size(
        workload.get("probe_block_size"),
        workload["cache_alignment"],
        engine.get("args", {}),
        engine.get("env", {}),
    )


def probe_cache_block(client, tokenizer, block_size, commit_tail_tokens):
    if type(block_size) is not int or block_size <= 0:
        raise ValueError("probe block size must be a positive integer")
    factory = PrefixPromptFactory(tokenizer)
    case_id = f"cache_probe_{uuid4().hex}"
    prompts = factory.build_case_prompts(case_id, 2 * block_size, block_size, 1)
    seed_text, seed_ids = factory.make_seed_and_ids(
        case_id, prompts.seed_text, block_size, commit_tail_tokens
    )
    client.configure_scheduler(1, mode="prefill")
    responses = {}
    for phase, text, ids, reuse in (
        ("seed", seed_text, seed_ids, 0),
        ("measure", prompts.run_texts[0], prompts.run_ids[0], block_size),
    ):
        response = client.request(
            phase=phase,
            case_id=case_id,
            request_index=0,
            run_index=None if phase == "seed" else 0,
            prompt=text,
            input_ids=ids,
            output_length=1,
        )
        invalid = PrefillRunner._validate(
            response, input_len=len(ids), expected_reuse=reuse
        )
        if invalid:
            raise ValueError(f"cache block probe {phase}: {invalid[0]}: {invalid[1]}")
        responses[phase] = response
    return {
        "schema_version": 1,
        "status": "ok",
        "block_size": block_size,
        "commit_tail_tokens": commit_tail_tokens,
        "responses": responses,
    }

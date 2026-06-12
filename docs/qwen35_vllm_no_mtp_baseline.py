#!/usr/bin/env python3
"""vLLM greedy baseline for the Qwen3.5 397B no-MTP smoke prompts."""

import argparse
import json
import os
import sys
import time

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
sys.stdout = os.fdopen(sys.stdout.fileno(), "w", buffering=1)

from vllm import LLM, SamplingParams

MODEL_PATH = "/home/zw193905/models/Qwen3.5-397B-A17B-FP8"
OUT_PATH = (
    "/home/zw193905/RTP-LLM/github-opensource/docs/" "qwen35_vllm_no_mtp_baseline.json"
)

SMOKE_MESSAGES = {
    "capital_france": [{"role": "user", "content": "What is the capital of France?"}],
    "translate_hello": [
        {"role": "user", "content": "Translate to French: 'Hello, how are you today?'"}
    ],
}


def build_prompt(tokenizer, messages):
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=MODEL_PATH)
    parser.add_argument("--out", default=OUT_PATH)
    parser.add_argument("--tp", type=int, default=4)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--gpu-mem", type=float, default=0.9)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--seed", type=int, default=12345)
    args = parser.parse_args()

    print(f"Loading vLLM model: {args.model}", flush=True)
    t0 = time.time()
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tp,
        trust_remote_code=True,
        dtype="bfloat16",
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_mem,
        enforce_eager=True,
        seed=args.seed,
        mamba_cache_mode="align",
        mamba_ssm_cache_dtype="float32",
        skip_mm_profiling=True,
    )
    print(f"Loaded in {time.time() - t0:.1f}s", flush=True)

    tokenizer = llm.get_tokenizer()
    params = SamplingParams(
        temperature=0.0,
        top_k=1,
        top_p=1.0,
        max_tokens=args.max_tokens,
    )

    prompts = []
    names = []
    for name, messages in SMOKE_MESSAGES.items():
        prompt = build_prompt(tokenizer, messages)
        names.append(name)
        prompts.append(prompt)
        print(f"{name}: prompt tokens={len(tokenizer.encode(prompt))}", flush=True)

    t1 = time.time()
    outputs = llm.generate(prompts, params)
    print(f"Generated {len(outputs)} prompts in {time.time() - t1:.1f}s", flush=True)

    results = {
        "engine": "vllm",
        "model": args.model,
        "tensor_parallel_size": args.tp,
        "max_model_len": args.max_model_len,
        "sampling": {
            "temperature": 0.0,
            "top_k": 1,
            "top_p": 1.0,
            "max_tokens": args.max_tokens,
            "seed": args.seed,
        },
        "tests": {},
    }

    for name, prompt, out in zip(names, prompts, outputs):
        completion = out.outputs[0]
        token_ids = list(completion.token_ids)
        results["tests"][name] = {
            "messages": SMOKE_MESSAGES[name],
            "prompt": prompt,
            "input_token_ids": list(out.prompt_token_ids or []),
            "input_token_count": len(out.prompt_token_ids or []),
            "output_text": completion.text,
            "output_token_ids": token_ids,
            "output_token_count": len(token_ids),
            "finish_reason": completion.finish_reason,
        }
        print(f"\n=== {name} ===", flush=True)
        print(f"output_text={completion.text!r}", flush=True)
        print(f"output_token_ids={token_ids}", flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved {args.out}", flush=True)


if __name__ == "__main__":
    main()

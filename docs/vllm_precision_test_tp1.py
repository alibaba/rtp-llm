#!/usr/bin/env python3
"""vLLM precision test for GLM-5 4-layer model — TP=1 single GPU baseline."""
import json
import os
import sys

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
sys.stdout = os.fdopen(sys.stdout.fileno(), "w", buffering=1)

from vllm import LLM, SamplingParams

MODEL_PATH = "/home/zw193905/models/GLM-5-BF16-4layer"
OUTPUT_FILE = (
    "/home/zw193905/RTP-LLM/github-opensource/docs/vllm_precision_output_tp1.json"
)


def build_long_prompt(target_tokens=4096):
    base = (
        "Please provide a comprehensive analysis of the following topics. "
        "For each topic, discuss the historical background, current state, "
        "and future implications. Be thorough and detailed in your analysis.\n\n"
    )
    topics = [
        "Topic 1: The evolution of artificial intelligence from rule-based systems "
        "to modern deep learning architectures. Discuss key milestones including "
        "expert systems, neural networks, convolutional networks, transformers, "
        "and large language models. Analyze the role of compute scaling, data "
        "availability, and algorithmic innovations.\n\n",
        "Topic 2: The impact of climate change on global food security. Examine "
        "how rising temperatures, changing precipitation patterns, and extreme "
        "weather events affect agricultural production. Discuss adaptation "
        "strategies including drought-resistant crops, precision agriculture, "
        "and changes in farming practices.\n\n",
        "Topic 3: The transformation of global financial systems through "
        "digital currencies and blockchain technology. Analyze the rise of "
        "Bitcoin, Ethereum, and central bank digital currencies. Discuss "
        "regulatory challenges, environmental concerns, and the potential "
        "for financial inclusion.\n\n",
        "Topic 4: The role of space exploration in advancing human knowledge "
        "and technology. Discuss missions to Mars, the development of reusable "
        "rockets, space tourism, and the search for extraterrestrial life. "
        "Analyze the economic and scientific benefits.\n\n",
    ]
    prompt = base
    while len(prompt) < target_tokens * 3:
        for topic in topics:
            prompt += topic
            if len(prompt) >= target_tokens * 3:
                break
    return prompt


def main():
    print(f"Loading vLLM model from {MODEL_PATH} with TP=1...", flush=True)
    llm = LLM(
        model=MODEL_PATH,
        tensor_parallel_size=1,
        trust_remote_code=True,
        max_model_len=8192,
        gpu_memory_utilization=0.8,
        enforce_eager=True,
        dtype="bfloat16",
    )

    tokenizer = llm.get_tokenizer()

    # Test 1: Short prompt
    print("\n=== Test 1: Short prompt ===", flush=True)
    short_params = SamplingParams(temperature=0.0, top_k=1, max_tokens=20)
    short_outputs = llm.generate(["The capital of France is"], short_params)
    short_text = short_outputs[0].outputs[0].text
    short_ids = list(short_outputs[0].outputs[0].token_ids)
    print(f"  Output: {short_text!r}", flush=True)
    print(f"  Token IDs: {short_ids}", flush=True)

    # Test 2: Long prompt (~2k tokens)
    print("\n=== Test 2: Long prompt (~2k tokens) ===", flush=True)
    long_prompt = build_long_prompt(4096)
    input_ids = tokenizer.encode(long_prompt)
    print(f"  Input tokens: {len(input_ids)}", flush=True)
    long_params = SamplingParams(temperature=0.0, top_k=1, max_tokens=200)
    long_outputs = llm.generate([long_prompt], long_params)
    long_text = long_outputs[0].outputs[0].text
    long_ids = list(long_outputs[0].outputs[0].token_ids)
    print(f"  Output: {long_text!r}", flush=True)
    print(f"  Token IDs (first 30): {long_ids[:30]}", flush=True)

    # Test 3: Medium prompt, 100 output tokens
    print("\n=== Test 3: Medium prompt, 100 output tokens ===", flush=True)
    medium_prompt = "Write a detailed essay about the future of quantum computing and its applications in drug discovery, cryptography, and materials science."
    medium_input_ids = tokenizer.encode(medium_prompt)
    print(f"  Input tokens: {len(medium_input_ids)}", flush=True)
    medium_params = SamplingParams(temperature=0.0, top_k=1, max_tokens=100)
    medium_outputs = llm.generate([medium_prompt], medium_params)
    medium_text = medium_outputs[0].outputs[0].text
    medium_ids = list(medium_outputs[0].outputs[0].token_ids)
    print(f"  Output: {medium_text!r}", flush=True)
    print(f"  Token IDs: {medium_ids}", flush=True)

    results = {
        "model": MODEL_PATH,
        "engine": "vllm",
        "version": "0.21.0",
        "config": "TP=1, BF16, single GPU",
        "tests": {
            "short": {
                "prompt": "The capital of France is",
                "output_text": short_text,
                "output_token_ids": short_ids,
            },
            "long_4k": {
                "prompt_prefix": long_prompt[:200] + "...",
                "full_prompt": long_prompt,
                "input_token_count": len(input_ids),
                "output_text": long_text,
                "output_token_ids": long_ids,
            },
            "medium": {
                "prompt": medium_prompt,
                "input_token_count": len(medium_input_ids),
                "output_text": medium_text,
                "output_token_ids": medium_ids,
            },
        },
    }

    with open(OUTPUT_FILE, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {OUTPUT_FILE}", flush=True)
    print("vLLM TP=1 precision test COMPLETE", flush=True)


if __name__ == "__main__":
    main()

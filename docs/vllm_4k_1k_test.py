#!/usr/bin/env python3
"""vLLM 4k input + 1k output precision test — TP=1 single GPU."""
import json
import os
import sys

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
sys.stdout = os.fdopen(sys.stdout.fileno(), "w", buffering=1)

from vllm import LLM, SamplingParams

MODEL_PATH = "/home/zw193905/models/GLM-5-BF16-4layer"


def build_4k_prompt():
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
    while len(prompt) < 24000:
        for topic in topics:
            prompt += topic
            if len(prompt) >= 24000:
                break
    return prompt


def main():
    print("Loading vLLM TP=1...", flush=True)
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
    prompt_4k = build_4k_prompt()
    input_ids = tokenizer.encode(prompt_4k)
    print(f"4k prompt: {len(prompt_4k)} chars, {len(input_ids)} tokens", flush=True)

    params = SamplingParams(temperature=0.0, top_k=1, max_tokens=1000)
    outputs = llm.generate([prompt_4k], params)
    text = outputs[0].outputs[0].text
    ids = list(outputs[0].outputs[0].token_ids)
    print(f"Output: {len(ids)} tokens", flush=True)
    print(f"First 30 IDs: {ids[:30]}", flush=True)

    result = {
        "model": MODEL_PATH,
        "engine": "vllm",
        "version": "0.21.0",
        "config": "TP=1, BF16, single GPU",
        "prompt_chars": len(prompt_4k),
        "input_tokens": len(input_ids),
        "output_tokens": len(ids),
        "full_prompt": prompt_4k,
        "output_text": text,
        "output_token_ids": ids,
    }

    with open(
        "/home/zw193905/RTP-LLM/github-opensource/docs/vllm_4k_1k_output.json", "w"
    ) as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print("Saved to docs/vllm_4k_1k_output.json", flush=True)


if __name__ == "__main__":
    main()

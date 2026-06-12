#!/usr/bin/env python3
"""Generate vLLM baselines for the GLM-5 full-checkpoint smoke queries."""

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
sys.stdout = os.fdopen(sys.stdout.fileno(), "w", buffering=1)

from vllm import LLM, SamplingParams

MODEL_PATH = "/home/admin/zw193905/models/GLM-5-FP8"
QUERY_FILES = [
    Path("/home/zw193905/query.json"),
    Path("/home/zw193905/query2.json"),
]
OUTPUT_FILE = Path(
    "/home/admin/zw193905/RTP-LLM/github-opensource/docs/glm5_full_ckpt_vllm_baseline.json"
)


def _top_k(query):
    if query.get("top_k") is not None:
        return query["top_k"]
    return (query.get("extend_fields") or {}).get("top_k", 1)


def _finish_reason(reason):
    if reason in ("stop", "length", "tool_calls", "function_call"):
        return reason
    if reason is None:
        return "stop"
    return str(reason)


def main():
    queries = []
    for path in QUERY_FILES:
        with path.open(encoding="utf-8") as f:
            queries.append({"source": str(path), "query": json.load(f)})

    print(f"Loading vLLM model: {MODEL_PATH}")
    llm = LLM(
        model=MODEL_PATH,
        tensor_parallel_size=int(os.environ.get("VLLM_TP_SIZE", "4")),
        trust_remote_code=True,
        max_model_len=int(os.environ.get("VLLM_MAX_MODEL_LEN", "32768")),
        gpu_memory_utilization=float(
            os.environ.get("VLLM_GPU_MEMORY_UTILIZATION", "0.80")
        ),
        enforce_eager=True,
        dtype="bfloat16",
    )
    tokenizer = llm.get_tokenizer()

    results = []
    for idx, item in enumerate(queries):
        query = item["query"]
        prompt = tokenizer.apply_chat_template(
            query["messages"],
            tokenize=False,
            add_generation_prompt=True,
        )
        prompt_ids = tokenizer.encode(prompt)
        sampling_params = SamplingParams(
            max_tokens=int(query.get("max_tokens", 1024)),
            temperature=float(query.get("temperature", 0.0)),
            top_p=float(query.get("top_p", 1.0)),
            top_k=int(_top_k(query)),
        )
        print(
            f"Running query {idx}: source={item['source']} "
            f"prompt_tokens={len(prompt_ids)} max_tokens={sampling_params.max_tokens}"
        )
        output = llm.generate([prompt], sampling_params)[0]
        choice = output.outputs[0]
        token_ids = list(choice.token_ids)
        content = choice.text
        result = {
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": content,
                    },
                    "finish_reason": _finish_reason(choice.finish_reason),
                }
            ],
            "usage": {
                "prompt_tokens": len(output.prompt_token_ids or prompt_ids),
                "completion_tokens": len(token_ids),
                "total_tokens": len(output.prompt_token_ids or prompt_ids)
                + len(token_ids),
            },
        }
        results.append(
            {
                "source": item["source"],
                "query": query,
                "prompt_tokens": len(prompt_ids),
                "output_token_ids": token_ids,
                "result": result,
            }
        )
        print(
            f"Query {idx} done: completion_tokens={len(token_ids)} "
            f"finish_reason={result['choices'][0]['finish_reason']} "
            f"preview={content[:160]!r}"
        )

    payload = {
        "engine": "vllm",
        "model_path": MODEL_PATH,
        "tensor_parallel_size": int(os.environ.get("VLLM_TP_SIZE", "4")),
        "results": results,
    }
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_FILE.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"Saved baseline to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()

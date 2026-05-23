#!/usr/bin/env python3
"""vLLM hidden-state dump for GLM-5-BF16-4layer.

Registers forward hooks on every DeepseekV2DecoderLayer to capture the
returned (hidden_states, residual) pair, then computes the combined output
``hidden_states + residual`` (the real cross-layer "hidden state").

Produces per-prompt .pt files under DUMP_DIR. For each prompt we capture:
  - embed_out
  - layerXX_hidden, layerXX_residual, layerXX_combined  (XX in 00..N-1)
  - final_norm
  - logits_last (last-token logits)
  - output_token_ids
"""
import argparse
import hashlib
import json
import os
import sys
import time

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
# Force EngineCore in-process so we can register forward hooks
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
sys.stdout = os.fdopen(sys.stdout.fileno(), "w", buffering=1)

import torch
from vllm import LLM, SamplingParams

MODEL_PATH = "/home/zw193905/models/GLM-5-BF16-4layer"
DUMP_DIR = "/home/zw193905/RTP-LLM/github-opensource/docs/hidden_align/vllm_dumps"
META_FILE = os.path.join(DUMP_DIR, "meta.json")


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


SHORT_PROMPT = "The capital of France is"
MEDIUM_PROMPT = (
    "Write a detailed essay about the future of quantum computing and its "
    "applications in drug discovery, cryptography, and materials science."
)


def install_layer_hooks(model_runner_model, capture: dict, mode: dict):
    """Register forward hooks on DeepseekV2DecoderLayer.

    Each layer's forward returns (hidden_states, residual).

    We capture the FIRST forward of every dump round (== prefill, contains all
    input tokens). ``mode['phase']`` is set externally per dump round:
       'prefill'  → record on the first call, then ignore
       'all'      → overwrite each call (useful for last-decode capture)
    """
    base_model = model_runner_model
    inner = getattr(base_model, "model", base_model)
    layers = inner.layers
    print(f"  Hooking {len(layers)} decoder layers", flush=True)

    handles = []

    def _should_capture():
        if mode.get("phase") == "prefill":
            return not mode.get("captured", False)
        return True

    def _maybe_mark():
        if mode.get("phase") == "prefill":
            mode["captured"] = True

    def make_hook(idx):
        def hook(module, args, output):
            if not _should_capture():
                return
            try:
                hs, res = output
            except Exception:
                return
            capture[f"layer{idx:02d}_hidden"] = hs.detach().to(torch.float32).cpu()
            capture[f"layer{idx:02d}_residual"] = res.detach().to(torch.float32).cpu()
            capture[f"layer{idx:02d}_combined"] = (
                (hs + res).detach().to(torch.float32).cpu()
            )

        return hook

    for idx, layer in enumerate(layers):
        handles.append(layer.register_forward_hook(make_hook(idx)))

    def embed_hook(module, args, output):
        if not _should_capture():
            return
        capture["embed_out"] = output.detach().to(torch.float32).cpu()

    handles.append(inner.embed_tokens.register_forward_hook(embed_hook))

    def final_norm_hook(module, args, output):
        if not _should_capture():
            return
        if isinstance(output, tuple):
            out0 = output[0]
        else:
            out0 = output
        capture["final_norm"] = out0.detach().to(torch.float32).cpu()
        _maybe_mark()  # the very last hook of prefill marks done

    handles.append(inner.norm.register_forward_hook(final_norm_hook))
    return handles


def summarize(t: torch.Tensor) -> dict:
    cpu = t.to(torch.float32)
    n = cpu.numel()
    return {
        "shape": tuple(cpu.shape),
        "dtype": str(t.dtype),
        "mean": float(cpu.mean().item()) if n else 0.0,
        "std": float(cpu.std().item()) if n > 1 else 0.0,
        "abs_max": float(cpu.abs().max().item()) if n else 0.0,
        "n_nan": int(torch.isnan(cpu).sum().item()),
        "n_inf": int(torch.isinf(cpu).sum().item()),
        "numel": n,
        "md5": hashlib.md5(cpu.contiguous().numpy().tobytes()).hexdigest(),
    }


def dump_one(
    name: str, prompt: str, max_tokens: int, llm: "LLM", hook_capture: dict, mode: dict
):
    print(f"\n=== Prompt {name!r} ===", flush=True)
    hook_capture.clear()
    mode["captured"] = False  # reset prefill capture flag
    tokenizer = llm.get_tokenizer()
    input_ids = tokenizer.encode(prompt)
    print(f"  Input tokens: {len(input_ids)}", flush=True)
    params = SamplingParams(temperature=0.0, top_k=1, max_tokens=max_tokens)
    t0 = time.time()
    outputs = llm.generate([prompt], params)
    out_text = outputs[0].outputs[0].text
    out_ids = list(outputs[0].outputs[0].token_ids)
    print(
        f"  Output tokens: {len(out_ids)} (gen time: {time.time()-t0:.1f}s)", flush=True
    )
    print(f"  Output (first 80 chars): {out_text[:80]!r}", flush=True)

    stats = {}
    saved_keys = []
    pt_payload = {}
    for k, v in hook_capture.items():
        stats[k] = summarize(v)
        # Only keep full tensor for embed/final + small layer-by-layer (always for 4 layers)
        pt_payload[k] = v
        saved_keys.append(k)

    payload = {
        "name": name,
        "prompt": prompt,
        "input_token_count": len(input_ids),
        "input_token_ids": input_ids,
        "output_text": out_text,
        "output_token_ids": out_ids,
        "stats": stats,
        "tensors": pt_payload,
    }
    out_path = os.path.join(DUMP_DIR, f"{name}.pt")
    torch.save(payload, out_path)
    print(f"  Saved {out_path} ({len(saved_keys)} tensors)", flush=True)
    # Also write the lightweight stats JSON
    json_path = os.path.join(DUMP_DIR, f"{name}.stats.json")
    with open(json_path, "w") as f:
        json.dump(
            {
                "name": name,
                "prompt_prefix": prompt[:200],
                "input_token_count": len(input_ids),
                "output_text": out_text,
                "output_token_ids": out_ids,
                "stats": stats,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    return out_ids, out_text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=MODEL_PATH)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--gpu-mem", type=float, default=0.8)
    parser.add_argument(
        "--prompts",
        nargs="*",
        default=["short", "medium", "long_4k"],
        choices=["short", "medium", "long_4k"],
    )
    args = parser.parse_args()

    os.makedirs(DUMP_DIR, exist_ok=True)

    print(f"Loading vLLM model from {args.model}", flush=True)
    llm = LLM(
        model=args.model,
        tensor_parallel_size=1,
        trust_remote_code=True,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_mem,
        enforce_eager=True,
        dtype="bfloat16",
    )

    # Reach into the model (in-process EngineCore required: VLLM_ENABLE_V1_MULTIPROCESSING=0)
    engine = llm.llm_engine
    # vLLM v1 inproc: engine.model_executor is set in __init__
    model_executor = engine.model_executor
    # InprocClient / Executor pattern: driver_worker.worker.model_runner.model
    workers = getattr(model_executor, "workers", None)
    if workers is None:
        # Some executors expose .driver_worker
        driver = getattr(model_executor, "driver_worker", None)
        if driver is not None:
            workers = [driver]
    assert workers, f"Cannot locate workers on executor {type(model_executor).__name__}"
    worker = workers[0]
    # worker may itself be a WorkerWrapperBase that proxies to .worker
    inner_worker = getattr(worker, "worker", worker)
    model_runner = inner_worker.model_runner
    inner_model = model_runner.model
    print(f"  Model class: {type(inner_model).__name__}", flush=True)
    capture: dict = {}
    mode: dict = {"phase": "prefill", "captured": False}
    install_layer_hooks(inner_model, capture, mode)

    prompt_specs = {
        "short": (SHORT_PROMPT, 20),
        "medium": (MEDIUM_PROMPT, 100),
        "long_4k": (build_long_prompt(4096), 50),
    }

    meta = {
        "model": args.model,
        "engine": "vllm",
        "version": __import__("vllm").__version__,
        "dtype": "bfloat16",
        "tp": 1,
        "enforce_eager": True,
        "prompts": {},
    }

    for name in args.prompts:
        prompt, max_tokens = prompt_specs[name]
        out_ids, out_text = dump_one(name, prompt, max_tokens, llm, capture, mode)
        meta["prompts"][name] = {
            "input_token_count": len(llm.get_tokenizer().encode(prompt)),
            "output_token_ids": out_ids,
            "output_text": out_text,
            "max_tokens": max_tokens,
        }

    with open(META_FILE, "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"\nSaved metadata to {META_FILE}", flush=True)
    print("vLLM dump COMPLETE", flush=True)


if __name__ == "__main__":
    main()

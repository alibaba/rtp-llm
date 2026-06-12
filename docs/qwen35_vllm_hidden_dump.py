#!/usr/bin/env python3
"""vLLM hidden-state dump for Qwen3.5 397B smoke prompts."""

import argparse
import hashlib
import json
import os
import sys
import time

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
sys.stdout = os.fdopen(sys.stdout.fileno(), "w", buffering=1)

import torch
from vllm import LLM, SamplingParams

MODEL_PATH = "/home/zw193905/models/Qwen3.5-397B-A17B-FP8"
DUMP_DIR = "/home/zw193905/RTP-LLM/github-opensource/docs/" "qwen35_vllm_hidden_dumps"

SMOKE_MESSAGES = {
    "capital_france": [{"role": "user", "content": "What is the capital of France?"}],
    "translate_hello": [
        {"role": "user", "content": "Translate to French: 'Hello, how are you today?'"}
    ],
}


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


def locate_runner_model(llm):
    model_executor = llm.llm_engine.model_executor
    workers = getattr(model_executor, "workers", None)
    if workers is None:
        driver = getattr(model_executor, "driver_worker", None)
        if driver is not None:
            workers = [driver]
    assert workers, f"Cannot locate workers on {type(model_executor).__name__}"
    worker = workers[0]
    inner_worker = getattr(worker, "worker", worker)
    model_runner = getattr(inner_worker, "model_runner", None)
    assert model_runner is not None, f"Cannot locate model_runner on {type(worker)}"
    model = getattr(model_runner, "model", None)
    assert model is not None, f"Cannot locate model on {type(model_runner)}"
    return model


def locate_text_model(model):
    # Qwen3.5 multimodal wrapper: model.language_model.model
    language_model = getattr(model, "language_model", None)
    if language_model is not None and hasattr(language_model, "model"):
        return language_model.model
    inner = getattr(model, "model", None)
    if inner is not None:
        language_model = getattr(inner, "language_model", None)
        if language_model is not None and hasattr(language_model, "model"):
            return language_model.model
        if hasattr(inner, "layers"):
            return inner
    if hasattr(model, "layers"):
        return model
    raise RuntimeError(f"Cannot locate text model under {type(model)}")


def install_hooks(text_model, capture: dict, mode: dict):
    layers = text_model.layers
    print(f"Hooking {len(layers)} layers", flush=True)
    handles = []

    def should_capture():
        return not mode.get("captured", False)

    def make_hook(idx):
        def hook(module, args, output):
            if not should_capture():
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
        if should_capture():
            capture["embed_out"] = output.detach().to(torch.float32).cpu()

    handles.append(text_model.embed_tokens.register_forward_hook(embed_hook))

    def final_norm_hook(module, args, output):
        if not should_capture():
            return
        out0 = output[0] if isinstance(output, tuple) else output
        capture["final_norm"] = out0.detach().to(torch.float32).cpu()
        mode["captured"] = True

    handles.append(text_model.norm.register_forward_hook(final_norm_hook))
    return handles


def build_prompt(tokenizer, messages):
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def dump_one(name, prompt, llm, capture, mode, max_tokens, dump_dir):
    tokenizer = llm.get_tokenizer()
    input_ids = tokenizer.encode(prompt)
    print(f"\n=== {name} ===", flush=True)
    print(f"input_tokens={len(input_ids)}", flush=True)
    capture.clear()
    mode["captured"] = False

    params = SamplingParams(
        temperature=0.0,
        top_k=1,
        top_p=1.0,
        max_tokens=max_tokens,
    )
    t0 = time.time()
    outputs = llm.generate([prompt], params)
    print(f"generate_time={time.time() - t0:.1f}s", flush=True)
    completion = outputs[0].outputs[0]
    print(f"output_text={completion.text!r}", flush=True)
    print(f"output_token_ids={list(completion.token_ids)}", flush=True)

    stats = {k: summarize(v) for k, v in capture.items()}
    payload = {
        "name": name,
        "prompt": prompt,
        "input_token_ids": input_ids,
        "input_token_count": len(input_ids),
        "output_text": completion.text,
        "output_token_ids": list(completion.token_ids),
        "stats": stats,
        "tensors": dict(capture),
    }
    out_path = os.path.join(dump_dir, f"{name}.pt")
    torch.save(payload, out_path)
    with open(os.path.join(dump_dir, f"{name}.stats.json"), "w") as f:
        json.dump(
            {
                "name": name,
                "input_token_count": len(input_ids),
                "output_text": completion.text,
                "output_token_ids": list(completion.token_ids),
                "stats": stats,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"saved {out_path} tensors={len(capture)}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=MODEL_PATH)
    parser.add_argument("--dump-dir", default=DUMP_DIR)
    parser.add_argument("--tp", type=int, default=8)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--gpu-mem", type=float, default=0.9)
    parser.add_argument("--max-tokens", type=int, default=1)
    args = parser.parse_args()

    os.makedirs(args.dump_dir, exist_ok=True)

    print(f"Loading vLLM model: {args.model}", flush=True)
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tp,
        trust_remote_code=True,
        dtype="bfloat16",
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_mem,
        enforce_eager=True,
        mamba_cache_mode="align",
        mamba_ssm_cache_dtype="float32",
    )
    text_model = locate_text_model(locate_runner_model(llm))
    capture = {}
    mode = {}
    handles = install_hooks(text_model, capture, mode)
    try:
        tokenizer = llm.get_tokenizer()
        for name, messages in SMOKE_MESSAGES.items():
            dump_one(
                name,
                build_prompt(tokenizer, messages),
                llm,
                capture,
                mode,
                args.max_tokens,
                args.dump_dir,
            )
    finally:
        for handle in handles:
            handle.remove()


if __name__ == "__main__":
    main()

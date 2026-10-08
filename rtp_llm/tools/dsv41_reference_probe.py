"""Audit and run the released V4.1 text reference without checkpoint conversion.

Run GPU modes under the repository's GPU lock. The official inference Python
directory is supplied explicitly; no RTP model or attention owner is imported.
"""

import argparse
import hashlib
import importlib
import json
import math
import os
import sys
import time
from collections import Counter
from datetime import timedelta
from pathlib import Path
from unittest.mock import patch


def text_name(source):
    name = source.removeprefix("model.")
    if name.startswith(("vision.", "mtp.", "aligner.", "image_")):
        return None
    if name.endswith(".bias_vl"):
        return None
    return (
        name.replace("self_attn", "attn")
        .replace("mlp", "ffn")
        .replace("weight_scale_inv", "scale")
        .replace("e_score_correction_bias", "bias")
    )


def shard_spec(name, shape, rank, world, experts, mapping):
    """Follow official convert.py sharding, including Engram's padded last rank."""
    shape = list(shape)
    axis, start, stop = None, 0, 0
    if ".experts." in name:
        if experts % world:
            raise ValueError("Expert count must divide the reference world size")
        local = experts // world
        expert = int(name.split(".")[-3])
        if not rank * local <= expert < (rank + 1) * local:
            return None
    elif ".engram.embed." in name:
        axis = 0
        size = (shape[0] + world - 1) // world
        start, stop = min(rank * size, shape[0]), min((rank + 1) * size, shape[0])
        shape[0] = size
    else:
        bare = any(
            part in name
            for part in ("hc", "attn_sink", "tie2eid", "tid2eid", "ape", "image_")
        )
        key = name.split(".")[-1 if bare else -2]
        axis = mapping.get(key, (key, None))[1]
        if axis is not None:
            if shape[axis] % world:
                raise ValueError(f"{name}: tensor dimension does not divide TP")
            shape[axis] //= world
            start, stop = rank * shape[axis], (rank + 1) * shape[axis]
    return tuple(shape), axis, start, stop


def checkpoint_headers(root):
    from rtp_llm.utils.safetensors_header import (
        read_safetensors_header,
        safetensors_file_identity,
        unique_json_object,
    )

    index = json.loads(
        (root / "model.safetensors.index.json").read_text(),
        object_pairs_hook=unique_json_object,
    )["weight_map"]
    headers, identities = {}, {}
    for filename in sorted(set(index.values())):
        path = (root / filename).resolve()
        if path.parent != root or path.name != filename:
            raise ValueError("Checkpoint shards must be directly in the checkpoint")
        identity = safetensors_file_identity(path.stat())
        entries, _ = read_safetensors_header(path)
        if identity != safetensors_file_identity(path.stat()):
            raise RuntimeError("Checkpoint changed during header admission")
        for name, entry in entries.items():
            if name in headers or index.get(name) != filename:
                raise ValueError(f"Checkpoint index/header mismatch: {name}")
            headers[name] = entry
        identities[filename] = identity
    if set(index) != set(headers):
        raise ValueError("Checkpoint index references missing tensors")
    return index, headers, identities


def model_args(module, root, max_seq_len):
    config = json.loads((root / "inference/config.json").read_text())
    config.update(
        max_batch_size=1,
        max_seq_len=max_seq_len,
        vision_n_layers=0,
        dspark_block_size=0,
        n_mtp_layers=0,
        dspark_target_layer_ids=(),
        temperature=0,
    )
    return module.ModelArgs(**config)


def audit(module, converter, root, args, tokenizer, rank, world):
    import torch

    with (
        patch.object(module.dist, "is_initialized", return_value=True),
        patch.object(module.dist, "get_world_size", return_value=world),
        patch.object(module.dist, "get_rank", return_value=rank),
        torch.device("meta"),
    ):
        reference = module.Transformer(args, tokenizer)
    expected = dict(reference.named_parameters())
    index, headers, identities = checkpoint_headers(root)
    dtypes = {
        "BF16": torch.bfloat16,
        "F32": torch.float32,
        "F8_E4M3": torch.float8_e4m3fn,
        "F8_E8M0": torch.float8_e8m0fnu,
        "I8": torch.int8,
        "I64": torch.int64,
    }
    normalized = {}
    for source in headers:
        name = text_name(source)
        if name is not None:
            if name in normalized:
                raise ValueError(f"Duplicate normalized checkpoint name: {name}")
            normalized[name] = source
    plan, casts, dequant_scales = {}, {}, {}
    for source, entry in headers.items():
        name = text_name(source)
        if name is None or name.endswith("wo_a.scale"):
            continue
        spec = shard_spec(
            name, entry["shape"], rank, world, args.n_routed_experts, converter.mapping
        )
        if spec is None:
            continue
        if name not in expected or name in plan:
            raise ValueError(f"Unexpected or duplicate reference parameter: {name}")
        shape, axis, start, stop = spec
        target = expected[name]
        if shape != tuple(target.shape):
            raise ValueError(f"{name}: shape mismatch: {shape} != {target.shape}")
        dtype = dtypes[entry["dtype"]]
        if name.endswith("wo_a.weight"):
            scale_name = normalized[name.removesuffix("weight") + "scale"]
            scale = headers[scale_name]
            if (
                index[scale_name] != index[source]
                or dtype != torch.float8_e4m3fn
                or scale["dtype"] != "F8_E8M0"
                or len(scale["shape"]) != 2
                or any(size <= 0 for size in scale["shape"])
                or any(
                    size % block for size, block in zip(entry["shape"], scale["shape"])
                )
            ):
                raise ValueError(f"{name}: invalid dequantization scale")
            blocks = tuple(
                size // block for size, block in zip(entry["shape"], scale["shape"])
            )
            if (
                blocks not in ((32, 32), (128, 128))
                or start % blocks[0]
                or stop % blocks[0]
            ):
                raise ValueError(f"{name}: unsupported dequantization block geometry")
            dequant_scales[name] = scale_name
            dtype = torch.bfloat16
        elif ".experts." in name and dtype == torch.int8:
            dtype = torch.float4_e2m1fn_x2
        if dtype != target.dtype:
            if (dtype, target.dtype) != (torch.bfloat16, torch.float32):
                raise ValueError(f"{name}: unexpected dtype conversion")
            casts[name] = [str(dtype), str(target.dtype)]
        plan[name] = (source, shape, axis, start, stop)
    if set(plan) != set(expected):
        raise ValueError(f"Missing reference parameters: {set(expected) - set(plan)}")
    groups = Counter()
    for name, value in expected.items():
        group = (
            "engram_tables"
            if ".engram.embed." in name
            else "routed_experts" if ".experts." in name else "other_text_weights"
        )
        groups[group] += value.numel() * value.element_size()
    report = {
        "rank": rank,
        "world_size": world,
        "parameter_bytes": dict(groups),
        "total_parameter_bytes": sum(groups.values()),
        "buffer_bytes": sum(x.numel() * x.element_size() for x in reference.buffers()),
        "parameter_count": len(expected),
        "dtype_casts": casts,
        "max_seq_len": args.max_seq_len,
    }
    return report, (plan, index, identities, dequant_scales)


def copy_parameter(target, source, spec, *, scale=None, chunk_bytes=64 * 1024**2):
    """Bounded CPU slices; no full-table temporary and no second GPU weight copy."""
    import torch

    _, shape, axis, start, stop = spec
    source_shape = source.get_shape()
    rows = stop - start if axis == 0 else source_shape[0]
    row_bytes = max(1, math.prod(source_shape[1:]) * target.element_size())
    chunk_rows = max(1, chunk_bytes // row_bytes)
    if scale is not None:
        scale_shape = scale.get_shape()
        out_block = source_shape[0] // scale_shape[0]
        in_block = source_shape[1] // scale_shape[1]
        if (out_block, in_block) not in ((32, 32), (128, 128)):
            raise ValueError("Unexpected wo_a quantization geometry")
        chunk_rows = max(out_block, chunk_rows // out_block * out_block)
    with torch.no_grad():
        for offset in range(0, rows, chunk_rows):
            count = min(chunk_rows, rows - offset)
            begin = offset + (start if axis == 0 else 0)
            selection = [slice(None)] * len(source_shape)
            selection[0] = slice(begin, begin + count)
            if axis == 1:
                selection[1] = slice(start, stop)
            value = source[tuple(selection)]
            if scale is not None:
                scales = scale[begin // out_block : (begin + count) // out_block]
                value = (
                    (
                        value.unflatten(0, (-1, out_block))
                        .unflatten(-1, (-1, in_block))
                        .float()
                        * scales[:, None, :, None].float()
                    )
                    .flatten(2, 3)
                    .flatten(0, 1)
                    .bfloat16()
                )
            elif target.dtype == torch.float4_e2m1fn_x2:
                value = value.view(torch.float4_e2m1fn_x2)
            target[offset : offset + count].copy_(value)
        if rows < shape[0]:
            target[rows:].fill_(
                1 if spec[0].endswith((".weight_scale_inv", ".scale")) else 0
            )


def load_checkpoint(reference, root, admitted):
    from safetensors import safe_open

    from rtp_llm.utils.safetensors_header import safetensors_file_identity

    plan, index, identities, dequant_scales = admitted
    by_file = {}
    for name, spec in plan.items():
        by_file.setdefault(index[spec[0]], []).append((name, spec))
    parameters = dict(reference.named_parameters())
    started = time.monotonic()
    for file_index, (filename, entries) in enumerate(sorted(by_file.items()), 1):
        path = root / filename
        if safetensors_file_identity(path.stat()) != identities[filename]:
            raise RuntimeError("Checkpoint changed before loading")
        with safe_open(path, framework="pt", device="cpu") as handle:
            for name, spec in entries:
                scale = None
                if name.endswith("wo_a.weight"):
                    scale_name = dequant_scales[name]
                    scale = handle.get_slice(scale_name)
                copy_parameter(
                    parameters[name], handle.get_slice(spec[0]), spec, scale=scale
                )
        if safetensors_file_identity(path.stat()) != identities[filename]:
            raise RuntimeError("Checkpoint changed during loading")
        print(
            f"rank={os.environ.get('RANK', '0')} reference shard={file_index}/{len(by_file)} "
            f"file={filename} seconds={time.monotonic() - started:.2f}",
            flush=True,
        )
    print(
        f"official reference loaded seconds={time.monotonic() - started:.2f}",
        flush=True,
    )


def sample(module, root, args, tokenizer, admitted, prompts, new_tokens):
    import torch
    import torch.distributed as dist

    with torch.device("cuda"):
        reference = module.Transformer(args, tokenizer)
    load_checkpoint(reference, root, admitted)
    reference.eval()
    torch.cuda.synchronize()
    dist.barrier()
    results = []
    for prompt in prompts:
        ids = tokenizer.encode(prompt, add_special_tokens=False)
        if not ids or len(ids) + new_tokens > args.max_seq_len:
            raise ValueError("Prompt and output must fit the admitted context")
        tokens = torch.tensor([ids], device="cuda", dtype=torch.long)
        generated, top_values, top_ids = [], [], []
        start = 0
        started = time.monotonic()
        for _ in range(new_tokens):
            output, logits, _ = reference(tokens, start)
            if not torch.isfinite(logits).all():
                raise ValueError("Official reference produced nonfinite logits")
            next_token = int(output.reshape(-1)[-1])
            top = logits.reshape(-1, args.vocab_size)[-1].float().topk(5)
            generated.append(next_token)
            top_values.append(top.values.tolist())
            top_ids.append(top.indices.tolist())
            start += tokens.shape[1]
            tokens = torch.tensor([[next_token]], device="cuda", dtype=torch.long)
        result = {
            "prompt": prompt,
            "prompt_ids": ids,
            "generated_ids": generated,
            "generated_text": tokenizer.decode(generated, skip_special_tokens=False),
            "top5_ids": top_ids,
            "top5_logits": top_values,
            "seconds": time.monotonic() - started,
            "peak_torch_bytes": torch.cuda.max_memory_allocated(),
        }
        peers = [None] * dist.get_world_size()
        dist.all_gather_object(peers, generated)
        if any(value != generated for value in peers):
            raise ValueError("Official reference TP ranks produced different tokens")
        if dist.get_rank() == 0:
            print(json.dumps(result), flush=True)
        results.append(result)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("audit", "sample"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--extra-package-path", action="append", default=[])
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=8)
    parser.add_argument("--max-seq-len", type=int, default=256)
    parser.add_argument("--new-tokens", type=int, default=9)
    parser.add_argument("--prompt", action="append")
    parser.add_argument("--report", type=Path)
    cli = parser.parse_args()
    root = cli.checkpoint.resolve()
    sys.path[0:0] = [str(root / "inference"), *cli.extra_package_path]
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    import torch
    import torch.distributed as dist
    from transformers import AutoTokenizer

    torch.set_num_threads(1)
    torch.set_default_dtype(torch.bfloat16)
    module = importlib.import_module("model")
    converter = importlib.import_module("convert")
    for imported, filename in ((module, "model.py"), (converter, "convert.py")):
        if Path(imported.__file__).resolve() != root / "inference" / filename:
            raise RuntimeError(f"Unexpected official reference module: {filename}")
    tokenizer = AutoTokenizer.from_pretrained(root, trust_remote_code=False)
    args = model_args(module, root, cli.max_seq_len)
    rank, world = cli.rank, cli.world_size
    if cli.mode == "sample":
        rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        if world != 8 or torch.cuda.get_device_capability() != (12, 0):
            raise ValueError("Reference sampling requires TP8 on SM120")
        if not 1 <= cli.new_tokens <= 16 or cli.max_seq_len != 256:
            raise ValueError(
                "Reference sampling is bounded to 256 positions/16 outputs"
            )
        dist.init_process_group("nccl", timeout=timedelta(seconds=3600))
    if world < 1 or not 0 <= rank < world or cli.max_seq_len < 1:
        raise ValueError("Invalid reference geometry")
    try:
        report, admitted = audit(module, converter, root, args, tokenizer, rank, world)
        report["official_source_sha256"] = {
            name: hashlib.sha256((root / "inference" / name).read_bytes()).hexdigest()
            for name in (
                "model.py",
                "kernel.py",
                "engram.py",
                "convert.py",
                "config.json",
            )
        }
        report["torch"] = torch.__version__
        report["runner_sha256"] = hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest()
        print(json.dumps(report), flush=True)
        if cli.mode == "sample":
            import tilelang

            report["tilelang"] = tilelang.__version__
            report["cuda"] = torch.version.cuda
            report["device"] = torch.cuda.get_device_name()
            free, _ = torch.cuda.mem_get_info()
            required = (
                report["total_parameter_bytes"] + report["buffer_bytes"] + 8 * 1024**3
            )
            if free < required:
                raise RuntimeError(
                    f"Reference needs {required} free bytes; available {free}"
                )
            dist.barrier()
            report["outputs"] = sample(
                module,
                root,
                args,
                tokenizer,
                admitted,
                cli.prompt or ["The capital of France is", "2 + 2 ="],
                cli.new_tokens,
            )
        if cli.report and rank == 0:
            cli.report.write_text(json.dumps(report, indent=2) + "\n")
    finally:
        if cli.mode == "sample" and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()

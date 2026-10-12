"""Bounded selected-checkpoint attention probe; does not start the model/engine."""

import argparse
import hashlib
import importlib.machinery
import json
import os
from pathlib import Path
import subprocess
import sys
import types


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--layers", nargs="+", type=int, default=[2, 20])
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--without-engine-bindings", action="store_true")
    parser.add_argument("--flashinfer-source-jit", action="store_true")
    parser.add_argument("--extra-package-path", action="append", default=[])
    args = parser.parse_args()
    if not args.layers or any(layer not in (2, 8, 14, 20) for layer in args.layers):
        parser.error("The probe admits only KV/index source layers 2, 8, 14 and 20")
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root))
    sys.path.extend(args.extra_package_path)
    if args.flashinfer_source_jit:
        sys.modules["flashinfer_jit_cache"] = None
    if args.without_engine_bindings:
        name = "rtp_llm.models_py.modules"
        package = types.ModuleType(name)
        package.__path__ = [str(root / "rtp_llm/models_py/modules")]
        package.__spec__ = importlib.machinery.ModuleSpec(name, None, is_package=True)
        sys.modules[name] = package

    import torch
    import torch.nn.functional as F
    import triton

    from rtp_llm.config.dsv41_config import V41Config
    from rtp_llm.models.dsv41_kv_cache import dsv41_page_layouts
    from rtp_llm.models_py.modules.dsv4.rope import precompute_freqs_cis
    from rtp_llm.models_py.modules.dsv4.v41_attention import V41SparseAttention
    from rtp_llm.models_py.modules.dsv4.v41_attention_projection import (
        build_v41_projections,
    )
    from rtp_llm.models_py.modules.dsv4.v41_cache import V41PagedCache
    from rtp_llm.models_py.modules.dsv4.v41_compressor import (
        V41PairMetadata,
        V41PairState,
    )
    from rtp_llm.models_py.modules.dsv4.v41_indexer import V41Indexer
    from rtp_llm.utils.model_weight import W
    from rtp_llm.utils.safetensors_header import named_safetensors_spans

    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0):
        raise RuntimeError("V4.1 attention probe requires SM120")
    config = V41Config.from_path(args.checkpoint)
    text = config.text
    shape = tuple(
        text[key]
        for key in (
            "hidden_size",
            "q_lora_rank",
            "o_lora_rank",
            "num_attention_heads",
            "o_groups",
            "head_dim",
            "index_n_heads",
            "index_head_dim",
            "qk_rope_head_dim",
        )
    )
    if shape != (5120, 1280, 1024, 64, 8, 512, 32, 128, 64):
        raise ValueError("Checkpoint geometry differs from the qualified Flash profile")
    layer_sources = config.layer_sources()
    manifest = {
        "checkpoint": str(args.checkpoint),
        "git_head": subprocess.check_output(
            ["git", "-c", f"safe.directory={root}", "rev-parse", "HEAD"],
            cwd=root,
            text=True,
        ).strip(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "triton": triton.__version__,
        "gpu_uuid": str(torch.cuda.get_device_properties(0).uuid),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "engine_bindings_bypassed": args.without_engine_bindings,
        "flashinfer_source_jit": args.flashinfer_source_jit,
        "allow_bf16_reduced_precision_reduction": torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
        "input_seed": 424,
        "qualification": "selected attention weights, four synthetic normalized token rows; no model inference",
        "layers": {},
    }
    cpu_freq = precompute_freqs_cis(
        64,
        8,
        text["rope_scaling"]["original_max_position_embeddings"],
        text["compress_rope_theta"],
        text["rope_scaling"]["factor"],
        text["rope_scaling"]["beta_fast"],
        text["rope_scaling"]["beta_slow"],
    )

    def error_metrics(actual, reference):
        value, expected = actual.float(), reference.float()
        error = value - expected
        return {
            "relative_rms_error": float(
                error.square().mean().sqrt()
                / expected.square().mean().sqrt().clamp_min(1e-12)
            ),
            "max_absolute_error": float(error.abs().max()),
            "scaled_max_error": float(
                error.abs().max() / expected.abs().max().clamp_min(1.0)
            ),
            "output_absmax": float(value.abs().max()),
        }

    for layer in args.layers:
        print(f"Admitting L{layer} attention headers", flush=True)
        sources = layer_sources[layer]
        ratio = sources.compress_ratio
        expected, names = {}, {}
        prefix = f"layers.{layer}.attn."
        for ckpt, weight_key, scale_key, n, k in (
            ("wq_a", W.v4_attn_wq_a_w, W.v4_attn_wq_a_s, 1280, 5120),
            ("wq_b", W.v4_attn_wq_b_w, W.v4_attn_wq_b_s, 32768, 1280),
            ("wkv", W.v4_attn_wkv_w, W.v4_attn_wkv_s, 512, 5120),
            ("wo_a", W.v4_attn_wo_a_w, W.v4_attn_wo_a_s, 8192, 4096),
            ("wo_b", W.v4_attn_wo_b_w, W.v4_attn_wo_b_s, 5120, 8192),
            ("indexer.wq_b", W.v4_indexer_wq_b_w, W.v4_indexer_wq_b_s, 4096, 1280),
        ):
            expected[prefix + ckpt + ".weight"] = ((n, k), "F8_E4M3")
            expected[prefix + ckpt + ".scale"] = ((n // 32, k // 32), "F8_E8M0")
            names[prefix + ckpt + ".weight"], names[prefix + ckpt + ".scale"] = (
                weight_key,
                scale_key,
            )
        for suffix, key, dims, dtype in (
            ("q_norm.weight", W.v4_attn_q_norm, (1280,), "BF16"),
            ("kv_norm.weight", W.v4_attn_kv_norm, (512,), "BF16"),
            ("attn_sink", W.v4_attn_sink, (64,), "F32"),
            ("compressor.wkv.weight", W.v4_compressor_wkv, (512, 5120), "BF16"),
            ("compressor.norm.weight", W.v4_compressor_norm, (512,), "BF16"),
            ("indexer.wk.weight", W.v41_indexer_wk, (128, 512), "BF16"),
            ("indexer.k_norm.weight", W.v41_indexer_k_norm, (128,), "BF16"),
            (
                "indexer.weights_proj.weight",
                W.v4_indexer_weights_proj_w,
                (32, 5120),
                "BF16",
            ),
        ):
            expected[prefix + suffix], names[prefix + suffix] = (dims, dtype), key
        if ratio == 2:
            expected[prefix + "compressor.wgate.weight"] = ((512, 5120), "BF16")
            names[prefix + "compressor.wgate.weight"] = W.v4_compressor_wgate
        spans = named_safetensors_spans(args.checkpoint, expected)
        byte_count = sum(span.size for span in spans.values())
        if byte_count > 256 * 1024**2:
            raise ValueError(
                "Attention probe exceeds its per-layer 256 MiB weight budget"
            )
        weights, hashes = {}, {}
        for name, span in spans.items():
            raw = torch.empty(span.size, dtype=torch.uint8)
            span.read_into(raw.numpy())
            hashes[name] = hashlib.sha256(memoryview(raw.numpy())).hexdigest()
            dtype = {
                "F8_E4M3": torch.float8_e4m3fn,
                "F8_E8M0": torch.uint8,
                "BF16": torch.bfloat16,
                "F32": torch.float32,
            }[span.dtype]
            weights[names[name]] = raw.view(dtype).reshape(span.shape)
            if (span.dtype == "F8_E8M0" and bool((raw == 255).any())) or not bool(
                torch.isfinite(weights[names[name]].float()).all()
            ):
                raise ValueError(f"Nonfinite checkpoint attention tensor {name}")
        print(f"L{layer}: loaded and hashed {byte_count:,} bytes", flush=True)
        x = torch.randn(
            5, 5120, generator=torch.Generator().manual_seed(424)
        ).bfloat16()
        x[-1] = float("nan")

        def run(device):
            backend = "eager" if device == "cpu" else "triton"
            op, kv = build_v41_projections(
                sources,
                {key: value.to(device) for key, value in weights.items()},
                cpu_freq.to(device),
                backend="eager" if device == "cpu" else "flashinfer",
            )
            hidden = x.to(device)
            meta = V41PairMetadata(
                torch.tensor([0, 0, 0, 0, -999], device=device),
                torch.tensor([0, 1, 2, 3, -999], device=device),
                torch.tensor([True, True, True, True, False], device=device),
                torch.tensor([-1, 0, -1, 2, 99999], device=device),
                torch.full((5,), -1, dtype=torch.int64, device=device),
                torch.tensor([0, 1, 2, 3, 99999], device=device),
            )
            state = V41PairState(torch.zeros(1, 4096, device=device))
            q = op.prepare(hidden, meta.positions, meta.valid_rows)
            pair_args = (
                {"pair_state": state, "pair_metadata": meta} if ratio == 2 else {}
            )
            produced = kv.prepare(hidden, meta.positions, meta.valid_rows, **pair_args)

            def cache(tag):
                return V41PagedCache(
                    torch.zeros(
                        2,
                        dsv41_page_layouts(1)[tag].stride_bytes,
                        device=device,
                        dtype=torch.uint8,
                    ),
                    tag,
                )

            swa, global_kv, index_k = (
                cache("swa_kv"),
                cache("csa_kv" if ratio == 2 else "hca_kv"),
                cache("indexer_kv"),
            )
            logical = torch.tensor(
                [0, 0, 1, 1, -999] if ratio == 2 else [0, 1, 2, 3, -999], device=device
            )
            produced.publish(
                swa,
                torch.arange(5, device=device),
                global_cache=global_kv,
                global_slots=logical + global_kv.layout.entries,
                index_cache=index_k,
                index_slots=logical + index_k.layout.entries,
            )
            indexer = V41Indexer(sources, max_positions=4, backend=backend)
            result = indexer(
                q.index_query,
                q.index_weights,
                index_k,
                torch.ones(5, 1, device=device, dtype=torch.int32),
                meta.positions,
                valid_rows=meta.valid_rows,
            )
            swa_reads = torch.full((5, 128), -1, device=device, dtype=torch.int64)
            for row in range(4):
                swa_reads[row, : row + 1] = torch.arange(row + 1, device=device)
            slots = torch.where(
                result.topk >= 0, result.topk.long() + global_kv.layout.entries, -1
            )
            value = V41SparseAttention(backend=backend)(
                q.query,
                op.sinks,
                swa,
                swa_reads,
                valid_rows=meta.valid_rows,
                global_cache=global_kv,
                global_slots=slots,
            )
            output = op.output(value, q)
            diagnostic = {}
            if device == "cuda":
                # Hold quantization and key selection fixed to isolate the attention kernel.
                common = V41SparseAttention(backend="eager")(
                    q.query.cpu(),
                    op.sinks.cpu(),
                    V41PagedCache(swa.raw.cpu(), "swa_kv"),
                    swa_reads.cpu(),
                    valid_rows=meta.valid_rows.cpu(),
                    global_cache=V41PagedCache(global_kv.raw.cpu(), global_kv.tag),
                    global_slots=slots.cpu(),
                )
                torch.testing.assert_close(value.cpu(), common, atol=0.02, rtol=0.02)
                diagnostic["attention_identical_query_and_cache"] = error_metrics(
                    value.cpu(), common
                )
                if ratio == 1:
                    cpu_bf16 = F.linear(x[:4], weights[W.v4_compressor_wkv])
                    gpu_bf16 = F.linear(hidden[:4], kv.compressor_weight).cpu()
                    cpu_fp32 = F.linear(
                        x[:4].float(), weights[W.v4_compressor_wkv].float()
                    ).bfloat16()
                    diagnostic["bf16_compressor_cpu_vs_cuda"] = error_metrics(
                        gpu_bf16, cpu_bf16
                    )
                    diagnostic["cpu_bf16_vs_fp32_rounded"] = error_metrics(
                        cpu_bf16, cpu_fp32
                    )
                    diagnostic["cuda_bf16_vs_cpu_fp32_rounded"] = error_metrics(
                        gpu_bf16, cpu_fp32
                    )
                    reduction = (
                        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
                    )
                    try:
                        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = (
                            False
                        )
                        full_reduction = F.linear(
                            hidden[:4], kv.compressor_weight
                        ).cpu()
                    finally:
                        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = (
                            reduction
                        )
                    diagnostic["cuda_full_reduction_vs_cpu_fp32_rounded"] = (
                        error_metrics(full_reduction, cpu_fp32)
                    )
            outputs = {
                "query": q.query.cpu(),
                "index_query": q.index_query.cpu(),
                "swa": produced.swa.cpu(),
                "global": produced.global_kv.cpu(),
                "index_key": produced.index_k.cpu(),
                "attention": value.cpu(),
                "output": output.cpu(),
                "topk": result.topk.cpu(),
            }
            packed = global_kv.gather(
                torch.where(produced.complete, logical + global_kv.layout.entries, -1),
                backend=backend,
            ).cpu()
            return outputs, diagnostic, packed

        reference, _, reference_bytes = run("cpu")
        actual, diagnostic, actual_bytes = run("cuda")
        torch.testing.assert_close(actual["topk"], reference["topk"], atol=0, rtol=0)
        metrics = {}
        for name in actual.keys() - {"topk"}:
            value = actual[name].float()
            metrics[name] = error_metrics(value, reference[name])
            if (
                not bool(torch.isfinite(value).all())
                or metrics[name]["relative_rms_error"] > 0.04
                or metrics[name]["scaled_max_error"] > 0.1
            ):
                raise AssertionError(
                    f"L{layer} {name} failed selected-weight comparison: {metrics[name]}"
                )
            if not bool((value[-1] == 0).all()):
                raise AssertionError(f"L{layer} {name} failed dead-row masking")
        manifest["layers"][str(layer)] = {
            "bytes_read": byte_count,
            "tensor_sha256": hashes,
            "metrics": metrics,
            "topk_exact": True,
            "diagnostic": diagnostic,
            "global_packed_byte_differences": int(
                (actual_bytes != reference_bytes).sum()
            ),
        }
        print(
            json.dumps({"layer": layer, "metrics": metrics, "diagnostic": diagnostic}),
            flush=True,
        )
    manifest["passed"] = True
    manifest["source_sha256"] = {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for pattern in ("v41_*.py", "_v41_*.py", "fp8/_v41_*.py", "rope.py")
        for path in (root / "rtp_llm/models_py/modules/dsv4").glob(pattern)
    }
    manifest["probe_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    print("Manifest", args.manifest, flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Real RTP Engram model-hook integration, using a small controlled shell.

Runs DeepSeekV41Model's actual loading/hash hooks and actual Engram.forward,
V41MXFP8Linear, RTP native quantization, DeepGEMM and gated residual. There is no
attention/MoE/scheduler/HTTP runtime in this shell. This is neither full-LLM
acceptance nor a latency benchmark. No native/projection shim is permitted.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.machinery
import importlib.util
import json
import math
import os
from pathlib import Path, PurePosixPath
import socket
import sys
from types import SimpleNamespace


REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
FRAMEWORK = Path(__file__).resolve().parent
if str(FRAMEWORK) not in sys.path:
    sys.path.insert(0, str(FRAMEWORK))

BF16_ULPS = 2
ABSOLUTE_FLOOR = 1e-5


def emit(stage, **values):
    print(json.dumps(dict(stage=stage, **values), sort_keys=True, allow_nan=False), flush=True)


def _digest(raw):
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def bf16_bound(reference):
    """Two BF16 ULPs at the reference exponent, with a small near-zero floor.

    A BF16 value has seven fraction bits. frexp returns exponent e, making its
    normal-value ULP 2**(e-8). The floor covers FP32 reduction/transcendental
    cancellation near zero; it is fixed before observing any device results.
    This is a small-fixture correctness bound, not an LLM accuracy guarantee.
    """
    if not math.isfinite(reference):
        raise ValueError("non-finite reference")
    if reference == 0:
        return ABSOLUTE_FLOOR
    exponent = math.frexp(abs(reference))[1]
    return max(ABSOLUTE_FLOOR, BF16_ULPS * math.ldexp(1.0, exponent - 8))


def compare_bf16(actual, expected, label):
    """CPU comparison with explicit per-element bounds and finite checks."""
    import torch

    if actual.dtype != torch.bfloat16 or expected.dtype != torch.bfloat16:
        raise AssertionError(f"{label}: output dtype must remain BF16")
    if tuple(actual.shape) != tuple(expected.shape):
        raise AssertionError(f"{label}: output shape mismatch")
    actual, expected = actual.detach().cpu().float(), expected.detach().cpu().float()
    if not torch.isfinite(actual).all() or not torch.isfinite(expected).all():
        raise AssertionError(f"{label}: non-finite output")
    if not expected.numel():
        return dict(elements=0, max_abs_error=0.0, max_bound_ratio=0.0,
                    accepted_bf16_ulps=BF16_ULPS, absolute_floor=ABSOLUTE_FLOOR)
    _, exponents = torch.frexp(expected.abs())
    bounds = torch.ldexp(torch.full_like(expected, float(BF16_ULPS)), exponents - 8)
    bounds = torch.where(expected == 0, torch.zeros_like(bounds), bounds)
    bounds = bounds.clamp_min(ABSOLUTE_FLOOR)
    difference = (actual - expected).abs()
    ratio = difference / bounds
    result = dict(elements=expected.numel(), max_abs_error=float(difference.max()),
                  max_bound_ratio=float(ratio.max()), accepted_bf16_ulps=BF16_ULPS,
                  absolute_floor=ABSOLUTE_FLOOR)
    if bool((difference > bounds).any()):
        raise AssertionError(f"{label}: BF16 correctness bound exceeded: {result}")
    return result


def check_consumer_fixture(root):
    """Inspect only small consumer files; reject a consumer that owns full tables."""
    root = Path(root)
    if not root.is_absolute() or not root.is_dir():
        raise ValueError("fixture root must be an existing absolute directory")
    if (root / "owner_checkpoint").exists():
        raise ValueError("OWNER_CHECKPOINT_PRESENT_IN_CONSUMER_FIXTURE")
    metadata = json.loads((root / "fixture.json").read_text())
    if metadata.get("fixture_component_only") is not True:
        raise ValueError("expected the explicitly marked Engram component fixture")
    selected = {
        name: checksum for name, checksum in metadata["files_sha256"].items()
        if name.startswith("consumer_checkpoint/") or name in ("config.json", "oracle_inputs.safetensors")
    }
    if "oracle_inputs.safetensors" not in selected or not any(name.startswith("consumer_checkpoint/") for name in selected):
        raise ValueError("consumer evidence files missing")
    for name, expected in selected.items():
        relative = PurePosixPath(name)
        if relative.is_absolute() or any(part in (".", "..") for part in relative.parts):
            raise ValueError("invalid fixture relative path")
        path = root / relative
        if not path.resolve(strict=True).is_relative_to(root.resolve()):
            raise ValueError("fixture file escapes root")
        if not path.is_file() or path.stat().st_size > 16 << 20:
            raise ValueError("expected bounded fixture files")
        if _digest(path.read_bytes()) != expected:
            raise ValueError("fixture file digest mismatch")
    checkpoint = root / "consumer_checkpoint"
    index = json.loads((checkpoint / "model.safetensors.index.json").read_text())["weight_map"]
    expected_keys = {
        f"layers.{layer}.engram.{suffix}"
        for layer in (0, 1) for suffix in ("wkv.weight", "wkv.scale", "q_weight", "k_weight")
    }
    if set(index) != expected_keys:
        raise ValueError("consumer checkpoint must contain only eight WKV/q/k entries")
    if {path.name for path in checkpoint.rglob("*.safetensors")} != set(index.values()):
        raise ValueError("consumer checkpoint contains undeclared tensor files")
    if any("embedding" in path.name for path in checkpoint.rglob("*")):
        raise ValueError("consumer contains an embedding checkpoint file")
    config = json.loads((checkpoint / "config.json").read_text())
    config["checkpoint_path"] = str(checkpoint)  # Relocation is not a model semantic change.
    return config, metadata


def require_real_native():
    """Bounded preflight avoids RTP's recursive missing-library fallback search."""
    config_name = "libth_transformer_config.so"
    config_path = REPO / "rtp_llm" / "libs" / config_name
    if not config_path.is_file():
        # Match the bounded locations checked by rtp_llm.ops before its
        # recursive fallback. PYTHONPATH alone does not satisfy that check.
        config_path = None
        level = [REPO / "bazel-bin"]
        for depth in range(3):
            for directory in level:
                candidate = directory / config_name
                if candidate.is_file():
                    config_path = candidate
                    break
            if config_path is not None:
                break
            if depth < 2:
                level = [child for directory in level if directory.is_dir()
                         for child in sorted(directory.iterdir()) if child.is_dir()]
    if config_path is None:
        raise RuntimeError(
            "REAL_RTP_NATIVE_REQUIRED: bounded config library missing; place "
            "libth_transformer_config.so in rtp_llm/libs or bazel-bin (at most two directory levels)"
        )
    directory = str(config_path.parent)
    if directory not in sys.path:
        sys.path.insert(0, directory)
    evidence = {}
    for name in ("libth_transformer_config", "librtp_compute_ops"):
        spec = importlib.util.find_spec(name)
        if (spec is None or not spec.origin
                or not isinstance(spec.loader, importlib.machinery.ExtensionFileLoader)):
            raise RuntimeError(f"REAL_RTP_NATIVE_REQUIRED: {name}; build main native targets before this harness")
        if name == "libth_transformer_config" and Path(spec.origin).resolve() != config_path.resolve():
            raise RuntimeError("REAL_RTP_NATIVE_REQUIRED: imported config does not match the bounded repo library")
        module = importlib.import_module(name)
        origin = Path(module.__file__).resolve()
        evidence[name] = dict(path=str(origin), size_bytes=origin.stat().st_size)
    return evidence


def selected_rows_reference(layer_id, ids):
    """CPU oracle generates ONLY selected deterministic fixture rows."""
    import torch
    from checkpoint_fixture import table_rows

    weights, scales = table_rows(layer_id, ids)
    return (weights.float() * scales.float().repeat_interleave(32, dim=-1)).to(torch.bfloat16)


def load_cpu_projection(checkpoint, layer_id):
    """Read consumer-local small WKV/q/k; never open the owner's checkpoint."""
    from safetensors import safe_open
    from rtp_llm.models_py.modules.dsv41.engram import _ReferenceWKV, dequantize_block_weight

    index = json.loads((checkpoint / "model.safetensors.index.json").read_text())["weight_map"]
    values = {}
    for suffix in ("wkv.weight", "wkv.scale", "q_weight", "k_weight"):
        name = f"layers.{layer_id}.engram.{suffix}"
        with safe_open(checkpoint / index[name], framework="pt", device="cpu") as file:
            values[suffix] = file.get_tensor(name)
    projection = _ReferenceWKV(dequantize_block_weight(values["wkv.weight"], values["wkv.scale"]))
    return projection, values["q_weight"], values["k_weight"]


def run(fixture_root, shard_json, device_ordinal=0):
    config, metadata = check_consumer_fixture(fixture_root)
    shard_json = [Path(item) for item in shard_json]
    if not shard_json or any(not item.is_absolute() or not item.is_file() for item in shard_json):
        raise ValueError("shard-json must be existing absolute files")
    if type(device_ordinal) is not int or device_ordinal < 0:
        raise ValueError("device must be a nonnegative local CUDA ordinal")
    import torch
    # Native RTP extensions depend on libtorch symbols loaded by this import.
    # Bind the authorized device before dependency imports can initialize CUDA.
    if not torch.cuda.is_available() or device_ordinal >= torch.cuda.device_count():
        raise RuntimeError("AUTHORIZED_CUDA_DEVICE_REQUIRED")
    torch.cuda.set_device(device_ordinal)
    device = torch.device("cuda", device_ordinal)
    native_evidence = require_real_native()
    from safetensors.torch import load_file
    from rtp_llm import ops
    ops.ensure_compute_ops_loaded()
    from rtp_llm.models_py.model_desc.deepseek_v41_model import DeepSeekV41Model
    from rtp_llm.models_py.modules.dsv41.engram import gated_engram_residual
    from rtp_llm.models_py.modules.dsv41.engram_bootstrap import stage_package
    from rtp_llm.models_py.modules.dsv41.engram_model_attach import EngramModelInitializationError
    from rtp_llm.models_py.modules.dsv41.engram_startup import package_from_descriptors
    from rtp_llm.models_py.modules.dsv41.utils import V41MXFP8Linear
    from checkpoint_fixture import HC, HIDDEN_DIM, LAYERS, WKV_N, make_inputs

    stage_package(package_from_descriptors([str(item) for item in shard_json],
                                            str(Path(fixture_root) / "consumer_checkpoint")))
    shell = SimpleNamespace(
        _v4_args=SimpleNamespace(v41_config=config),
        v4=SimpleNamespace(
            embed=torch.nn.Embedding(config["vocab_size"], HIDDEN_DIM, device=device, dtype=torch.bfloat16),
            layers=[SimpleNamespace(engram=None) for _ in LAYERS],
        ),
        py_hw_kernel_config=SimpleNamespace(enable_cuda_graph=False),
        _engram_hash_state=None, _engram_layers=(), _image_plan=None,
    )
    emit("runtime", hostname=socket.gethostname(), pid=os.getpid(),
         device_ordinal=device_ordinal, torch=torch.__version__,
         cuda=torch.version.cuda, native=native_evidence,
         scope="real Engram model hooks and layer forward; no full LLM or HTTP")
    with torch.inference_mode():
        DeepSeekV41Model._load_extra_weights(shell, None)
        storage = shell._engram_storage
        if storage is None or storage.info()["state"] != "READY":
            raise AssertionError("real model hook did not establish a READY external storage")
        for layer in shell.v4.layers:
            if not isinstance(layer.engram.wkv, V41MXFP8Linear):
                raise AssertionError("production V41MXFP8Linear is required")
            embedding = layer.engram.embed_tokens
            if embedding.weight is not None or embedding.scales is not None:
                raise AssertionError("consumer unexpectedly holds a private full table")
        pointers = tuple(tuple(view.data_ptr() for view in layer.engram.embed_tokens._uva)
                         for layer in shell.v4.layers)
        info = storage.info()
        tensors = {tensor.name: tensor for tensor in storage.validated.manifest.tensors}
        for layer_id, _ in LAYERS:
            for role, view in zip(("weight", "scale"), shell.v4.layers[layer_id].engram.embed_tokens._uva):
                part = tensors[f"layers.{layer_id}.engram.embed.{role}"].shards[0]
                expected_address = info["allocations"][part.allocation_id]["base_address"] + part.byte_offset
                if view.data_ptr() != expected_address:
                    raise AssertionError("model table pointer is not the imported native allocation view")
        # Re-enter the exact loading hook: it must reuse generation/addresses.
        old_modules = tuple(layer.engram for layer in shell.v4.layers)
        DeepSeekV41Model._load_extra_weights(shell, None)
        if (shell._engram_storage is not storage
                or tuple(layer.engram for layer in shell.v4.layers) != old_modules
                or tuple(tuple(view.data_ptr() for view in layer.engram.embed_tokens._uva)
                         for layer in shell.v4.layers) != pointers):
            raise AssertionError("second model initialization changed the external binding")
        try:
            DeepSeekV41Model.prepare_fmha_impl(shell, None, is_cuda_graph=True)
        except EngramModelInitializationError as error:
            if error.code != "EXTERNAL_ENGRAM_GRAPH_UNSUPPORTED":
                raise
        else:
            raise AssertionError("unmanaged engine graph was not rejected")
        emit("model_attached", dataset_id=info["dataset_id"], generation=info["generation"],
             allocation_count=info["allocation_count"], physical_bytes=info["physical_bytes"],
             layers=[layer for layer, _ in shell._engram_layers],
             repeated_hook_reused_mapping=True, graph_mode_rejected=True,
             consumer_checkpoint_has_tables=False, pointers_match_native=True)
        oracle = load_file(str(Path(fixture_root) / "oracle_inputs.safetensors"), device="cpu")
        projections = {lid: load_cpu_projection(Path(config["checkpoint_path"]), lid) for lid, _ in LAYERS}
        records = []
        for case in metadata["cases"]:
            label = case["name"]
            expected_inputs = make_inputs(case["batch_size"], case["step"])
            for key, expected in expected_inputs.items():
                torch.testing.assert_close(oracle[f"{label}.{key}"], expected, rtol=0, atol=0)
            windows = oracle[f"{label}.engram_token_windows"]
            hidden = oracle[f"{label}.hidden"]
            mask = oracle[f"{label}.token_mask"]
            expected_hashes = oracle[f"{label}.hash_ids"]
            DeepSeekV41Model._prepare_engram(shell, SimpleNamespace(engram_token_windows=windows))
            for layer_id, hash_index in shell._engram_layers:
                layer = shell.v4.layers[layer_id]
                ids, gpu_mask = layer.engram_hashes, layer.engram_token_mask
                torch.testing.assert_close(ids.cpu(), expected_hashes[:, hash_index], rtol=0, atol=0)
                torch.testing.assert_close(gpu_mask.cpu(), mask, rtol=0, atol=0)
                expected_rows = selected_rows_reference(layer_id, expected_hashes[:, hash_index])
                actual_rows = layer.engram.embed_tokens(ids, device)
                torch.testing.assert_close(actual_rows.cpu(), expected_rows, rtol=0, atol=0)
                projection, q_weight, k_weight = projections[layer_id]
                if hidden.shape[0]:
                    expected_kv = projection(expected_rows.flatten(-2))
                else:
                    expected_kv = torch.empty((0, WKV_N), dtype=torch.bfloat16)
                actual_kv = layer.engram.wkv(actual_rows.flatten(-2))
                kv_result = compare_bf16(actual_kv, expected_kv, f"{label}/layer{layer_id}/WKV")
                expected_output = (gated_engram_residual(
                    hidden, expected_kv, q_weight, k_weight, config["rms_norm_eps"], mask
                ) if hidden.shape[0] else hidden)
                # This is the unmodified production layer forward, not a shim.
                actual_output = layer.engram(hidden.to(device), ids, gpu_mask)
                output_result = compare_bf16(actual_output, expected_output, f"{label}/layer{layer_id}/forward")
                if bool((~mask).any()):
                    torch.testing.assert_close(actual_output.cpu()[~mask], hidden[~mask], rtol=0, atol=0)
                record = dict(case=label, layer=layer_id, batch=hidden.shape[0],
                              hash_exact=True, mask_exact=True, lookup_exact=True,
                              wkv=kv_result, forward=output_result)
                records.append(record)
                emit("case_pass", **record)
            for layer in shell.v4.layers:
                layer.engram_hashes = None
                layer.engram_token_mask = None
        torch.cuda.synchronize(device)
        emit("consumer_complete", passed=True, layer_cases=len(records),
             dataset_id=info["dataset_id"], generation=info["generation"],
             model_hook="DeepSeekV41Model._load_extra_weights/_prepare_engram",
             real_projection="V41MXFP8Linear/native RTP quant/DeepGEMM",
             full_llm=False, http_serving=False, performance_claim=False,
             teardown="mapping retained until consumer process exit; owner must outlive process")
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture-root", type=Path, required=True)
    parser.add_argument("--shard-json", type=Path, action="append", required=True)
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()
    try:
        run(args.fixture_root, args.shard_json, args.device)
    except BaseException as error:
        emit("consumer_failed", error_type=type(error).__name__, message=str(error)[:1024],
             full_llm=False, cleanup_boundary="consumer process exit; do not release owner early")
        raise


if __name__ == "__main__":
    main()

"""Isolate retained V4.1 FFN arithmetic through maintained shared/routed owners.

Inputs and routes are fixed official tensors. GPU-only hooks retain boundaries;
CPU comparisons happen after forwards. This does not measure model accuracy.
"""

import argparse
import hashlib
import importlib
import json
import subprocess
import sys
from contextlib import ExitStack, contextmanager
from pathlib import Path
from unittest.mock import patch

import torch
import torch.nn.functional as F

from rtp_llm.models_py.modules.dsv4 import v41_routed
from rtp_llm.models_py.modules.dsv4.moe import expert as expert_module
from rtp_llm.models_py.modules.dsv4.moe.shared_expert import W13SharedExpert
from rtp_llm.models_py.modules.dsv4.v41_moe import V41LocalMoE
from rtp_llm.models_py.modules.dsv4.v41_mxfp8 import quantize_group32
from rtp_llm.tools.dsv41_attention_replay import NASReader, history_identities
from rtp_llm.tools.dsv41_reference_probe import model_args
from rtp_llm.tools.dsv41_trace_compare import tensor_difference


def difference(left, right):
    return tensor_difference(left.cpu(), right.cpu())


def digest(tensor):
    raw = tensor.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def byte_difference(left, right):
    if left.shape != right.shape or left.dtype != right.dtype:
        raise ValueError("Byte comparison requires identical shape and dtype")
    a, b = (
        left.contiguous().view(torch.uint8).cpu(),
        right.contiguous().view(torch.uint8).cpu(),
    )
    return {
        "shape": list(left.shape),
        "dtype": str(left.dtype),
        "different_bytes": int((a != b).sum()),
    }


@contextmanager
def capture_linears(owner, *, merged=False, native_activation=False):
    """Clones only; no transfer, scalar read or synchronization in hooks."""
    captured = {}

    def output(name):
        def retain(_module, _args, value):
            captured[name] = value.clone()

        return retain

    def input_hook(_module, args):
        captured["w2.input"] = args[0].clone()

    with ExitStack() as stack:
        for name in (("w13",) if merged else ("w1", "w3")) + ("w2",):
            handle = getattr(owner, name).register_forward_hook(
                output(name + ".output")
            )
            stack.callback(handle.remove)
        handle = owner.w2.register_forward_pre_hook(input_hook)
        stack.callback(handle.remove)
        if native_activation:
            original = expert_module.silu_mul_split

            def activation(*args, **kwargs):
                result = original(*args, **kwargs)
                captured["activation.fp32"] = result.clone()
                return result

            stack.enter_context(
                patch.object(expert_module, "silu_mul_split", activation)
            )
        yield captured


@contextmanager
def capture_grouped(owner):
    captured = {"gemms": []}
    original_gemm, original_quant = owner._masked, v41_routed.quantize_swiglu

    def gemm(*args, **kwargs):
        result = original_gemm(*args, **kwargs)
        captured["gemms"].append(args[2].clone())
        return result

    def quant(gate, up, valid, **kwargs):
        result = original_quant(gate, up, valid, **kwargs)
        captured["quantized_values"] = result[0].clone()
        captured["quantized_scales"] = (
            result[1].reshape(*valid.shape, gate.shape[-1] // 32).clone()
        )
        return result

    with patch.object(owner, "_masked", gemm), patch.object(
        v41_routed, "quantize_swiglu", quant
    ):
        yield captured


def released_activation(gate, up, limit):
    if limit > 0:
        gate, up = gate.clamp(max=limit), up.clamp(-limit, limit)
    return F.silu(gate) * up


def shared_boundaries(native, reference, x):
    with capture_linears(reference) as official:
        expected = reference(x)
    with capture_linears(native, merged=True, native_activation=True) as captured:
        actual = native(x)
    gate, up = captured["w13.output"].float().chunk(2, -1)
    official_gate, official_up = (
        official["w1.output"].float(),
        official["w3.output"].float(),
    )
    limit = native.swiglu_limit
    expected_activation = released_activation(official_gate, official_up, limit)
    if not torch.equal(expected_activation.bfloat16(), official["w2.input"]):
        raise AssertionError(
            "Released shared activation replay changed its BF16 boundary"
        )
    torch_activation = released_activation(gate, up, limit)
    native_on_official = expert_module.require_silu_mul_split()(
        official_gate.contiguous(), official_up.contiguous(), clamp_limit=limit
    )
    candidates = {
        "native_projection_torch_activation": native.w2(torch_activation.bfloat16()),
        "official_projection_native_activation": native.w2(
            native_on_official.bfloat16()
        ),
        "official_hidden_native_w2": native.w2(official["w2.input"]),
        "native_hidden_official_w2": reference.w2(captured["w2.input"]),
    }
    clamp_gate, clamp_up = gate.clamp(max=limit), up.clamp(-limit, limit)
    official_clamp_gate = official_gate.clamp(max=limit)
    official_clamp_up = official_up.clamp(-limit, limit)
    # Retain the native SiLU result with unit up, outside all capture hooks.
    native_silu = expert_module.require_silu_mul_split()(
        clamp_gate.contiguous(), torch.ones_like(clamp_up), clamp_limit=0
    )
    return (
        expected,
        actual,
        candidates,
        {
            "gate_projection": difference(official_gate, gate),
            "up_projection": difference(official_up, up),
            "clamped_gate": difference(official_clamp_gate, clamp_gate),
            "clamped_up": difference(official_clamp_up, clamp_up),
            "silu_fp32": difference(F.silu(official_clamp_gate), native_silu),
            "activation_fp32": difference(
                expected_activation, captured["activation.fp32"]
            ),
            "w2_input_bf16": difference(official["w2.input"], captured["w2.input"]),
            "output_bf16": difference(expected, actual),
            "substitutions": {
                name: difference(expected, value) for name, value in candidates.items()
            },
        },
    )


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("reference_traces", type=Path)
    parser.add_argument("native_traces", type=Path)
    parser.add_argument("baseline_report", type=Path)
    parser.add_argument("report", type=Path)
    args = parser.parse_args()
    if args.report.exists():
        raise ValueError("FFN boundary report must be fresh")
    repo = Path(__file__).resolve().parents[2]
    revision = subprocess.check_output(
        ["git", "-c", f"safe.directory={repo}", "rev-parse", "HEAD"],
        cwd=repo,
        text=True,
    ).strip()
    if torch.cuda.get_device_capability() != (12, 0):
        raise RuntimeError("Retained FFN diagnostic requires SM120 under GPU lock")
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.bfloat16)
    torch.set_float32_matmul_precision("highest")
    root = args.checkpoint.resolve()
    sys.path.insert(0, str(root / "inference"))
    official = importlib.import_module("model")
    if Path(official.__file__).resolve() != root / "inference/model.py":
        raise ValueError("Unexpected released reference module")
    config = model_args(official, root, 256)
    baseline = json.loads(args.baseline_report.read_text())
    trace_files = [
        args.reference_traces / f"reference-prompt0-step{i}.pt" for i in range(3)
    ]
    native_files = [args.native_traces / f"native-prompt0-step{i}.pt" for i in range(3)]
    traces = [
        torch.load(path, map_location="cpu", weights_only=True) for path in trace_files
    ]
    native_traces = [
        torch.load(path, map_location="cpu", weights_only=True) for path in native_files
    ]
    identities = history_identities(traces, native_traces)
    reader, results, controls = NASReader(root), {}, 0

    def parameters(module, prefix):
        for name, parameter in module.named_parameters():
            value = reader.load(prefix + name)
            if parameter.element_size() == 1:
                parameter.view(torch.uint8).copy_(value.view(torch.uint8))
            else:
                parameter.copy_(value)

    for layer in range(3):
        prefix = f"layers.{layer}.ffn."
        with torch.device("cuda"):
            gate = official.Gate(layer, config)
            shared_reference = official.Expert(
                config.dim, config.moe_inter_dim, swiglu_limit=config.swiglu_limit
            )
        parameters(gate, prefix + "gate.")
        parameters(shared_reference, prefix + "shared_experts.")
        values = {
            (part, suffix): reader.load(prefix + f"shared_experts.{part}.{suffix}")
            for part in ("w1", "w3", "w2")
            for suffix in ("weight", "scale")
        }
        shared = W13SharedExpert(
            config.dim,
            config.moe_inter_dim,
            {
                "w13_w": torch.cat([values[p, "weight"] for p in ("w1", "w3")]),
                "w13_s": torch.cat([values[p, "scale"] for p in ("w1", "w3")]),
                "w2_w": values["w2", "weight"],
                "w2_s": values["w2", "scale"],
            },
            swiglu_limit=config.swiglu_limit,
        )
        frames = []
        for step in range(3):
            x = (
                traces[step]["tensors"][f"layer{layer:02d}.ffn.input"]
                .reshape(-1, config.dim)
                .cuda()
            )
            weights, ids = gate(x)
            frames.append((x, weights, ids))
        expert_ids = sorted(
            set(torch.cat([frame[2].flatten() for frame in frames]).tolist())
        )
        bank = {name: [] for name in ("w13_w", "w13_s", "w2_w", "w2_s")}
        routed_reference = [
            torch.zeros_like(frame[0], dtype=torch.float32) for frame in frames
        ]
        per_expert = [dict() for _ in frames]
        for expert_id in expert_ids:
            with torch.device("cuda"):
                expert = official.Expert(
                    config.dim,
                    config.moe_inter_dim,
                    dtype=torch.float4_e2m1fn_x2,
                    swiglu_limit=config.swiglu_limit,
                )
            parameters(expert, prefix + f"experts.{expert_id}.")
            for suffix, field in (("w", "weight"), ("s", "scale")):
                bank[f"w13_{suffix}"].append(
                    torch.cat(
                        [
                            getattr(getattr(expert, part), field).view(torch.uint8)
                            for part in ("w1", "w3")
                        ]
                    )
                )
                bank[f"w2_{suffix}"].append(getattr(expert.w2, field).view(torch.uint8))
            for step, (x, weights, ids) in enumerate(frames):
                rows, top = torch.where(ids == expert_id)
                if rows.numel():
                    with capture_linears(expert) as captured:
                        output = expert(x[rows], weights[rows, top, None])
                    routed_reference[step][rows] += output
                    per_expert[step][expert_id] = (rows, captured)
        routed = V41LocalMoE(
            v41_routed.V41RoutedExperts(
                {name: torch.stack(rows) for name, rows in bank.items()}
            ),
            max_tokens=128,
            masked_token_limit=128,
            max_topk=6,
        )
        mapping = torch.full(
            (config.n_routed_experts,), -1, dtype=torch.int64, device="cuda"
        )
        mapping[torch.tensor(expert_ids, device="cuda")] = torch.arange(
            len(expert_ids), device="cuda"
        )
        layer_results = []
        for step, (x, weights, ids) in enumerate(frames):
            protected = {"x": digest(x), "weights": digest(weights), "ids": digest(ids)}
            expected_shared, actual_shared, candidates, shared_report = (
                shared_boundaries(shared, shared_reference, x)
            )
            expected = (routed_reference[step] + expected_shared.float()).bfloat16()
            with capture_grouped(routed.experts) as grouped:
                actual_routed = routed(
                    x, weights, mapping[ids].contiguous(), layout="masked"
                )
            if len(grouped["gemms"]) != 2:
                raise AssertionError("Expected grouped W13 and W2 captures")
            actual = (actual_routed + actual_shared.float()).bfloat16()
            captured_output = traces[step]["tensors"][
                f"layer{layer:02d}.ffn.output"
            ].reshape(-1, config.dim)
            measurements = {
                "reference_replay": difference(captured_output, expected),
                "shared": difference(expected_shared, actual_shared),
                "routed": difference(routed_reference[step], actual_routed),
                "ffn": difference(expected, actual),
            }
            previous = baseline[str(layer)][step]
            if ids.tolist() != previous["expert_ids"]:
                raise AssertionError("Fixed official route IDs changed")
            for name, measurement in measurements.items():
                if measurement != previous[name]:
                    raise AssertionError(
                        f"Saved baseline changed at L{layer}/step{step}/{name}"
                    )
                controls += 1
            expert_report = {}
            for local, expert_id in enumerate(expert_ids):
                if expert_id not in per_expert[step]:
                    continue
                rows, captured = per_expert[step][expert_id]
                count = rows.numel()
                projected = grouped["gemms"][0][local, :count].float()
                g, u = projected.chunk(2, -1)
                values, scales = quantize_group32(
                    captured["w2.input"], backend="triton"
                )
                expert_report[str(expert_id)] = {
                    "source_rows": rows.tolist(),
                    "gate_projection": difference(captured["w1.output"].float(), g),
                    "up_projection": difference(captured["w3.output"].float(), u),
                    "w2_input_fp8_bytes": byte_difference(
                        values, grouped["quantized_values"][local, :count]
                    ),
                    "w2_input_scale_bytes": byte_difference(
                        scales, grouped["quantized_scales"][local, :count]
                    ),
                    "output_bf16": difference(
                        captured["w2.output"], grouped["gemms"][1][local, :count]
                    ),
                }
            after = {"x": digest(x), "weights": digest(weights), "ids": digest(ids)}
            if after != protected:
                raise AssertionError("Replay mutated input or route bytes")
            record = {
                "step": step,
                "controls": measurements,
                "shared_boundaries": shared_report,
                "routed_experts": expert_report,
                "protected_bytes": protected,
                "ffn_substitutions": {
                    name: difference(
                        expected, (actual_routed + value.float()).bfloat16()
                    )
                    for name, value in candidates.items()
                },
            }
            layer_results.append(record)
            print(
                f"L{layer} step={step} shared_input_differences={shared_report['w2_input_bf16']['different_values']} "
                f"ffn_differences={measurements['ffn']['different_values']} "
                f"torch_activation_ffn_differences={record['ffn_substitutions']['native_projection_torch_activation']['different_values']}",
                flush=True,
            )
        results[str(layer)] = layer_results
    for name, value in reader.values.items():
        if digest(value) != reader.hashes[name]:
            raise AssertionError(f"Replay mutated source weight bytes: {name}")
    sources = [
        Path(__file__),
        repo / "rtp_llm/models_py/modules/dsv4/moe/shared_expert.py",
        repo / "rtp_llm/models_py/modules/dsv4/_silu_mul_split_triton.py",
        repo / "rtp_llm/models_py/modules/dsv4/v41_routed.py",
        repo / "rtp_llm/models_py/modules/dsv4/v41_expert.py",
        root / "inference/model.py",
        root / "inference/kernel.py",
        args.baseline_report,
        *trace_files,
        *native_files,
    ]
    report = {
        "revision": revision,
        "identities": identities,
        "baseline_controls_exact": controls,
        "source_sha256": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources
        },
        "weight_sha256": reader.hashes,
        "weight_bytes_unchanged": len(reader.hashes),
        "layers": results,
        "limitations": [
            "Fixed official inputs/routes; no upstream propagation, full-model accuracy or performance claim."
        ],
    }
    with args.report.open("x") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write("\n")


if __name__ == "__main__":
    main()

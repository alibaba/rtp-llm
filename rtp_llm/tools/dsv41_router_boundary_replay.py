"""Hold qualified FFN inputs and compare maintained/released router boundaries."""

import argparse
import hashlib
import importlib
import json
import sys
from pathlib import Path
from unittest.mock import patch

import torch
import torch.nn.functional as F

from rtp_llm.tools.dsv41_attention_replay import NASReader, difference
from rtp_llm.tools.dsv41_reference_probe import model_args
from rtp_llm.utils.model_weight import W


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def gate_difference(reference, candidate):
    if reference.dtype != torch.int64:
        return difference(reference, candidate)
    assert reference.shape == candidate.shape and candidate.dtype == torch.int64
    reference, candidate = reference.cpu(), candidate.cpu()
    ref_sorted = reference.sort(dim=-1).values
    candidate_sorted = candidate.sort(dim=-1).values
    membership = ref_sorted.unsqueeze(-1) == candidate_sorted.unsqueeze(-2)
    first = torch.ones((len(reference), 1), dtype=torch.bool)
    ref_unique = torch.cat((first, ref_sorted[:, 1:] != ref_sorted[:, :-1]), dim=-1)
    candidate_unique = torch.cat(
        (first, candidate_sorted[:, 1:] != candidate_sorted[:, :-1]), dim=-1
    )
    missing = int((~membership.any(-1) & ref_unique).sum())
    extra = int((~membership.any(-2) & candidate_unique).sum())
    ordered = torch.equal(reference, candidate)
    return {
        "shape": list(reference.shape),
        "reference_dtype": str(reference.dtype),
        "candidate_dtype": str(candidate.dtype),
        "bitwise_equal": ordered,
        "different_values": int((reference != candidate).sum()),
        "order_equal": ordered,
        "sets_equal": missing == extra == 0,
        "missing_ids": missing,
        "extra_ids": extra,
        "duplicate_reference_ids": int((ref_sorted[:, 1:] == ref_sorted[:, :-1]).sum()),
        "duplicate_candidate_ids": int(
            (candidate_sorted[:, 1:] == candidate_sorted[:, :-1]).sum()
        ),
    }


def captured_gate(owner, value):
    scores = []
    original = F.linear

    def retain(*args, **kwargs):
        result = original(*args, **kwargs)
        scores.append(result.clone())
        return result

    with patch.object(F, "linear", retain):
        weights, ids = owner(value)
    assert len(scores) == 1
    return {"scores": scores[0], "weights": weights.clone(), "ids": ids.clone()}


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "admission_manifest", "report"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--tensor-artifact", type=Path, required=True)
    cli = parser.parse_args()
    assert not cli.report.exists() and not cli.tensor_artifact.exists()
    assert torch.cuda.get_device_capability() == (12, 0)
    from rtp_llm.models_py.modules.dsv4.moe import gate as native_module
    from rtp_llm.tools.dsv41_corrected_history_replay import cpu_tree
    from rtp_llm.tools.dsv41_query_boundary_replay import require_bytes

    torch.set_num_threads(1)
    torch.set_default_dtype(torch.bfloat16)
    torch.set_float32_matmul_precision("highest")
    assert not torch.backends.cuda.matmul.allow_tf32
    manifest = json.loads(cli.admission_manifest.read_text())
    root = Path(manifest["repo_root"])
    for name, expected in manifest["source_sha256"].items():
        assert digest(root / name) == expected, name
    for field in ("asset_sha256", "helper_sha256"):
        for name, expected in manifest[field].items():
            assert digest(Path(name)) == expected, name
    packet_report = json.loads(Path(manifest["packet_report"]).read_text())
    assert (
        digest(Path(manifest["packet_artifact"]))
        == packet_report["tensor_artifact_sha256"]
    )
    packet = torch.load(
        manifest["packet_artifact"], map_location="cpu", weights_only=True
    )
    assert (
        packet_report["current_WKV_inputs_admitted"]
        and packet_report["cuda_uninitialized"]
    )
    assert (
        packet_report["parent_qualification_sha256"]
        == manifest["full_qualification_sha256"]
    )
    assert packet["profiles"] == packet_report["profiles"]
    assert len(packet["cases"]) == 4
    selected = [
        c for c in packet["cases"] if c["profile"] == manifest["packet_profile"]
    ]
    assert len(selected) == 3
    nas = cli.checkpoint.resolve()
    sys.path.insert(0, str(nas / "inference"))
    official = importlib.import_module("model")
    assert Path(official.__file__).resolve() == nas / "inference/model.py"
    args = model_args(official, nas, 256)
    assert (
        args.gate_temp == 1
        and args.norm_topk_prob
        and args.score_func == "sqrtsoftplus"
    )
    assert native_module._use_fused_gate(args.score_func, 1)
    reader, owners, weights_before = NASReader(nas), {}, {}
    for layer in sorted({c["layer"] for c in selected}):
        prefix = "layers.{}.ffn.gate.".format(layer)
        weight, bias = reader.load(prefix + "weight"), reader.load(prefix + "bias")
        experts, topk = args.get_moe_config(layer)
        native = native_module.Gate(
            layer,
            args.dim,
            experts,
            topk,
            args.score_func,
            args.route_scale,
            layer_weights={W.v4_router_w: weight, W.v4_router_bias: bias},
            fp32_logits=True,
        )
        with torch.device("cuda"):
            released = official.Gate(layer, args)
            released.weight.copy_(weight)
            released.bias.copy_(bias)
        owners[layer] = native, released
        weights_before[str(layer)] = {"weight": weight.clone(), "bias": bias.clone()}
    original_policy = native_module._use_fused_gate
    cases, records = [], []
    exact_candidates = unchanged_scores = id_controls = restoration_controls = 0
    for source in selected:
        native, released = owners[source["layer"]]
        value = source["input"].cuda().reshape(-1, args.dim).contiguous().clone()
        require_bytes(
            source["input"].reshape_as(value),
            source["input_control"].reshape_as(value),
            "admitted FFN input",
        )
        before = value.clone()
        reference = captured_gate(released, value)
        baseline = captured_gate(native, value)
        with patch.object(native_module, "_use_fused_gate", return_value=False):
            unfused = captured_gate(native, value)
        assert native_module._use_fused_gate is original_policy
        restored = captured_gate(native, value)
        require_bytes(before, value, "held router input")
        metrics = {
            mode + "." + field: gate_difference(reference[field], output[field])
            for mode, output in (
                ("native", baseline),
                ("unfused", unfused),
                ("restored", restored),
            )
            for field in ("scores", "weights", "ids")
        }
        for field in ("scores", "weights", "ids"):
            metric = gate_difference(baseline[field], restored[field])
            metrics["restoration." + field] = metric
            assert metric["bitwise_equal"]
            restoration_controls += 1
        for mode in ("native", "unfused", "restored"):
            assert metrics[mode + ".scores"]["bitwise_equal"]
            unchanged_scores += 1
        assert (
            metrics["native.ids"]["bitwise_equal"]
            and metrics["restored.ids"]["bitwise_equal"]
        )
        id_controls += 2
        exact_candidates += all(
            metrics["unfused." + field]["bitwise_equal"]
            for field in ("scores", "weights", "ids")
        )
        fields = {
            name: source[name]
            for name in ("profile", "prompt_index", "step", "layer", "identity")
        }
        records.append(dict(fields, metrics=metrics))
        cases.append(
            cpu_tree(
                dict(
                    fields,
                    input=before,
                    input_after=value.clone(),
                    native=baseline,
                    reference=reference,
                    unfused=unfused,
                    restored=restored,
                )
            )
        )
    after = {
        str(layer): {"weight": native.weight, "bias": native.bias}
        for layer, (native, _) in owners.items()
    }
    for layer in weights_before:
        for name, value in weights_before[layer].items():
            require_bytes(
                value, after[layer][name], "unchanged router weight " + layer + name
            )
    controls = {
        "held_cases": 3,
        "exact_unfused_cases": exact_candidates,
        "unchanged_score_controls": unchanged_scores,
        "native_id_controls": id_controls,
        "restoration_controls": restoration_controls,
        "NAS_weights": len(reader.hashes),
        "numerical_promotion": False,
    }
    assert (
        controls["NAS_weights"] == 4
        and unchanged_scores == 9
        and id_controls == 6
        and restoration_controls == 9
    )
    with cli.tensor_artifact.open("xb") as handle:
        torch.save(
            {
                "cases": cases,
                "weights_before": cpu_tree(weights_before),
                "weights_after": cpu_tree(after),
            },
            handle,
        )
    result = {
        "records": records,
        "controls": controls,
        "weight_sha256": reader.hashes,
        "packet_artifact_sha256": digest(Path(manifest["packet_artifact"])),
        "tensor_artifact_sha256": digest(cli.tensor_artifact),
        "replay_source_sha256": digest(Path(__file__).resolve()),
        "native_fused_policy": True,
        "current_WKV_inputs_admitted": True,
        "TF32_disabled": True,
        "numerical_promotion": False,
    }
    with cli.report.open("x") as handle:
        handle.write(
            json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n"
        )
    print("DSV41_ROUTER_BOUNDARY_CONTROLS " + json.dumps(controls), flush=True)
    assert exact_candidates == 3, "Unfused router mismatch preserved in raw evidence"


if __name__ == "__main__":
    main()

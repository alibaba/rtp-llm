"""Hold current NAS router inputs and qualify explicit eager selection and graphs."""

import argparse
import importlib
import json
import sys
from pathlib import Path

import torch

from rtp_llm.models_py.modules.dsv4.fp8.test import v41_router_eager_reference as scope
from rtp_llm.models_py.modules.dsv4.moe import gate as native_module
from rtp_llm.tools.dsv41_attention_replay import NASReader
from rtp_llm.tools.dsv41_corrected_history_replay import cpu_tree
from rtp_llm.tools.dsv41_query_boundary_replay import require_bytes
from rtp_llm.tools.dsv41_reference_probe import model_args
from rtp_llm.tools.dsv41_router_boundary_replay import (
    captured_gate,
    digest,
    gate_difference,
)
from rtp_llm.utils.model_weight import W

FIELDS = ("scores", "weights", "ids")


def compare(reference, candidate, mode):
    return {
        mode + "." + field: gate_difference(reference[field], candidate[field])
        for field in FIELDS
    }


def clone(output):
    return {key: value.clone() for key, value in output.items()}


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "admission_manifest", "report"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--tensor-artifact", type=Path, required=True)
    cli = parser.parse_args()
    assert not cli.report.exists() and not cli.tensor_artifact.exists()
    assert torch.cuda.get_device_capability() == (12, 0)
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
    packet = torch.load(
        manifest["packet_artifact"], map_location="cpu", weights_only=True
    )
    selected = [
        c for c in packet["cases"] if c["profile"] == manifest["packet_profile"]
    ]
    parent = torch.load(
        manifest["held_parent_artifact"], map_location="cpu", weights_only=True
    )
    assert len(selected) == len(parent["cases"]) == 3
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
    unselected = None
    for layer in sorted({c["layer"] for c in selected}):
        prefix = "layers.{}.ffn.gate.".format(layer)
        weight, bias = reader.load(prefix + "weight"), reader.load(prefix + "bias")
        experts, topk = args.get_moe_config(layer)

        def native():
            return native_module.Gate(
                layer,
                args.dim,
                experts,
                topk,
                args.score_func,
                args.route_scale,
                layer_weights={W.v4_router_w: weight, W.v4_router_bias: bias},
                fp32_logits=True,
            )

        owner = native()
        if unselected is None:
            unselected = native()
        with torch.device("cuda"):
            released = official.Gate(layer, args)
            released.weight.copy_(weight)
            released.bias.copy_(bias)
        owners[layer] = owner, released
        weights_before[str(layer)] = {"weight": weight.clone(), "bias": bias.clone()}
    original_forward, original_policy = (
        native_module.Gate._forward_local,
        native_module._use_fused_gate,
    )
    controls = {
        "held_cases": 3,
        "exact_candidate_cases": 0,
        "unchanged_score_controls": 0,
        "native_id_controls": 0,
        "restoration_controls": 0,
        "unselected_controls": 0,
        "unselected_restorations": 0,
        "graphs": 2,
        "graph_initial_controls": 0,
        "changed_graph_replays": 0,
        "graph_candidate_controls": 0,
        "graph_score_controls": 0,
        "graph_restoration_controls": 0,
        "NAS_weights": 4,
        "numerical_promotion": False,
    }
    cases, records = [], []
    for source, admitted in zip(selected, parent["cases"]):
        fields = {
            name: source[name]
            for name in ("profile", "prompt_index", "step", "layer", "identity")
        }
        for name, expected in fields.items():
            assert admitted[name] == expected
        owner, released = owners[source["layer"]]
        value = source["input"].cuda().reshape(-1, args.dim).contiguous().clone()
        require_bytes(
            source["input_control"].reshape_as(value), value.cpu(), "held FFN input"
        )
        before = value.clone()
        reference, baseline = captured_gate(released, value), captured_gate(
            owner, value
        )
        for field in FIELDS:
            require_bytes(
                admitted["reference"][field],
                reference[field].cpu(),
                "released held " + field,
            )
            require_bytes(
                admitted["native"][field], baseline[field].cpu(), "native held " + field
            )
        default = captured_gate(unselected, value)
        with scope.router_eager_reference(True, gates=[owner]):
            candidate = captured_gate(owner, value)
            other = captured_gate(unselected, value)
        assert (
            native_module.Gate._forward_local is original_forward
            and native_module._use_fused_gate is original_policy
        )
        restored, other_restored = captured_gate(owner, value), captured_gate(
            unselected, value
        )
        require_bytes(before, value, "unchanged held router input")
        metrics = {}
        for mode, output in (
            ("native", baseline),
            ("candidate", candidate),
            ("restored", restored),
        ):
            metrics.update(compare(reference, output, mode))
        metrics.update(compare(baseline, restored, "restoration"))
        metrics.update(compare(default, other, "unselected"))
        metrics.update(compare(default, other_restored, "unselected_restoration"))
        for field in FIELDS:
            for mode, counter in (
                ("restoration", "restoration_controls"),
                ("unselected", "unselected_controls"),
                ("unselected_restoration", "unselected_restorations"),
            ):
                assert metrics[mode + "." + field]["bitwise_equal"]
                controls[counter] += 1
        for mode in ("native", "candidate", "restored"):
            assert metrics[mode + ".scores"]["bitwise_equal"]
            controls["unchanged_score_controls"] += 1
        for mode in ("native", "restored"):
            assert metrics[mode + ".ids"]["bitwise_equal"]
            controls["native_id_controls"] += 1
        controls["exact_candidate_cases"] += all(
            metrics["candidate." + field]["bitwise_equal"] for field in FIELDS
        )
        records.append(dict(fields, metrics=metrics))
        cases.append(
            cpu_tree(
                dict(
                    fields,
                    input=before,
                    input_after=value.clone(),
                    reference=reference,
                    native=baseline,
                    candidate=candidate,
                    restored=restored,
                    unselected_before=default,
                    unselected=other,
                    unselected_after=other_restored,
                )
            )
        )
    graphs, graph_records = [], []
    for layer, source in ((8, selected[1]), (12, selected[0])):
        assert source["layer"] == layer
        owner, released = owners[layer]
        value = source["input"].cuda().reshape(-1, args.dim).contiguous().clone()
        initial_input = value.clone()
        initial_reference = captured_gate(released, value)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with scope.router_eager_reference(True, gates=[owner]):
            with torch.cuda.stream(stream):
                for _ in range(3):
                    owner(value)
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                captured = captured_gate(owner, value)
        assert (
            native_module.Gate._forward_local is original_forward
            and native_module._use_fused_gate is original_policy
        )
        graph.replay()
        initial_candidate = clone(captured)
        initial_metrics = compare(initial_reference, initial_candidate, "candidate")
        controls["graph_initial_controls"] += sum(
            m["bitwise_equal"] for m in initial_metrics.values()
        )
        graph_case = dict(
            layer=layer,
            source_prompt_index=source["prompt_index"],
            source_step=source["step"],
            initial_input=initial_input,
            input_after_capture=value.clone(),
            initial_reference=initial_reference,
            initial_candidate=initial_candidate,
            replays=[],
        )
        graph_record = dict(layer=layer, initial_metrics=initial_metrics, replays=[])
        previous = value.clone()
        for shift in (1, 2, 3):
            value.copy_(torch.roll(initial_input, shifts=shift, dims=-1))
            assert not torch.equal(previous, value)
            before = value.clone()
            reference, baseline = captured_gate(released, value), captured_gate(
                owner, value
            )
            graph.replay()
            candidate = clone(captured)
            restored = captured_gate(owner, value)
            require_bytes(before, value, "unchanged graph input")
            metrics = {}
            for mode, output in (
                ("native", baseline),
                ("candidate", candidate),
                ("restored", restored),
            ):
                metrics.update(compare(reference, output, mode))
            metrics.update(compare(baseline, restored, "restoration"))
            for field in FIELDS:
                assert metrics["restoration." + field]["bitwise_equal"]
                controls["graph_restoration_controls"] += 1
                controls["graph_candidate_controls"] += metrics["candidate." + field][
                    "bitwise_equal"
                ]
            for mode in ("native", "candidate", "restored"):
                assert metrics[mode + ".scores"]["bitwise_equal"]
                controls["graph_score_controls"] += 1
            controls["changed_graph_replays"] += 1
            graph_case["replays"].append(
                cpu_tree(
                    dict(
                        shift=shift,
                        input=before,
                        input_after=value.clone(),
                        reference=reference,
                        native=baseline,
                        candidate=candidate,
                        restored=restored,
                    )
                )
            )
            graph_record["replays"].append(dict(shift=shift, metrics=metrics))
            previous.copy_(value)
        graphs.append(cpu_tree(graph_case))
        graph_records.append(graph_record)
    after = {
        str(layer): {"weight": owner.weight, "bias": owner.bias}
        for layer, (owner, _) in owners.items()
    }
    for layer in weights_before:
        for name, value in weights_before[layer].items():
            require_bytes(value, after[layer][name], "unchanged NAS router operand")
    assert len(reader.hashes) == 4
    raw = {
        "cases": cases,
        "graphs": graphs,
        "weights_before": cpu_tree(weights_before),
        "weights_after": cpu_tree(after),
    }
    with cli.tensor_artifact.open("xb") as handle:
        torch.save(raw, handle)
    result = {
        "records": records,
        "graph_records": graph_records,
        "controls": controls,
        "weight_sha256": reader.hashes,
        "packet_artifact_sha256": digest(Path(manifest["packet_artifact"])),
        "held_parent_artifact_sha256": digest(Path(manifest["held_parent_artifact"])),
        "tensor_artifact_sha256": digest(cli.tensor_artifact),
        "replay_source_sha256": digest(Path(__file__).resolve()),
        "scope_source_sha256": digest(Path(scope.__file__).resolve()),
        "native_fused_policy": True,
        "current_WKV_inputs_admitted": True,
        "TF32_disabled": True,
        "numerical_promotion": False,
    }
    with cli.report.open("x") as handle:
        handle.write(
            json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n"
        )
    print("DSV41_ROUTER_EAGER_HELD_CONTROLS " + json.dumps(controls), flush=True)
    assert (
        controls["exact_candidate_cases"] == 3
        and controls["graph_initial_controls"] == 6
        and controls["graph_candidate_controls"] == 18
    ), "Eager router mismatch preserved in raw evidence"


if __name__ == "__main__":
    main()

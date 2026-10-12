"""Hold actual EP8 FFNs while selecting shared/routed input projection GEMMs."""

import argparse
import importlib
import json
import socket
import sys
from contextlib import ExitStack, contextmanager
from datetime import timedelta
from pathlib import Path
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from rtp_llm.models_py.modules.dsv4.fp8.test import (
    v41_expert_projection_reference as projection_scope,
)
from rtp_llm.models_py.modules.dsv4.fp8.test.v41_expert_precision import (
    expert_down_reference,
)
from rtp_llm.models_py.modules.dsv4.fp8.test.v41_router_eager_reference import (
    router_eager_reference,
)
from rtp_llm.models_py.modules.dsv4.moe._nccl_ep_combine_triton import fp32_peer_sum
from rtp_llm.models_py.modules.dsv4.v41_mxfp8 import V41MXFP8Linear
from rtp_llm.models_py.modules.dsv4.v41_routed import V41RoutedExperts
from rtp_llm.tools.dsv41_attention_replay import NASReader
from rtp_llm.tools.dsv41_corrected_history_replay import cpu_tree
from rtp_llm.tools.dsv41_ffn_boundary_replay import capture_linears
from rtp_llm.tools.dsv41_ffn_peer_replay import (
    admit,
    capture_native,
    capture_released,
    load_layer,
    owner_weights,
    reduce_held,
    selected_cases,
)
from rtp_llm.tools.dsv41_query_boundary_replay import require_bytes
from rtp_llm.tools.dsv41_query_owner_replay import tensor_digest
from rtp_llm.tools.dsv41_reference_probe import model_args
from rtp_llm.tools.dsv41_router_boundary_replay import captured_gate, digest

MODES = (
    ("baseline", False, False),
    ("shared_only", True, False),
    ("routed_only", False, True),
    ("composed", True, True),
    ("restored", False, False),
)


@contextmanager
def capture_reference_inputs(owner, official):
    captured = {"shared": {}, "experts": {}}
    with ExitStack() as stack:
        selected = [("shared", None, owner.shared_experts)]
        selected += [
            ("experts", str(e), expert)
            for e, expert in enumerate(owner.experts)
            if expert is not None
        ]
        for kind, name, expert in selected:
            fields = (
                captured[kind] if name is None else captured[kind].setdefault(name, {})
            )
            for part in ("w1", "w3"):
                original = getattr(expert, part).forward

                def forward(
                    *args, _call=original, _fields=fields, _part=part, **kwargs
                ):
                    original_quant = official.act_quant
                    operands = {}
                    _fields[_part] = operands
                    operands["input"] = args[0].clone()

                    def quant(*qargs, **qkwargs):
                        result = original_quant(*qargs, **qkwargs)
                        assert "values" not in operands
                        operands["values"] = result[0].clone()
                        operands["scales"] = result[1].view(torch.uint8).clone()
                        return result

                    with patch.object(official, "act_quant", quant):
                        return _call(*args, **kwargs)

                stack.enter_context(
                    patch.object(getattr(expert, part), "forward", forward)
                )
        yield captured


@contextmanager
def capture_routed_inputs(owner, counts, captured):
    local = owner._strategy._local.local
    experts = local.experts
    original_run = V41RoutedExperts._run
    original_projection = projection_scope._routed_projection

    def retain(value, name):
        for expert_id, rows in counts.items():
            captured["experts"][str(expert_id)][name] = value[
                expert_id - local.expert_start_id, :rows
            ].clone()

    def run(provider, x, valid, weights, key, gemm, *, out, input_scales):
        if provider is experts:
            assert input_scales is not None and key[0] == "masked"
            retain(x, "input_values")
            retain(input_scales, "input_scales")
            retain(weights, "route_weights")
        return original_run(
            provider, x, valid, weights, key, gemm, out=out, input_scales=input_scales
        )

    def projection(provider, values, scales, key, kernel):
        result = original_projection(provider, values, scales, key, kernel)
        if provider is experts:
            retain(result, "w13")
        return result

    with patch.object(V41RoutedExperts, "_run", run), patch.object(
        projection_scope, "_routed_projection", projection
    ):
        yield


@contextmanager
def capture_shared_inputs(owner):
    captured = {}
    original = V41MXFP8Linear.forward_quantized
    provider = owner.w13._sm120_linear

    def forward(current, values, scales, *, out=None):
        if current is provider:
            assert not captured
            captured["input_values"] = values.clone()
            captured["input_scales"] = scales.clone()
        return original(current, values, scales, out=out)

    with patch.object(V41MXFP8Linear, "forward_quantized", forward):
        yield captured


@torch.inference_mode()
def worker(rank, port, manifest_path):
    manifest = json.loads(Path(manifest_path).read_text())
    torch.set_num_threads(1)
    torch.cuda.set_device(rank)
    assert torch.cuda.get_device_capability() == (12, 0)
    torch.set_default_dtype(torch.bfloat16)
    torch.set_float32_matmul_precision("highest")
    assert not torch.backends.cuda.matmul.allow_tf32
    dist.init_process_group(
        "nccl",
        init_method="tcp://127.0.0.1:{}".format(port),
        rank=rank,
        world_size=8,
        timeout=timedelta(seconds=600),
    )
    try:
        admit(manifest)
        cases = selected_cases(manifest)
        parent_path = Path(manifest["parent_rank_artifact_pattern"].format(rank=rank))
        parent = torch.load(parent_path, map_location="cpu", weights_only=True)
        assert parent["rank"] == rank and len(parent["records"]) == 3
        nas = Path(manifest["checkpoint"])
        sys.path.insert(0, str(nas / "inference"))
        official = importlib.import_module("model")
        assert Path(official.__file__).resolve() == nas / "inference/model.py"
        official.world_size, official.rank = 8, rank
        args = model_args(official, nas, 256)
        reader, owners, prepared_before = NASReader(nas), {}, {}
        kernel = importlib.import_module("kernel")
        records, controls = [], {
            "reference_outputs": 0,
            "native_outputs": 0,
            "restorations": 0,
        }
        original_dense, original_run = (
            V41MXFP8Linear.forward_quantized,
            V41RoutedExperts._run,
        )
        for case_index, source in enumerate(cases):
            layer = source["layer"]
            if layer not in owners:
                owners[layer] = load_layer(reader, args, layer, rank, official)
                prepared_before[str(layer)] = owner_weights(owners[layer][0])
                assert (
                    prepared_before[str(layer)]
                    == parent["prepared_weight_sha256_before"][str(layer)]
                )
            native, released = owners[layer]
            held = source["input"].reshape(-1, args.dim).cuda().contiguous().clone()
            protected = held.clone()
            active = source["prompt_index"] == 1 or rank == 0
            value = held if active else held[:0]
            ids = torch.tensor(source["identity"]["input_ids"], device="cuda")
            if not active:
                ids = ids[:0]
            routes = captured_gate(released.gate, held)
            with router_eager_reference(True, gates=[native.gate]):
                native_routes = captured_gate(native.gate, held)
            for field in ("scores", "weights", "ids"):
                require_bytes(
                    routes[field],
                    native_routes[field],
                    "unchanged released route " + field,
                )
            route_ids = routes["ids"].cpu()
            counts = {
                e: int((route_ids == e).sum())
                for e in range(native.local_expert_start, native.local_expert_end)
                if bool((route_ids == e).any())
            }
            with capture_released(
                released, official
            ) as reference, capture_reference_inputs(
                released, official
            ) as reference_inputs:
                reference_output = released(held).clone()
            require_bytes(
                source["reference_output"].reshape_as(reference_output).cuda(),
                reference_output,
                "saved released FFN output",
            )
            require_bytes(
                parent["records"][case_index]["reference_output"].cuda(),
                reference_output,
                "parent released FFN output",
            )
            controls["reference_outputs"] += 1
            reference_repeat = reduce_held(reference["partial"])
            require_bytes(
                reference["reduced"], reference_repeat, "released reduction repeat"
            )
            modes = {}
            provider = native.shared_experts.w13._sm120_linear
            experts = native._strategy._local.local.experts
            for mode, shared_selected, routed_selected in MODES:
                shared_owners = [provider] if shared_selected else []
                routed_owners = [experts] if routed_selected else []
                with expert_down_reference(
                    True, shared=True, routed=True, kernel=kernel
                ), projection_scope.expert_projection_reference(
                    shared_selected or routed_selected,
                    shared=shared_owners,
                    routed=routed_owners,
                    kernel=kernel,
                ), router_eager_reference(
                    True, gates=[native.gate]
                ), capture_native(
                    native, counts
                ) as captured, capture_routed_inputs(
                    native, counts, captured
                ):
                    output = native(value, ids).clone()
                assert V41MXFP8Linear.forward_quantized is original_dense
                assert V41RoutedExperts._run is original_run
                if active and mode in ("baseline", "restored"):
                    require_bytes(
                        source["native_output"].reshape_as(output).cuda(),
                        output,
                        "saved native FFN output " + mode,
                    )
                    require_bytes(
                        parent["records"][case_index]["modes"]["eager"][
                            "output"
                        ].cuda(),
                        output,
                        "parent native FFN output " + mode,
                    )
                    controls["native_outputs"] += 1
                primary_partial = captured["local_partial"][: len(held)].clone()
                primary_reduced = reduce_held(primary_partial)
                gathered = [torch.empty_like(primary_partial) for _ in range(8)]
                dist.all_gather(gathered, primary_partial)
                peer_major = torch.cat(gathered)
                peer = torch.empty_like(primary_partial)
                fp32_peer_sum(peer_major, 8, peer)
                if rank == 0:
                    require_bytes(
                        peer_major, captured["returned"], "unchanged return wire"
                    )
                    require_bytes(peer, captured["routed"], "native rank-order sum")
                with expert_down_reference(
                    True, shared=True, routed=True, kernel=kernel
                ), projection_scope.expert_projection_reference(
                    shared_selected or routed_selected,
                    shared=shared_owners,
                    routed=routed_owners,
                    kernel=kernel,
                ), capture_linears(
                    native.shared_experts, merged=True, native_activation=True
                ) as shared_fields, capture_shared_inputs(
                    native.shared_experts
                ) as shared_input:
                    shared = native.shared_experts(held).clone()
                if active:
                    require_bytes(shared, captured["shared"], "shared replay")
                shared_fields.update(shared_input)
                modes[mode] = dict(
                    captured,
                    output=output,
                    primary_partial=primary_partial,
                    peer_major=peer_major,
                    peer_sum=peer,
                    nccl_sum=primary_reduced,
                    shared_full=shared,
                    shared_boundaries=shared_fields,
                    nccl_ffn=(primary_reduced + shared.float()).bfloat16(),
                )
                if mode == "restored":
                    for key in (
                        "output",
                        "local_partial",
                        "returned",
                        "routed",
                        "shared_full",
                    ):
                        require_bytes(
                            modes["baseline"][key], modes[mode][key], "restored " + key
                        )
                    controls["restorations"] += 1
                print(
                    "DSV41_FFN_PROJECTION_CASE "
                    + json.dumps(dict(rank=rank, case=case_index, mode=mode)),
                    flush=True,
                )
            require_bytes(protected, held, "held input after all projection replays")
            fields = {
                k: source[k]
                for k in ("profile", "prompt_index", "step", "layer", "identity")
            }
            records.append(
                cpu_tree(
                    dict(
                        fields,
                        input=protected,
                        input_after=held,
                        reference_output=reference_output,
                        reference=reference,
                        reference_inputs=reference_inputs,
                        reference_reduce_repeat=reference_repeat,
                        routes={"reference": routes, "native": native_routes},
                        modes=modes,
                    )
                )
            )
        weights_after = {
            name: tensor_digest(value) for name, value in reader.values.items()
        }
        assert weights_after == reader.hashes == parent["weight_sha256"]
        prepared_after = {
            str(layer): owner_weights(owners[layer][0]) for layer in owners
        }
        assert prepared_after == prepared_before
        assert V41MXFP8Linear.forward_quantized is original_dense
        assert V41RoutedExperts._run is original_run
        result = dict(
            rank=rank,
            records=records,
            controls=controls,
            weight_sha256=reader.hashes,
            weight_sha256_after=weights_after,
            prepared_weight_sha256_before=prepared_before,
            prepared_weight_sha256_after=prepared_after,
            parent_rank_packet_sha256=digest(parent_path),
            numerical_promotion=False,
        )
        with Path(manifest["rank_artifact_pattern"].format(rank=rank)).open(
            "xb"
        ) as handle:
            torch.save(result, handle)
        print(
            "DSV41_FFN_PROJECTION_RANK " + json.dumps(dict(rank=rank, **controls)),
            flush=True,
        )
        dist.barrier()
    finally:
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("admission_manifest", type=Path)
    cli = parser.parse_args()
    manifest = json.loads(cli.admission_manifest.read_text())
    admit(manifest)
    selected_cases(manifest)
    assert torch.cuda.device_count() == 8
    assert not any(
        Path(manifest["rank_artifact_pattern"].format(rank=r)).exists()
        for r in range(8)
    )
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    mp.spawn(worker, args=(port, str(cli.admission_manifest)), nprocs=8, join=True)
    admit(manifest)
    print("DSV41_FFN_PROJECTION_REPLAY_COMPLETE", flush=True)


if __name__ == "__main__":
    main()

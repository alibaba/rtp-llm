"""Replay three admitted NAS FFNs and separate expert math from peer reduction."""

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
from rtp_llm.models_py.modules.dsv4 import v41_routed
from rtp_llm.models_py.modules.dsv4.fp8.test.v41_expert_precision import (
    expert_down_reference,
)
from rtp_llm.models_py.modules.dsv4.fp8.test.v41_router_eager_reference import (
    router_eager_reference,
)
from rtp_llm.models_py.modules.dsv4.moe._nccl_ep_combine_triton import fp32_peer_sum
from rtp_llm.models_py.modules.dsv4.moe.moe_layer import MoE
from rtp_llm.models_py.modules.dsv4.v41_expert import V41_EXPERT_RECIPE
from rtp_llm.tools.dsv41_attention_replay import NASReader
from rtp_llm.tools.dsv41_corrected_history_replay import cpu_tree
from rtp_llm.tools.dsv41_ffn_boundary_replay import capture_linears
from rtp_llm.tools.dsv41_query_boundary_replay import require_bytes
from rtp_llm.tools.dsv41_query_owner_replay import tensor_digest
from rtp_llm.tools.dsv41_reference_probe import model_args
from rtp_llm.tools.dsv41_router_boundary_replay import captured_gate, digest
from rtp_llm.utils.model_weight import W


def admit(manifest):
    root = Path(manifest["repo_root"])
    for name, expected in manifest["source_sha256"].items():
        assert digest(root / name) == expected, name
    for field in ("asset_sha256", "helper_sha256"):
        for name, expected in manifest[field].items():
            assert digest(Path(name)) == expected, name


def selected_cases(manifest):
    report = json.loads(Path(manifest["packet_report"]).read_text())
    packet = torch.load(
        manifest["packet_artifact"], map_location="cpu", weights_only=True
    )
    assert (
        report["current_eager_router_inputs_admitted"] and report["cuda_uninitialized"]
    )
    assert packet["profiles"] == report["profiles"]
    cases = [c for c in packet["cases"] if c["profile"] == packet["profiles"][-1]]
    assert [(c["prompt_index"], c["step"], c["layer"]) for c in cases] == [
        (0, 0, 12),
        (0, 1, 8),
        (1, 0, 8),
    ]
    for c in cases:
        rows, width = c["input"].shape[-2:]
        shapes = ((rows, width), (1, rows, width))
        assert tuple(c["input"].shape) in shapes
        assert tuple(c["input_control"].shape) in shapes
        require_bytes(
            c["input"].reshape(rows, width),
            c["input_control"].reshape(rows, width),
            "admitted FFN input",
        )
    return cases


def load_layer(reader, args, layer, rank, official):
    prefix = "layers.{}.ffn.".format(layer)
    experts, topk = args.get_moe_config(layer)
    local = experts // 8
    assert experts % 8 == 0
    values = {
        W.v4_router_w: reader.load(prefix + "gate.weight"),
        W.v4_router_bias: reader.load(prefix + "gate.bias"),
    }
    for part, key in (
        ("w1", W.v4_routed_w1_w),
        ("w3", W.v4_routed_w3_w),
        ("w2", W.v4_routed_w2_w),
    ):
        scale_key = {
            "w1": W.v4_routed_w1_s,
            "w3": W.v4_routed_w3_s,
            "w2": W.v4_routed_w2_s,
        }[part]
        for suffix, target in (("weight", key), ("scale", scale_key)):
            values[target] = torch.stack(
                [
                    reader.load(
                        prefix + "experts.{}.{}.{}".format(e, part, suffix)
                    ).view(torch.uint8)
                    for e in range(rank * local, (rank + 1) * local)
                ]
            )
    for suffix, key in (("weight", W.v4_shared_w13_w), ("scale", W.v4_shared_w13_s)):
        values[key] = torch.cat(
            [
                reader.load(prefix + "shared_experts.{}.{}".format(part, suffix))
                for part in ("w1", "w3")
            ]
        )
    values[W.v4_shared_w2_w] = reader.load(prefix + "shared_experts.w2.weight")
    values[W.v4_shared_w2_s] = reader.load(prefix + "shared_experts.w2.scale")
    native = MoE(
        layer_id=layer,
        dim=args.dim,
        moe_inter_dim=args.moe_inter_dim,
        n_routed_experts=experts,
        n_activated_experts=topk,
        n_shared_experts=1,
        score_func=args.score_func,
        route_scale=args.route_scale,
        swiglu_limit=args.swiglu_limit,
        n_hash_layers=0,
        vocab_size=args.vocab_size,
        layer_weights=values,
        ep_size=8,
        ep_rank=rank,
        max_tokens_per_rank=128,
        strategy="v41_nccl_ep",
        expert_recipe=V41_EXPERT_RECIPE,
        router_fp32=True,
    )
    with torch.device("cuda"):
        released = official.MoE(layer, args)
    for name, parameter in released.named_parameters():
        value = reader.load(prefix + name)
        if parameter.element_size() == 1:
            parameter.view(torch.uint8).copy_(value.view(torch.uint8))
        else:
            parameter.copy_(value)
    return native, released


@contextmanager
def capture_released(owner, official):
    captured = {"experts": {}}
    original_reduce = dist.all_reduce

    def reduce(value, *args, **kwargs):
        assert not kwargs.get("async_op", False)
        assert "partial" not in captured
        captured["partial"] = value.clone()
        result = original_reduce(value, *args, **kwargs)
        captured["reduced"] = value.clone()
        return result

    with ExitStack() as stack:
        for expert_id, expert in enumerate(owner.experts):
            if expert is None:
                continue
            original = expert.forward

            def forward(*args, _owner=expert, _call=original, _id=expert_id, **kwargs):
                with capture_linears(_owner, reference_module=official) as fields:
                    output = _call(*args, **kwargs)
                captured["experts"][str(_id)] = dict(fields, output=output.clone())
                return output

            stack.enter_context(patch.object(expert, "forward", forward))
        stack.enter_context(patch.object(dist, "all_reduce", reduce))
        with capture_linears(owner.shared_experts, reference_module=official) as shared:
            handle = owner.shared_experts.register_forward_hook(
                lambda _owner, _args, result: captured.update(shared=result.clone())
            )
            stack.callback(handle.remove)
            yield captured
        captured["shared_boundaries"] = shared


@contextmanager
def capture_native(owner, counts):
    strategy, local = owner._strategy, owner._strategy._local.local
    experts = local.experts
    captured = {"experts": {str(e): {} for e in counts}}
    original_a2a = strategy._all_to_all
    original_received = local.forward_received
    original_gemm, original_quant = experts._masked, v41_routed.quantize_swiglu
    original_masked = experts.forward_masked

    def retain(value, name):
        for expert_id, rows in counts.items():
            index = expert_id - local.expert_start_id
            captured["experts"][str(expert_id)][name] = value[index, :rows].clone()

    def gemm(*args, **kwargs):
        result = original_gemm(*args, **kwargs)
        assert "projection" not in captured
        captured["projection"] = True
        retain(args[2], "w13")
        return result

    def quant(gate, up, valid, **kwargs):
        result = original_quant(gate, up, valid, **kwargs)
        retain(result[0], "w2_values")
        retain(result[1].reshape(*valid.shape, gate.shape[-1] // 32), "w2_scales")
        return result

    def masked(*args, **kwargs):
        result = original_masked(*args, **kwargs)
        retain(result, "output")
        return result

    def received(payload, dispatch_layout, **kwargs):
        captured["received"] = payload.clone()
        result = original_received(payload, dispatch_layout, **kwargs)
        captured["local_partial"] = result.clone()
        return result

    def a2a(send, send_counts, receive_counts, width, group):
        result = original_a2a(send, send_counts, receive_counts, width, group)
        if width == owner.dim * 4:
            captured["returned"] = result.view(torch.float32).clone()
        else:
            captured["sent"] = send.clone()
        return result

    with ExitStack() as stack:
        stack.enter_context(patch.object(strategy, "_all_to_all", a2a))
        stack.enter_context(patch.object(local, "forward_received", received))
        stack.enter_context(patch.object(experts, "_masked", gemm))
        stack.enter_context(patch.object(experts, "forward_masked", masked))
        stack.enter_context(patch.object(v41_routed, "quantize_swiglu", quant))
        handle = strategy.register_forward_hook(
            lambda _owner, _args, result: captured.update(routed=result.clone())
        )
        stack.callback(handle.remove)
        handle = owner.shared_experts.register_forward_hook(
            lambda _owner, _args, result: captured.update(shared=result.clone())
        )
        stack.callback(handle.remove)
        yield captured


def reduce_held(partial):
    original = partial.clone()
    value = partial.clone()
    dist.all_reduce(value)
    require_bytes(original, partial, "held partial is immutable")
    return value


def owner_weights(owner):
    result = {"gate.weight": owner.gate.weight, "gate.bias": owner.gate.bias}
    for name in ("w13", "w2", "s13", "s2", "_packed_s13", "_packed_s2"):
        value = getattr(owner._strategy._local.local.experts, name)
        if torch.is_tensor(value):
            result["routed." + name] = value
    for part in ("w13", "w2"):
        for name in ("weight", "weight_scales"):
            result["shared." + part + "." + name] = getattr(
                getattr(owner.shared_experts, part), name
            )
    return {name: tensor_digest(value) for name, value in result.items()}


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
        nas = Path(manifest["checkpoint"])
        sys.path.insert(0, str(nas / "inference"))
        official = importlib.import_module("model")
        assert Path(official.__file__).resolve() == nas / "inference/model.py"
        official.world_size, official.rank = 8, rank
        args = model_args(official, nas, 256)
        reader, owners, prepared_before = NASReader(nas), {}, {}
        records, controls = [], {
            "reference_outputs": 0,
            "native_outputs": 0,
            "restorations": 0,
        }
        kernel = importlib.import_module("kernel")
        for source in cases:
            layer = source["layer"]
            if layer not in owners:
                owners[layer] = load_layer(reader, args, layer, rank, official)
                prepared_before[str(layer)] = owner_weights(owners[layer][0])
            native, released = owners[layer]
            held = source["input"].reshape(-1, args.dim).cuda().contiguous().clone()
            protected = held.clone()
            active = source["prompt_index"] == 1 or rank == 0
            value = held if active else held[:0]
            ids = torch.tensor(source["identity"]["input_ids"], device="cuda")
            if not active:
                ids = ids[:0]
            reference_routes = captured_gate(released.gate, held)
            with router_eager_reference(True, gates=[native.gate]):
                eager_routes = captured_gate(native.gate, held)
            fused_routes = captured_gate(native.gate, held)
            for field in ("scores", "weights", "ids"):
                require_bytes(
                    reference_routes[field], eager_routes[field], "eager route " + field
                )
            route_ids = reference_routes["ids"].cpu()
            begin, end = native.local_expert_start, native.local_expert_end
            counts = {
                e: int((route_ids == e).sum())
                for e in range(begin, end)
                if bool((route_ids == e).any())
            }
            with capture_released(released, official) as reference:
                reference_output = released(held).clone()
            require_bytes(
                source["reference_output"].reshape_as(reference_output).cuda(),
                reference_output,
                "saved released FFN output",
            )
            controls["reference_outputs"] += 1
            reference_repeat = reduce_held(reference["partial"])
            require_bytes(
                reference["reduced"], reference_repeat, "released reduction repeat"
            )
            modes = {}
            for mode, enabled in (
                ("fused", False),
                ("eager", True),
                ("restored", False),
            ):
                with expert_down_reference(
                    True, shared=True, routed=True, kernel=kernel
                ), router_eager_reference(enabled, gates=[native.gate]), capture_native(
                    native, counts
                ) as captured:
                    output = native(value, ids).clone()
                if active:
                    require_bytes(
                        source["native_output"].reshape_as(output).cuda(),
                        output,
                        "saved native FFN output " + mode,
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
                ), capture_linears(
                    native.shared_experts, merged=True, native_activation=True
                ) as shared_fields:
                    shared = native.shared_experts(held).clone()
                if active:
                    require_bytes(shared, captured["shared"], "shared replay")
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
                            modes["fused"][key], modes[mode][key], "restored " + key
                        )
                    controls["restorations"] += 1
            require_bytes(protected, held, "held FFN input after all replays")
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
                        reference_reduce_repeat=reference_repeat,
                        routes={
                            "reference": reference_routes,
                            "eager": eager_routes,
                            "fused": fused_routes,
                        },
                        modes=modes,
                    )
                )
            )
        weights_after = {
            name: tensor_digest(value) for name, value in reader.values.items()
        }
        assert weights_after == reader.hashes, "Held GPU NAS operands changed"
        prepared_after = {
            str(layer): owner_weights(owners[layer][0]) for layer in owners
        }
        assert prepared_after == prepared_before, "Prepared native operands changed"
        result = {
            "rank": rank,
            "records": records,
            "controls": controls,
            "weight_sha256": reader.hashes,
            "weight_sha256_after": weights_after,
            "prepared_weight_sha256_before": prepared_before,
            "prepared_weight_sha256_after": prepared_after,
            "numerical_promotion": False,
        }
        path = Path(manifest["rank_artifact_pattern"].format(rank=rank))
        with path.open("xb") as handle:
            torch.save(result, handle)
        print(
            "DSV41_FFN_PEER_RANK " + json.dumps(dict(rank=rank, **controls)), flush=True
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
    print("DSV41_FFN_PEER_REPLAY_COMPLETE", flush=True)


if __name__ == "__main__":
    main()

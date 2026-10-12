"""Run with torchrun --standalone --nproc_per_node=8 on eight SM120 devices."""

import argparse
import dataclasses
import datetime
import hashlib
import json
import os
from pathlib import Path
import sys
import types


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--without-engine-bindings", action="store_true")
    parser.add_argument("--with-feed-forward", action="store_true")
    parser.add_argument("--flashinfer-source-jit", action="store_true")
    parser.add_argument("--extra-package-path", action="append", default=[])
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root))
    sys.path.extend(args.extra_package_path)
    if args.flashinfer_source_jit:
        sys.modules["flashinfer_jit_cache"] = None
    if args.without_engine_bindings:
        package = types.ModuleType("rtp_llm.models_py.modules")
        package.__path__ = [str(root / "rtp_llm/models_py/modules")]
        sys.modules[package.__name__] = package

    import torch
    import torch.distributed as dist

    from rtp_llm.models_py.modules.dsv4.fp8.test.test_v41_routed_hardware import (
        V41RoutedHardwareTest,
    )
    from rtp_llm.models_py.modules.dsv4.v41_collective import (
        V41CollectiveMoE,
        V41EPPlan,
    )
    from rtp_llm.models_py.modules.dsv4.v41_dispatch import V41DispatchLayout
    from rtp_llm.models_py.modules.dsv4.v41_moe import V41LocalMoE
    from rtp_llm.models_py.modules.dsv4.v41_routed import V41RoutedExperts
    from rtp_llm.models_py.modules.dsv4.fp8.test.test_v41_expert_hardware import (
        V41ExpertHardwareTest,
    )
    from rtp_llm.models_py.modules.dsv4.v41_expert import V41SharedExpert
    from rtp_llm.models_py.modules.dsv4.v41_feed_forward import (
        V41FeedForward,
        V41Router,
    )

    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    torch.backends.cuda.matmul.allow_tf32 = False
    if torch.cuda.get_device_capability() != (12, 0):
        raise RuntimeError("Collective qualification requires exact SM120 devices")
    dist.init_process_group(
        "nccl",
        timeout=datetime.timedelta(seconds=120),
        device_id=torch.device("cuda", rank),
    )
    if dist.get_world_size() != 8:
        raise RuntimeError("Collective qualification requires eight workers")
    groups = [dist.new_group(list(range(start, start + 4))) for start in (0, 4)]
    records = []
    try:
        for role in ("prefill", "decode"):
            world = 4 if role == "prefill" else 8
            stage = rank // 4 if role == "prefill" else 0
            group = groups[stage] if role == "prefill" else dist.group.WORLD
            peer = dist.get_rank(group)
            experts = 384 // world
            plan = V41EPPlan(
                V41DispatchLayout(5120, 6, 384),
                role,
                stage,
                21 if stage else 0,
                tuple(dist.get_process_group_ranks(group)),
                "hardware-qualification-1",
                4,
            )
            local = V41LocalMoE(
                V41RoutedExperts(V41RoutedHardwareTest.weights(experts)),
                expert_start_id=peer * experts,
                max_tokens=world * 4,
                masked_token_limit=world * 4,
            )
            op = V41CollectiveMoE(local, plan, group)
            op.admit()
            # Every member must reject, even though only one peer disagrees.
            op.plan = dataclasses.replace(
                plan, generation="wrong" if peer == 0 else plan.generation
            )
            try:
                op.admit()
            except RuntimeError as error:
                assert "mismatch" in str(error)
            else:
                raise AssertionError("Mismatched generation was admitted")
            op.plan = plan
            local.expert_start_id += int(peer == 0)
            try:
                op.admit()
            except RuntimeError as error:
                assert "mismatch" in str(error)
            else:
                raise AssertionError("Wrong expert ownership was admitted")
            local.expert_start_id = peer * experts
            op.plan = dataclasses.replace(
                plan, intermediate_dim=2560 if peer == 0 else 2304
            )
            try:
                op.admit()
            except RuntimeError as error:
                assert "mismatch" in str(error)
            else:
                raise AssertionError("Wrong intermediate dimension was admitted")
            op.plan = plan
            op.admit()

            count = peer % 4
            rng = torch.Generator(device="cuda").manual_seed(4141 + rank)
            x = torch.randn(
                count, 5120, device="cuda", dtype=torch.bfloat16, generator=rng
            )
            weights = torch.rand(count, 6, device="cuda", generator=rng)
            ids = torch.tensor(
                [(slot % world) * experts + slot for slot in range(6)],
                device="cuda",
                dtype=torch.int32,
            ).repeat(count, 1)
            valid = torch.arange(count, device="cuda") % 3 != 1
            x[~valid] = float("nan")

            def check(actual, shared=None):
                all_x = torch.zeros(
                    (world * 4, 5120), device="cuda", dtype=torch.bfloat16
                )
                all_w = torch.zeros((world * 4, 6), device="cuda")
                all_i = torch.full((world * 4, 6), -1, device="cuda", dtype=torch.int32)
                send_x, send_w, send_i = (
                    all_x[:4].clone(),
                    all_w[:4].clone(),
                    all_i[:4].clone(),
                )
                send_x[:count] = torch.where(valid[:, None], x, 0)
                send_w[:count] = torch.where(valid[:, None], weights, 0)
                send_i[:count] = torch.where(valid[:, None], ids, -1)
                for received, sent in (
                    (all_x, send_x),
                    (all_w, send_w),
                    (all_i, send_i),
                ):
                    dist.all_gather_into_tensor(received, sent, group=group)
                partial = local(
                    all_x,
                    all_w,
                    all_i,
                    layout="contiguous" if role == "prefill" else "masked",
                )
                all_partial = torch.empty(
                    (world * world * 4, 5120), device="cuda", dtype=torch.float32
                )
                dist.all_gather_into_tensor(all_partial, partial, group=group)
                expected = torch.zeros((4, 5120), device="cuda", dtype=torch.float32)
                for source in range(world):
                    begin = source * world * 4 + peer * 4
                    expected.add_(all_partial[begin : begin + 4])
                expected = expected[:count]
                if shared is not None:
                    expected = (expected + shared(x, valid).float()).bfloat16()
                assert actual.dtype == expected.dtype
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)

            check(op(x, weights, ids, valid_rows=valid))
            graph_checked = False
            if role == "decode":
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    for _ in range(3):
                        op(x, weights, ids, valid_rows=valid)
                torch.cuda.current_stream().wait_stream(stream)
                torch.cuda.synchronize()
                dist.barrier(group=group)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    actual = op(x, weights, ids, valid_rows=valid)
                for scale in (0, 0.37, 1.5):
                    x.normal_()
                    valid.logical_not_()
                    weights.fill_(scale)
                    ids.copy_(ids.flip(1))
                    graph.replay()
                    check(actual)
                graph_checked = True
                del graph, actual
            if args.with_feed_forward:
                router_rng = torch.Generator(device="cuda").manual_seed(4145)
                router = V41Router(
                    (
                        torch.randn(384, 5120, device="cuda", generator=router_rng)
                        / 5120**0.5
                    ).bfloat16(),
                    torch.randn(384, device="cuda", generator=router_rng) * 0.1,
                    max_rows=4,
                )
                shared = V41SharedExpert(V41ExpertHardwareTest.weights())
                ffn = V41FeedForward(plan.layer_id, router, shared, op)
                weights, ids = router(x, valid)
                check(ffn(x, valid), shared)
                if role == "decode":
                    stream = torch.cuda.Stream()
                    stream.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(stream):
                        for _ in range(3):
                            ffn(x, valid)
                    torch.cuda.current_stream().wait_stream(stream)
                    torch.cuda.synchronize()
                    dist.barrier(group=group)
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        actual = ffn(x, valid)
                    for live in (False, True):
                        x.normal_()
                        valid.fill_(live)
                        if not live:
                            x.fill_(float("nan"))
                        graph.replay()
                        weights, ids = router(x, valid)
                        check(actual, shared)
                    del graph, actual
                del ffn, router, shared
            records.append(
                {
                    "role": role,
                    "pp_rank": stage,
                    "world_ranks": plan.world_ranks,
                    "experts": experts,
                    "rows": count,
                    "decode_graph": graph_checked,
                    "peer_rejection": True,
                    "exact_fp32_return": True,
                    "complete_feed_forward": args.with_feed_forward,
                }
            )
            print(f"rank {rank}: {role} passed", flush=True)
            del op, local
            torch.cuda.empty_cache()
            dist.barrier()
        reports = [None] * 8
        dist.all_gather_object(
            reports,
            {
                "rank": rank,
                "gpu_uuid": str(torch.cuda.get_device_properties(rank).uuid),
                "results": records,
            },
        )
        if rank == 0 and args.manifest:
            manifest = {
                "torch": torch.__version__,
                "nccl": torch.cuda.nccl.version(),
                "engine_bindings_bypassed": args.without_engine_bindings,
                "flashinfer_source_jit": args.flashinfer_source_jit,
                "reports": reports,
                "source_sha256": {
                    str(path.relative_to(root)): hashlib.sha256(
                        path.read_bytes()
                    ).hexdigest()
                    for path in (root / "rtp_llm/models_py/modules/dsv4").glob(
                        "v41_*.py"
                    )
                },
            }
            args.manifest.parent.mkdir(parents=True, exist_ok=True)
            args.manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

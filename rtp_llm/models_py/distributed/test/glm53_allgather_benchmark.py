"""Eight-rank GLM53 CP AllGather: real kernels, exact bytes, no model.

Graph timing isolates GPU service time (includes symmetric scratch copy).
Eager timing calls the production CP dispatcher, including output allocation.
Capturing the selected kernels is ONLY a benchmark technique: the production
CP dispatcher deliberately declines symmetric memory during graph capture.
"""

import argparse
import datetime
import importlib.util
import json
import os
import socket
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

BASE = Path(__file__).resolve().parents[1]
if not (BASE / "symm_mem.py").exists():
    BASE = Path(__file__).resolve().parent


def load(name):
    full = "rtp_llm.models_py.distributed." + name
    spec = importlib.util.spec_from_file_location(full, BASE / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[full] = module
    spec.loader.exec_module(module)
    return module


symm = load("symm_mem")
cp = load("pynccl_cp")
ROWS = [500, 1000, 2000, 4000, 8000, 12000, 16000, 20000, 24000, 30000]
LAYOUTS = [
    ("kv_fused_or_history", 576, torch.bfloat16),
    ("indexer_new_k", 128, torch.bfloat16),
    ("kv_split_ckv", 512, torch.bfloat16),
    ("kv_split_kpe", 64, torch.bfloat16),
    ("indexer_packed", 132, torch.uint8),
]


def cases(quick):
    result = []
    for role, width, dtype in LAYOUTS:
        rows = [500, 30000] if quick else ROWS[:]
        if width == 576:
            rows += [14563, 14564]
        if width == 512:
            rows += [16383, 16384]
        result.extend((role, r, width, dtype) for r in sorted(set(rows)))
    return result


def worker(rank, port, dest, quick):
    torch.set_num_threads(1)
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group(
        "nccl",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=8,
        timeout=datetime.timedelta(seconds=180),
    )
    group = dist.group.WORLD
    control = dist.new_group(backend="gloo", timeout=datetime.timedelta(seconds=180))
    cp._resolve_process_group = lambda g: g
    cp._OPT_ENABLED = True
    cp.warmup(group, device)
    comm = symm.init_symm_mem_communicator(group)
    assert (
        comm is not None and not comm.disabled
    ), "Real eight-rank multicast unavailable"
    outdir = Path(dest)
    outdir.mkdir(parents=True, exist_ok=True)
    meta = dict(
        rank=rank,
        torch=torch.__version__,
        cuda=torch.version.cuda,
        nccl=torch.cuda.nccl.version(),
        device=torch.cuda.get_device_name(rank),
        capability=torch.cuda.get_device_capability(rank),
        max_bytes=comm.max_size,
        env={
            k: os.environ.get(k)
            for k in ["NCCL_ALGO", "NCCL_PROTO", "CUDA_VISIBLE_DEVICES", "GLM5_CP_OPT"]
        },
    )
    (outdir / f"environment-rank{rank}.json").write_text(json.dumps(meta, indent=2))
    results = []
    stream = torch.cuda.Stream(device=device)
    graph_calls = 8 if quick else 32
    repeats = 3 if quick else 9
    eager_calls = 8 if quick else 24
    with torch.cuda.stream(stream):
        for case_id, (role, rows, width, dtype) in enumerate(cases(quick)):
            x = (
                (torch.arange(rows * width, device=device).reshape(rows, width) % 113)
                + rank
            ).to(dtype)
            # Distinct ranks, varying elements and padded rows exercise rank order.
            x[-1].zero_()
            output = torch.empty((8, rows, width), device=device, dtype=dtype)
            cp_role = {
                "kv_fused_or_history": "mla_ckv",
                "indexer_new_k": "indexer_k_bf16",
                "kv_split_ckv": "mla_ckv",
                "kv_split_kpe": "mla_kpe",
                "indexer_packed": "indexer_k_packed",
            }[role]
            registered = cp._symm_output(
                cp_role, (8 * rows, width), dtype, device, group
            )
            nccl_output = (
                output if registered is None else registered.view(8, rows, width)
            )
            eligible = comm.should_torch_symm_mem_allgather(x, max_gathered_rows=None)
            capacity_ok = x.nbytes * 8 < comm.max_size
            expected_eligible = dtype == torch.bfloat16 and capacity_ok
            assert eligible == expected_eligible, (
                role,
                rows,
                eligible,
                expected_eligible,
            )
            record = dict(
                registered_nccl_output=registered is not None,
                production_role=cp_role,
                case_id=case_id,
                role=role,
                rows=rows,
                width=width,
                dtype=str(dtype),
                shard_bytes=x.nbytes,
                gathered_bytes=x.nbytes * 8,
                eligible=eligible,
                skip_reason=(
                    None
                    if eligible
                    else (
                        "dtype != bfloat16"
                        if dtype != torch.bfloat16
                        else "gathered_bytes >= 128 MiB"
                    )
                ),
                graph_calls=graph_calls,
                measurements={},
            )

            def nccl():
                cp.all_gather_into_tensor(
                    nccl_output.view(8 * rows, width), x, group=group
                )

            def symmetric():
                comm.all_gather(x, out=output)

            def raw_symm():
                torch.ops.symm_mem.multimem_all_gather_out(
                    x.view(-1), group.group_name, comm.buffer[: 8 * x.numel()]
                )

            def check_output(y):
                # Exact byte comparison independently checks every rank's payload.
                for source in range(8):
                    expected = (
                        (
                            torch.arange(rows * width, device=device).reshape(
                                rows, width
                            )
                            % 113
                        )
                        + source
                    ).to(dtype)
                    expected[-1].zero_()
                    assert torch.equal(
                        y[source].view(torch.uint8), expected.view(torch.uint8)
                    ), (case_id, rank, source)

            methods = {"nccl": nccl}
            if eligible:
                methods.update(symm=symmetric, symm_transport=raw_symm)
            graphs = {}
            for name, fn in methods.items():
                for _ in range(10):
                    fn()
                stream.synchronize()
                if name == "symm_transport":
                    check_output(comm.buffer[: 8 * x.numel()].view(8, rows, width))
                else:
                    check_output(nccl_output if name == "nccl" else output)
                stream.synchronize()
                dist.barrier(group=control)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream):
                    for _ in range(graph_calls):
                        fn()
                graph.replay()
                stream.synchronize()
                check_output(
                    comm.buffer[: 8 * x.numel()].view(8, rows, width)
                    if name == "symm_transport"
                    else (nccl_output if name == "nccl" else output)
                )
                graphs[name] = graph
                record["measurements"][name] = {"graph_us": []}
            # Alternating method order reduces systematic warm/clock bias.
            events = {
                n: [
                    (
                        torch.cuda.Event(enable_timing=True),
                        torch.cuda.Event(enable_timing=True),
                    )
                    for _ in range(repeats)
                ]
                for n in graphs
            }
            for rep in range(repeats):
                names = list(graphs)
                if rep % 2:
                    names.reverse()
                for name in names:
                    dist.barrier(group=control)
                    start, end = events[name][rep]
                    start.record(stream)
                    graphs[name].replay()
                    end.record(stream)
                    end.synchronize()
                    record["measurements"][name]["graph_us"].append(
                        start.elapsed_time(end) * 1000 / graph_calls
                    )
            # Eager path really invokes the current production dispatcher; no
            # forced eligibility or reinterpretation of uint8 as BF16.
            for name in ["nccl"] + (["symm"] if eligible else []):
                cp._TORCH_SYMM_GATHER = name == "symm"
                for _ in range(10):
                    y = cp.all_gather(x, group, role=cp_role)
                stream.synchronize()
                check_output(y.view(8, rows, width))
                pairs = [
                    (
                        torch.cuda.Event(enable_timing=True),
                        torch.cuda.Event(enable_timing=True),
                    )
                    for _ in range(eager_calls)
                ]
                stream.synchronize()
                dist.barrier(group=control)
                wall = time.perf_counter()
                for start, end in pairs:
                    start.record(stream)
                    y = cp.all_gather(x, group, role=cp_role)
                    end.record(stream)
                stream.synchronize()
                record["measurements"][name]["eager_event_us"] = [
                    a.elapsed_time(b) * 1000 for a, b in pairs
                ]
                record["measurements"][name]["eager_wall_us"] = (
                    (time.perf_counter() - wall) * 1e6 / eager_calls
                )
            # Profile actual eager calls separately; all ranks participate.
            dist.barrier(group=control)
            with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ]
            ) as prof:
                for name in ["nccl"] + (["symm"] if eligible else []):
                    cp._TORCH_SYMM_GATHER = name == "symm"
                    with torch.profiler.record_function("allgather_" + name):
                        for _ in range(3):
                            y = cp.all_gather(x, group, role=cp_role)
                    stream.synchronize()
            if rank == 0:
                prof.export_chrome_trace(
                    str(outdir / f"trace-{case_id:02d}-{role}-{rows}.json")
                )
            record["bitwise_pass"] = True
            results.append(record)
            (outdir / f"rank{rank}.json").write_text(json.dumps(results, indent=2))
            if rank == 0:
                print(
                    f"PASS {case_id} {role} rows={rows} eligible={eligible}", flush=True
                )
            del graphs, events, output, x, y, graph
            stream.synchronize()
    dist.barrier(group=control)
    cp.destroy()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    assert torch.cuda.device_count() == 8, "Requires exactly eight visible GPUs"
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    mp.spawn(worker, args=(port, args.output, args.quick), nprocs=8, join=True)
    print("PASS eight-rank AllGather benchmark", flush=True)

"""Multi-GPU correctness probe for the actual K3 FP8 AG/RS functions."""

import argparse
import ast
import datetime
import json
import os
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from fp8_shape_probe import load_quantizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="RTP source or server runfiles/rtp_llm directory",
    )
    root = parser.parse_args().source_root
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", timeout=datetime.timedelta(seconds=120))
    import deep_gemm
    from deep_gemm.gemm_rs import GemmRSBuffer
    from deep_gemm.utils.layout import (
        get_mn_major_tma_aligned_packed_ue8m0_tensor as pack,
    )

    quantize = load_quantizer(
        root / "rtp_llm/model_loader/per_block_fp8_quant_weight.py"
    )
    ns = {"torch": torch, "torch_symm_mem": symm_mem}
    for file, name in (
        (
            "rtp_llm/models_py/modules/kimi_k3/gemm_reduce_scatter.py",
            "_fp8_remote_gemm_reduce_scatter",
        ),
        ("rtp_llm/models_py/distributed/symm_mem.py", "fused_all_gather_fp8_linear"),
    ):
        tree = ast.parse((root / file).read_text())
        nodes = [
            n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name
        ]
        assert len(nodes) == 1
        exec(compile(ast.Module(body=nodes, type_ignores=[]), file, "exec"), ns)

    class Projection:
        scale_ue8m0 = True

        def __init__(self, n, k):
            self.N, self.K = n, k
            w = torch.randn(n, k, dtype=torch.bfloat16, device="cuda")
            self.wq, ws = quantize(w, 128, use_ue8m0=True)
            self.ws = pack(ws.index_select(0, torch.arange(n, device="cuda") // 128))

        def quantize_input(self, x):
            a = x.float().reshape(x.shape[0], self.K // 128, 128)
            scale = a.abs().amax(-1).clamp_min(1e-4) / 448.0
            scale = torch.pow(2.0, torch.ceil(torch.log2(scale)))
            return (a / scale[..., None]).reshape_as(x).to(torch.float8_e4m3fn), pack(
                scale
            )

        def forward_quantized(self, a, scale, out=None):
            if out is None:
                out = torch.empty(
                    (a.shape[0], self.N), device=a.device, dtype=torch.bfloat16
                )
            deep_gemm.fp8_gemm_nt((a, scale), (self.wq, self.ws), out)
            return out

        def __call__(self, x, out=None):
            return self.forward_quantized(*self.quantize_input(x), out=out)

    torch.manual_seed(17 + rank)
    world, group = dist.get_world_size(), dist.group.WORLD
    projection = Projection(7168, 128)
    workspace = GemmRSBuffer(group, 2048, 7168, device=torch.device("cuda", rank))
    state = SimpleNamespace(
        n=7168, max_m=2048, world_size=world, workspace=workspace, deep_gemm=deep_gemm
    )
    ag_projection = Projection(2112, 128)
    for m in (world, 16 * world, 128 * world):
        x = torch.randn(m, 128, dtype=torch.bfloat16, device="cuda")
        expected = projection(x)
        reference = torch.empty_like(expected[: m // world], dtype=torch.float32)
        dist.reduce_scatter_tensor(reference, expected.float(), group=group)
        reference = reference.to(torch.bfloat16)
        for repeat in range(3):
            actual = ns["_fp8_remote_gemm_reduce_scatter"](x, projection, state, m)
            torch.cuda.synchronize()
            torch.testing.assert_close(actual, reference, rtol=0.02, atol=0.125)
        shard = x[: m // world].contiguous()
        symm_mem.get_symm_mem_workspace(group.group_name, min_size=shard.numel() * 2)
        gathered = torch.empty_like(x)
        dist.all_gather_into_tensor(gathered, shard, group=group)
        expected_ag = ag_projection(gathered)
        actual_ag = ns["fused_all_gather_fp8_linear"](shard, [ag_projection], group)[0]
        torch.cuda.synchronize()
        torch.testing.assert_close(actual_ag, expected_ag, rtol=0, atol=0)
        if rank == 0:
            print(
                json.dumps({"m": m, "world": world, "rs_pass": True, "ag_pass": True}),
                flush=True,
            )
    # Capture only after all shapes and both communication paths are warmed.
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    capture_stream = torch.cuda.Stream()
    capture_stream.wait_stream(torch.cuda.current_stream())
    # Warm on the actual capture stream, including workspace stream ownership.
    with torch.cuda.stream(capture_stream):
        ns["_fp8_remote_gemm_reduce_scatter"](x, projection, state, m)
        ns["fused_all_gather_fp8_linear"](shard, [ag_projection], group)
    capture_stream.synchronize()
    with torch.cuda.graph(graph, stream=capture_stream):
        graph_rs = ns["_fp8_remote_gemm_reduce_scatter"](x, projection, state, m)
        graph_ag = ns["fused_all_gather_fp8_linear"](shard, [ag_projection], group)[0]
    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(graph_rs, reference, rtol=0.02, atol=0.125)
    torch.testing.assert_close(graph_ag, expected_ag, rtol=0, atol=0)
    if rank == 0:
        print(json.dumps({"world": world, "graph_pass": True}), flush=True)
    dist.barrier()
    workspace.destroy()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()

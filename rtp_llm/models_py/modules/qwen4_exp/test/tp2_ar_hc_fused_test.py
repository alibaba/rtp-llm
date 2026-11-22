import os
import socket
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from rtp_llm.models_py.modules.hybrid.gated_residual import inject_into_residual
from rtp_llm.models_py.modules.qwen4_exp.head_ipc_gather import Qwen4HeadIpcGather
from rtp_llm.models_py.modules.qwen4_exp.tp2_ar_hc_fused import (
    DeferredTp2Sublayer,
    Qwen4Tp2ArHcWorkspace,
    _DeferSlot,
    _attn_tp2_defer_slot,
    _moe_tp2_defer_slot,
    prepare_tp2_ar_hc_workspace,
    try_defer_attn_tp2,
    try_defer_moe_packed_tp2,
)
from rtp_llm.models_py.triton_kernels.common.gated_residual_prefill_inject_norm_triton import (
    try_inject_grouped_norm,
)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _tp2_worker(rank: int, port: int) -> None:
    from rtp_llm.ops import ensure_compute_ops_loaded, ensure_engine_ops_loaded

    ensure_compute_ops_loaded()
    ensure_engine_ops_loaded()
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group("nccl", rank=rank, world_size=2)
    group = dist.group.WORLD
    try:
        ws = Qwen4Tp2ArHcWorkspace(group, device)
        head_ws = Qwen4HeadIpcGather(group, device, max_rows=256)
        try:
            g_gen = torch.Generator(device=device)
            g_gen.manual_seed(20261122)
            gamma = (torch.randn(10240, generator=g_gen, device=device) * 0.1).bfloat16()

            for rows in (1, 4, 17, 48, 64, 128, 192, 256, 512):
                x_gen = torch.Generator(device=device)
                x_gen.manual_seed(1000 + rows)
                hyper_x = torch.randn(
                    (rows, 10240), generator=x_gen, device=device, dtype=torch.bfloat16
                )
                gates = torch.randn(
                    (rows, 4), generator=x_gen, device=device, dtype=torch.bfloat16
                )

                r_gen = torch.Generator(device=device)
                r_gen.manual_seed(5000 + rows * 10 + rank)
                local_single = torch.randn(
                    (rows, 2560), generator=r_gen, device=device, dtype=torch.bfloat16
                )
                local_packed = torch.randn(
                    (2 * rows, 2560), generator=r_gen, device=device, dtype=torch.bfloat16
                )

                peer_single = local_single.clone()
                dist.all_reduce(peer_single, group=group)
                peer_packed = local_packed.clone()
                dist.all_reduce(peer_packed, group=group)
                ref_dual = peer_packed[:rows] + peer_packed[rows:]

                for packed_dual, local_t, ref_sub in (
                    (False, local_single, peer_single),
                    (True, local_packed, ref_dual),
                ):
                    deferred = DeferredTp2Sublayer(local_t, packed_dual, ws)
                    actual_inj = ws.try_inject_only(hyper_x, deferred, gates)
                    expected_inj = inject_into_residual(hyper_x, ref_sub, gates)
                    assert torch.equal(actual_inj, expected_inj), (rows, packed_dual)

                    for collect in (False, True):
                        res, norm, scale = ws.try_inject_grouped_norm(
                            hyper_x, deferred, gates, gamma, 1e-6, collect_scale=collect
                        )
                        ref_res, ref_norm, ref_scale = try_inject_grouped_norm(
                            hyper_x, ref_sub, gates, gamma, 1e-6, collect_scale=collect
                        )
                        assert torch.equal(res, ref_res), (rows, packed_dual, collect)
                        torch.testing.assert_close(norm, ref_norm, atol=1e-2, rtol=1e-2)
                        if collect:
                            torch.testing.assert_close(
                                scale, ref_scale, atol=1e-6, rtol=1e-5
                            )

            # CUDA Graph replay across multiple steps and shapes
            for rows in (4, 64, 192):
                gx = torch.randn((rows, 10240), device=device, dtype=torch.bfloat16)
                gg = torch.randn((rows, 4), device=device, dtype=torch.bfloat16)
                gp = torch.randn((2 * rows, 2560), device=device, dtype=torch.bfloat16)
                deferred = DeferredTp2Sublayer(gp, True, ws)
                ws.try_inject_grouped_norm(gx, deferred, gg, gamma, 1e-6, collect_scale=False)
                torch.cuda.synchronize(device)
                dist.barrier(group=group)

                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    c_res, c_norm, _ = ws.try_inject_grouped_norm(
                        gx, deferred, gg, gamma, 1e-6, collect_scale=False
                    )
                for step in range(3):
                    s_gen = torch.Generator(device=device)
                    s_gen.manual_seed(9000 + rows + step)
                    gx.copy_(
                        torch.randn(gx.shape, generator=s_gen, device=device, dtype=torch.bfloat16)
                    )
                    gg.copy_(
                        torch.randn(gg.shape, generator=s_gen, device=device, dtype=torch.bfloat16)
                    )
                    p_gen = torch.Generator(device=device)
                    p_gen.manual_seed(12000 + rows + step * 10 + rank)
                    gp.copy_(
                        torch.randn(gp.shape, generator=p_gen, device=device, dtype=torch.bfloat16)
                    )
                    graph.replay()
                    ref_p = gp.clone()
                    dist.all_reduce(ref_p, group=group)
                    ref_sub = ref_p[:rows] + ref_p[rows:]
                    ref_res, ref_norm, _ = try_inject_grouped_norm(
                        gx, ref_sub, gg, gamma, 1e-6, collect_scale=False
                    )
                    assert torch.equal(c_res, ref_res)
                    torch.testing.assert_close(c_norm, ref_norm, atol=1e-2, rtol=1e-2)
                graph.reset()

            # Head IPC Gather (up to 256 rows) & Head IPC Argmax (eager + 2-step CUDA graph)
            for rows in (1, 4, 48, 64, 192, 256):
                h_gen = torch.Generator(device=device)
                h_gen.manual_seed(30000 + rows * 10 + rank)
                local_logits = torch.randn(
                    (rows, 124160), generator=h_gen, device=device, dtype=torch.bfloat16
                )
                # Inject ties across ranks on row 0
                local_logits[0, 17] = 32.0
                gathered = head_ws.try_gather(local_logits)
                shards = [torch.empty_like(local_logits) for _ in range(2)]
                dist.all_gather(shards, local_logits, group=group)
                expected_full = torch.cat(shards, dim=1).float()
                assert torch.equal(gathered, expected_full)

                for valid_v in (0, 248000):
                    eff = valid_v if valid_v > 0 else 248320
                    if valid_v > 0:
                        local_logits[:, -10:] = 64.0
                    actual_idx = head_ws.try_argmax(local_logits, valid_v)
                    expected_idx = torch.argmax(expected_full[:, :eff], dim=-1, keepdim=True)
                    shards_v = [torch.empty_like(local_logits) for _ in range(2)]
                    dist.all_gather(shards_v, local_logits, group=group)
                    expected_idx = torch.argmax(
                        torch.cat(shards_v, dim=1).float()[:, :eff], dim=-1, keepdim=True
                    )
                    assert torch.equal(actual_idx, expected_idx), (rows, valid_v)

            # 2-step draft loop CUDA Graph capture of try_argmax
            l0 = torch.randn((48, 124160), device=device, dtype=torch.bfloat16)
            l1 = torch.randn((48, 124160), device=device, dtype=torch.bfloat16)
            head_ws.try_argmax(l0, 0)
            torch.cuda.synchronize(device)
            dist.barrier(group=group)
            d_graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(d_graph):
                t0 = head_ws.try_argmax(l0, 0)
                t1 = head_ws.try_argmax(l1, 0)
            for step in range(3):
                lg = torch.Generator(device=device)
                lg.manual_seed(70000 + step * 10 + rank)
                l0.copy_(torch.randn(l0.shape, generator=lg, device=device, dtype=torch.bfloat16))
                l1.copy_(torch.randn(l1.shape, generator=lg, device=device, dtype=torch.bfloat16))
                d_graph.replay()
                for lt, actual_t in ((l0, t0), (l1, t1)):
                    sh = [torch.empty_like(lt) for _ in range(2)]
                    dist.all_gather(sh, lt, group=group)
                    ref_t = torch.argmax(torch.cat(sh, dim=1).float(), dim=-1, keepdim=True)
                    assert torch.equal(actual_t, ref_t)
            d_graph.reset()
        finally:
            head_ws.close()
    finally:
        dist.destroy_process_group()


class Tp2ArHcFusedTest(unittest.TestCase):
    def test_single_gpu_guards_and_defer_slots(self):
        cfg = SimpleNamespace(hc_mult=4, hidden_size=2560)
        pc = SimpleNamespace(tp_size=1, dp_size=1, local_world_size=1)
        self.assertIsNone(prepare_tp2_ar_hc_workspace(cfg, pc))

        ws = object.__new__(Qwen4Tp2ArHcWorkspace)
        ws.device = torch.device("cuda", torch.cuda.current_device())
        x = torch.zeros((8, 2560), device=ws.device, dtype=torch.bfloat16)
        packed = torch.zeros((16, 2560), device=ws.device, dtype=torch.bfloat16)

        from rtp_llm.models_py.distributed import collective_torch

        self.assertIsNone(try_defer_attn_tp2(x, collective_torch.Group.TP))
        tok = _attn_tp2_defer_slot.set(_DeferSlot(ws))
        try:
            d1 = try_defer_attn_tp2(x, collective_torch.Group.TP)
            self.assertIsInstance(d1, DeferredTp2Sublayer)
            self.assertIsNone(try_defer_attn_tp2(x, collective_torch.Group.TP))
        finally:
            _attn_tp2_defer_slot.reset(tok)

        mtok = _moe_tp2_defer_slot.set(_DeferSlot(ws))
        try:
            d2 = try_defer_moe_packed_tp2(packed)
            self.assertIsInstance(d2, DeferredTp2Sublayer)
            self.assertTrue(d2.packed_dual)
            self.assertIsNone(try_defer_moe_packed_tp2(packed))
        finally:
            _moe_tp2_defer_slot.reset(mtok)

    def test_two_gpu_p2p_fusion_and_argmax_parity(self):
        if (
            torch.cuda.device_count() < 2
            or torch.cuda.get_device_capability(0) != (9, 0)
            or not (
                torch.cuda.can_device_access_peer(0, 1)
                and torch.cuda.can_device_access_peer(1, 0)
            )
        ):
            self.skipTest("Requires 2 P2P-capable SM90 GPUs")
        port = _free_port()
        mp.spawn(_tp2_worker, args=(port,), nprocs=2, join=True)


if __name__ == "__main__":
    unittest.main()

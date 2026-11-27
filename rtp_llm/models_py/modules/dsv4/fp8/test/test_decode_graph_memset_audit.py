"""Empirical CUDA Graph node & profiler audit for DSv4 FP8 decode paths."""

from __future__ import annotations

import ctypes
import time
import unittest
from collections import Counter, defaultdict

import torch

from rtp_llm.models_py.modules.dsv4.fp8 import _dspark_meta_triton
from rtp_llm.models_py.modules.dsv4.fp8.decode.decode_attn_metadata import (
    allocate_decode_metadata_fp8,
    update_decode_metadata_in_place_fp8,
)
from rtp_llm.models_py.modules.dsv4.fp8.decode.mega_csa_adapter import MegaCSAAdapter
from rtp_llm.models_py.modules.dsv4.fp8.decode.mega_csa_runtime import MegaCSARuntime
from rtp_llm.models_py.modules.dsv4.fp8.decode.mega_csa_weights import (
    HEAD_DIM,
    PRO_GEOMETRY,
)
from rtp_llm.models_py.modules.dsv4.fp8.decode.mega_hca_adapter import MegaHCAAdapter
from rtp_llm.models_py.modules.dsv4.fp8.decode.mega_hca_weights import (
    HCA_COMPRESS_RATIO,
)
from rtp_llm.models_py.modules.dsv4.fp8.test import test_mega_csa_rtp_eager as csa_mod
from rtp_llm.models_py.modules.dsv4.fp8.test import test_mega_hca_rtp_eager as hca_mod
from rtp_llm.models_py.modules.dsv4.kv_cache_utils import CSA_KV, HCA_KV

_NODE_TYPE_NAMES = {
    0: "kernel",
    1: "memcpy",
    2: "memset",
    3: "host",
    4: "graph",
    5: "empty",
    6: "event_wait",
    7: "event_record",
}


class _CudaMemsetParams(ctypes.Structure):
    _fields_ = [
        ("dst", ctypes.c_void_p),
        ("pitch", ctypes.c_size_t),
        ("value", ctypes.c_uint),
        ("elementSize", ctypes.c_uint),
        ("width", ctypes.c_size_t),
        ("height", ctypes.c_size_t),
    ]


_CUDART = ctypes.CDLL("libcudart.so")
_CUDART.cudaGraphGetNodes.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_void_p),
    ctypes.POINTER(ctypes.c_size_t),
]
_CUDART.cudaGraphNodeGetType.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_int),
]
_CUDART.cudaGraphMemsetNodeGetParams.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(_CudaMemsetParams),
]


def _inspect_cuda_graph(graph: torch.cuda.CUDAGraph) -> dict:
    raw = ctypes.c_void_p(graph.raw_cuda_graph())
    num_nodes = ctypes.c_size_t(0)
    rc = _CUDART.cudaGraphGetNodes(raw, None, ctypes.byref(num_nodes))
    if rc != 0:
        raise RuntimeError(f"cudaGraphGetNodes(count) failed rc={rc}")
    n = num_nodes.value
    nodes = (ctypes.c_void_p * n)()
    rc = _CUDART.cudaGraphGetNodes(raw, nodes, ctypes.byref(num_nodes))
    if rc != 0:
        raise RuntimeError(f"cudaGraphGetNodes(list) failed rc={rc}")

    type_counts: Counter[str] = Counter()
    memset_details: Counter[tuple[int, int, int, int, int]] = Counter()
    memset_total_bytes = 0
    for i in range(n):
        ntype = ctypes.c_int(-1)
        _CUDART.cudaGraphNodeGetType(nodes[i], ctypes.byref(ntype))
        tname = _NODE_TYPE_NAMES.get(ntype.value, f"other_{ntype.value}")
        type_counts[tname] += 1
        if ntype.value == 2:
            p = _CudaMemsetParams()
            _CUDART.cudaGraphMemsetNodeGetParams(nodes[i], ctypes.byref(p))
            nbytes = int(p.elementSize) * int(p.width) * int(p.height)
            memset_total_bytes += nbytes
            memset_details[
                (int(p.elementSize), int(p.value), int(p.width), int(p.height), nbytes)
            ] += 1

    return {
        "total_nodes": n,
        "type_counts": dict(type_counts),
        "memset_nodes": type_counts.get("memset", 0),
        "memcpy_nodes": type_counts.get("memcpy", 0),
        "kernel_nodes": type_counts.get("kernel", 0),
        "memset_total_bytes": memset_total_bytes,
        "memset_details": dict(memset_details),
    }


def _extract_profiler_audit(prof: torch.profiler.profile) -> dict:
    interesting = (
        "memset",
        "memcpy",
        "zero_",
        "fill_",
        "copy_",
        "index_select",
        "indexselect",
        "eventsynchronize",
        "streamsynchronize",
        "devicesynchronize",
    )
    hits: dict[str, Counter[str]] = defaultdict(Counter)
    cuda_kernels: Counter[str] = Counter()
    for evt in prof.events():
        name_l = evt.name.lower()
        if any(tok in name_l for tok in interesting):
            stack = evt.stack[:4] if evt.stack else []
            stack_str = " <- ".join(str(s) for s in stack) or "<cpp>"
            hits[evt.name][stack_str] += 1
        if evt.device_type == torch.profiler.DeviceType.CUDA:
            cuda_kernels[evt.name] += 1
    return {
        "cpu_hits": {k: dict(v.most_common(8)) for k, v in hits.items()},
        "cuda_kernels": dict(cuda_kernels.most_common(30)),
    }


def _audit_graph(
    label: str,
    fn,
    *,
    warmup_iters: int = 3,
    replay_iters: int = 50,
    profile_capture: bool = True,
) -> dict:
    torch.cuda.synchronize()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(warmup_iters):
            fn()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph(keep_graph=True)
    prof_res = None
    t0 = time.perf_counter()
    if profile_capture:
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            with_stack=True,
            record_shapes=False,
        ) as prof:
            with torch.cuda.graph(graph, stream=stream):
                out = fn()
        prof_res = _extract_profiler_audit(prof)
    else:
        with torch.cuda.graph(graph, stream=stream):
            out = fn()
    torch.cuda.synchronize()
    capture_ms = (time.perf_counter() - t0) * 1000.0

    # Ensure replay works cleanly and time replays.
    for _ in range(5):
        graph.replay()
    torch.cuda.synchronize()

    start_ev = torch.cuda.Event(enable_timing=True)
    end_ev = torch.cuda.Event(enable_timing=True)
    latencies_us: list[float] = []
    for _ in range(replay_iters):
        start_ev.record()
        graph.replay()
        end_ev.record()
        end_ev.synchronize()
        latencies_us.append(start_ev.elapsed_time(end_ev) * 1000.0)
    latencies_us.sort()
    median_us = latencies_us[len(latencies_us) // 2]
    mean_us = sum(latencies_us) / len(latencies_us)

    # Also measure host-side launch overhead alone + sync batch.
    t_host0 = time.perf_counter()
    for _ in range(replay_iters):
        graph.replay()
    host_launch_us = ((time.perf_counter() - t_host0) * 1e6) / replay_iters
    torch.cuda.synchronize()
    batched_gpu_us = ((time.perf_counter() - t_host0) * 1e6) / replay_iters

    info = _inspect_cuda_graph(graph)
    info["label"] = label
    info["capture_ms"] = round(capture_ms, 3)
    info["replay_median_us"] = round(median_us, 2)
    info["replay_mean_us"] = round(mean_us, 2)
    info["host_launch_us"] = round(host_launch_us, 2)
    info["batched_gpu_us"] = round(batched_gpu_us, 2)
    if prof_res is not None:
        info["prof"] = prof_res

    print(
        f"[AUDIT] {label}: nodes={info['total_nodes']} "
        f"(kernel={info['kernel_nodes']}, memset={info['memset_nodes']}, "
        f"memcpy={info['memcpy_nodes']}, types={info['type_counts']}) "
        f"memset_bytes={info['memset_total_bytes']} "
        f"capture={info['capture_ms']:.2f}ms "
        f"replay_med={info['replay_median_us']:.2f}us "
        f"batched={info['batched_gpu_us']:.2f}us"
    )
    if info["memset_details"]:
        print(f"  memset_details[(elem,val,w,h,bytes)]: {info['memset_details']}")
    if prof_res:
        if prof_res["cpu_hits"]:
            print(f"  cpu_hits: {prof_res['cpu_hits']}")
        print(f"  cuda_kernels: {prof_res['cuda_kernels']}")

    graph.reset()
    del out, graph
    return info


class DecodeGraphMemsetAuditTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA required")
        cap = torch.cuda.get_device_capability()
        if cap not in ((10, 0), (10, 3)):
            raise unittest.SkipTest(f"Requires sm_100a/sm_103a, got {cap}")
        cls.device = torch.device("cuda", torch.cuda.current_device())

    def test_audit_mega_csa_hca_and_standard_decode_graphs(self) -> None:
        dev = self.device
        # Build one shared CSA block/runtime/adapter and one shared HCA block/runtime/adapter.
        csa_block, csa_runtime, csa_adapter = csa_mod.MegaCSARTPEagerTest._make_layer(
            dev, PRO_GEOMETRY
        )
        hca_block, hca_runtime, hca_adapter = hca_mod.MegaHCARTPEagerTest._make_layer(
            dev, PRO_GEOMETRY
        )

        # 1. Audit MegaCSA across bs = 1, 8, 32, 64, 96 (1-layer & 4-layer) and q_len = 4
        for bs in (1, 8, 32, 64, 96):
            pools_bs = csa_mod._make_pools(dev, batch_size=bs)
            csa_mod._fill_random_context(pools_bs, dev, seed=111 + bs)
            meta = csa_mod.build_decode_metadata_fp8(
                attention_inputs=torch.full((bs,), 32, dtype=torch.int32, device=dev),
                q_len=1,
                window_size=128,
                head_dim=HEAD_DIM,
                max_seq_len=pools_bs.max_seq_len,
                compress_ratios=[4],
                index_topk=PRO_GEOMETRY.index_topk,
                device=dev,
                paged_block_tables=pools_bs.block_tables,
                paged_pool_entries_per_block=pools_bs.entries_per_block,
                paged_pool_tokens_per_block=pools_bs.tokens_per_block,
            )
            hidden = torch.randn(
                bs, 1, csa_mod.HC, PRO_GEOMETRY.dim, dtype=torch.bfloat16, device=dev
            )

            def run_1_csa():
                csa_runtime.begin_decode(meta)
                return csa_adapter.forward_attention_sublayer(
                    csa_block, hidden, meta, kv_cache=pools_bs.kv_cache
                )

            res = _audit_graph(
                f"mega_csa_1layer_bs{bs}_q1",
                run_1_csa,
                profile_capture=(bs in (1, 32, 96)),
            )
            self.assertEqual(res["memset_nodes"], 0)
            self.assertEqual(res["memcpy_nodes"], 0)

        # Multi-layer (4 layers) + in-graph metadata update
        raw_csa_weights = csa_mod._make_layer_weights(dev, PRO_GEOMETRY)
        csa_adapters_4 = [
            MegaCSAAdapter(csa_block, raw_csa_weights, csa_runtime) for _ in range(4)
        ]
        for bs in (1, 32, 96):
            pools_bs = csa_mod._make_pools(dev, batch_size=bs)
            csa_mod._fill_random_context(pools_bs, dev, seed=411 + bs)
            pool_specs = {
                at: (
                    pools_bs.entries_per_block[at],
                    pools_bs.tokens_per_block[at],
                    int(pools_bs.block_tables[at].shape[1]),
                )
                for at in pools_bs.entries_per_block
            }
            meta_alloc = allocate_decode_metadata_fp8(
                max_batch_size=bs,
                q_len=1,
                window_size=128,
                head_dim=HEAD_DIM,
                max_seq_len=pools_bs.max_seq_len,
                compress_ratios=[4],
                index_topk=PRO_GEOMETRY.index_topk,
                device=dev,
                paged_pool_specs=pool_specs,
            )
            for at, bt in pools_bs.block_tables.items():
                if at in meta_alloc.pool_block_tables:
                    meta_alloc.pool_block_tables[at] = bt[:bs]
            start_pos = torch.full((bs,), 32, dtype=torch.int32, device=dev)
            hidden = torch.randn(
                bs, 1, csa_mod.HC, PRO_GEOMETRY.dim, dtype=torch.bfloat16, device=dev
            )

            def run_4_csa():
                update_decode_metadata_in_place_fp8(
                    meta_alloc,
                    start_pos,
                    paged_block_tables=pools_bs.block_tables,
                    paged_pool_entries_per_block=pools_bs.entries_per_block,
                    paged_pool_tokens_per_block=pools_bs.tokens_per_block,
                    forbid_realloc=True,
                )
                csa_runtime.begin_decode(meta_alloc)
                x = hidden
                for ad in csa_adapters_4:
                    x = ad.forward_attention_sublayer(
                        csa_block, x, meta_alloc, kv_cache=pools_bs.kv_cache
                    )
                return x

            res = _audit_graph(
                f"mega_csa_4layer_plus_meta_bs{bs}_q1",
                run_4_csa,
                profile_capture=(bs in (1, 32)),
            )
            self.assertEqual(res["memset_nodes"], 0)
            self.assertEqual(res["memcpy_nodes"], 0)

        # 2. Audit MegaHCA across bs = 1, 8, 32, 64, 96 and 4-layer
        for bs in (1, 8, 32, 64, 96):
            pools_bs = hca_mod._make_pools(dev, batch_size=bs)
            hca_mod._fill_random_context(pools_bs, dev, seed=511 + bs)
            hca_mod._fill_random_state(pools_bs, dev, seed=611 + bs)
            meta = hca_mod.build_decode_metadata_fp8(
                attention_inputs=torch.full((bs,), 260, dtype=torch.int32, device=dev),
                q_len=1,
                window_size=128,
                head_dim=HEAD_DIM,
                max_seq_len=pools_bs.max_seq_len,
                compress_ratios=[HCA_COMPRESS_RATIO],
                index_topk=PRO_GEOMETRY.index_topk,
                device=dev,
                paged_block_tables=pools_bs.block_tables,
                paged_pool_entries_per_block=pools_bs.entries_per_block,
                paged_pool_tokens_per_block=pools_bs.tokens_per_block,
            )
            hidden = torch.randn(
                bs, 1, hca_mod.HC, PRO_GEOMETRY.dim, dtype=torch.bfloat16, device=dev
            )

            def run_1_hca():
                hca_runtime.begin_decode(meta)
                return hca_adapter.forward_attention_sublayer(
                    hca_block, hidden, meta, kv_cache=pools_bs.kv_cache
                )

            res = _audit_graph(
                f"mega_hca_1layer_bs{bs}_q1",
                run_1_hca,
                profile_capture=(bs in (1, 32, 96)),
            )
            self.assertEqual(res["memset_nodes"], 0)
            self.assertEqual(res["memcpy_nodes"], 0)

        # 3. Audit standard (non-Mega) AttentionFP8.forward_decode + metadata update
        for bs, q_len in ((1, 1), (8, 1), (32, 1), (64, 1), (96, 1), (8, 4), (32, 4)):
            pools_bs = csa_mod._make_pools(dev, batch_size=bs)
            csa_mod._fill_random_context(pools_bs, dev, seed=711 + bs)
            pool_specs = {
                at: (
                    pools_bs.entries_per_block[at],
                    pools_bs.tokens_per_block[at],
                    int(pools_bs.block_tables[at].shape[1]),
                )
                for at in pools_bs.entries_per_block
            }
            meta_alloc = allocate_decode_metadata_fp8(
                max_batch_size=bs,
                q_len=q_len,
                window_size=128,
                head_dim=HEAD_DIM,
                max_seq_len=pools_bs.max_seq_len,
                compress_ratios=[4],
                index_topk=PRO_GEOMETRY.index_topk,
                device=dev,
                paged_pool_specs=pool_specs,
            )
            for at, bt in pools_bs.block_tables.items():
                if at in meta_alloc.pool_block_tables:
                    meta_alloc.pool_block_tables[at] = bt[:bs]
            start_pos = torch.full((bs,), 32, dtype=torch.int32, device=dev)
            x_in = torch.randn(
                bs, q_len, PRO_GEOMETRY.dim, dtype=torch.bfloat16, device=dev
            )

            def run_std_decode():
                update_decode_metadata_in_place_fp8(
                    meta_alloc,
                    start_pos,
                    paged_block_tables=pools_bs.block_tables,
                    paged_pool_entries_per_block=pools_bs.entries_per_block,
                    paged_pool_tokens_per_block=pools_bs.tokens_per_block,
                    forbid_realloc=True,
                )
                return csa_block.attn.forward_decode(
                    x_in, meta_alloc, kv_cache=pools_bs.kv_cache
                )

            res = _audit_graph(
                f"std_attn_fp8_1layer_bs{bs}_q{q_len}",
                run_std_decode,
                profile_capture=(bs in (1, 32) and q_len == 1)
                or (bs == 8 and q_len == 4),
            )
            self.assertEqual(res["memset_nodes"], 0)

        # 4. Audit DSpark meta triton + standalone update_decode_metadata_in_place_fp8
        bt_dspark = torch.arange(1, 32 * 16 + 1, dtype=torch.int32, device=dev).view(
            32, 16
        )
        pos_dspark = (
            torch.arange(128, 132, dtype=torch.int64, device=dev)
            .view(1, 4)
            .expand(32, 4)
            .contiguous()
        )
        pref_dspark = torch.full((32,), 128, dtype=torch.int32, device=dev)
        act_dspark = torch.ones((32,), dtype=torch.bool, device=dev)
        res = _audit_graph(
            "dspark_meta_triton_bs32_q4",
            lambda: _dspark_meta_triton.try_compute_dspark_decode_meta(
                bt_dspark,
                pos_dspark,
                pref_dspark,
                act_dspark,
                gamma=4,
                window=128,
                entries_per_block=64,
                tokens_per_block=128,
            ),
            profile_capture=True,
        )
        self.assertEqual(res["memset_nodes"], 0)

    def test_in_graph_non_aliased_block_table_capture_zero_memset_and_multistep_equivalence(
        self,
    ) -> None:
        dev = self.device
        # 1. Verify eager update_decode_metadata_in_place_fp8 zeroes tails for
        # narrow non-aliased block tables, while CUDA graph capture skips
        # redundant tail memsets.
        pools_full = hca_mod._make_pools(dev, batch_size=32)
        hca_mod._fill_random_context(pools_full, dev, seed=811)
        hca_mod._fill_random_state(pools_full, dev, seed=812)
        wide_specs = {
            at: (
                pools_full.entries_per_block[at],
                pools_full.tokens_per_block[at],
                max(8, int(pools_full.block_tables[at].shape[1]) + 4),
            )
            for at in pools_full.entries_per_block
        }
        meta_wide = allocate_decode_metadata_fp8(
            max_batch_size=32,
            q_len=1,
            window_size=128,
            head_dim=HEAD_DIM,
            max_seq_len=pools_full.max_seq_len,
            compress_ratios=[HCA_COMPRESS_RATIO],
            index_topk=PRO_GEOMETRY.index_topk,
            device=dev,
            paged_pool_specs=wide_specs,
        )
        narrow_tables = {
            at: bt[:12, : max(1, int(bt.shape[1]) - 1)].clone()
            for at, bt in pools_full.block_tables.items()
        }
        for dst_bt in meta_wide.pool_block_tables.values():
            dst_bt.fill_(777)
        sp_12 = torch.full((12,), 258, dtype=torch.int32, device=dev)
        update_decode_metadata_in_place_fp8(
            meta_wide,
            sp_12,
            paged_block_tables=narrow_tables,
            paged_pool_entries_per_block=pools_full.entries_per_block,
            paged_pool_tokens_per_block=pools_full.tokens_per_block,
            forbid_realloc=True,
        )
        for at, dst_bt in meta_wide.pool_block_tables.items():
            n_cols = min(narrow_tables[at].shape[1], dst_bt.shape[1])
            self.assertEqual(torch.count_nonzero(dst_bt[12:]).item(), 0)
            if n_cols < dst_bt.shape[1]:
                self.assertEqual(torch.count_nonzero(dst_bt[:12, n_cols:]).item(), 0)

        meta_info = _audit_graph(
            "update_meta_narrow_unaliased_bs12_of_32",
            lambda: update_decode_metadata_in_place_fp8(
                meta_wide,
                sp_12,
                paged_block_tables=narrow_tables,
                paged_pool_entries_per_block=pools_full.entries_per_block,
                paged_pool_tokens_per_block=pools_full.tokens_per_block,
                forbid_realloc=True,
            ),
            profile_capture=True,
        )
        self.assertEqual(meta_info["memset_nodes"], 0)

        # 2. Verify shrinking HCA metadata reuses pre-padded [0..16) buffers safely.
        meta_16 = allocate_decode_metadata_fp8(
            max_batch_size=16,
            q_len=1,
            window_size=128,
            head_dim=HEAD_DIM,
            max_seq_len=pools_full.max_seq_len,
            compress_ratios=[HCA_COMPRESS_RATIO],
            index_topk=PRO_GEOMETRY.index_topk,
            device=dev,
            paged_pool_specs={
                at: (
                    pools_full.entries_per_block[at],
                    pools_full.tokens_per_block[at],
                    int(pools_full.block_tables[at].shape[1]),
                )
                for at in pools_full.entries_per_block
            },
        )
        tables_16 = {
            at: bt[:16].contiguous() for at, bt in pools_full.block_tables.items()
        }
        update_decode_metadata_in_place_fp8(
            meta_16,
            torch.full((16,), 300, dtype=torch.int32, device=dev),
            paged_block_tables=tables_16,
            paged_pool_entries_per_block=pools_full.entries_per_block,
            paged_pool_tokens_per_block=pools_full.tokens_per_block,
            forbid_realloc=True,
        )
        tables_1 = {
            at: bt[:1].contiguous() for at, bt in pools_full.block_tables.items()
        }
        update_decode_metadata_in_place_fp8(
            meta_16,
            torch.tensor([129], dtype=torch.int32, device=dev),
            paged_block_tables=tables_1,
            paged_pool_entries_per_block=pools_full.entries_per_block,
            paged_pool_tokens_per_block=pools_full.tokens_per_block,
            forbid_realloc=True,
        )
        self.assertEqual(
            torch.count_nonzero(meta_16.hca_front_position_ids_16[1:]).item(), 0
        )
        self.assertTrue(torch.all(meta_16.hca_front_state_rows_16[1:] == -1).item())

        # 3. Interleaved consecutive graph replays across varying bs & inputs
        # must match eager reference outputs without stale state.
        hca_block, hca_runtime, hca_adapter = hca_mod.MegaHCARTPEagerTest._make_layer(
            dev, PRO_GEOMETRY
        )
        csa_block, csa_runtime, csa_adapter = csa_mod.MegaCSARTPEagerTest._make_layer(
            dev, PRO_GEOMETRY
        )
        for name, mod, make_pools_fn, fill_fn, adapter, runtime, ratios in (
            (
                "mega_csa",
                csa_block,
                csa_mod._make_pools,
                lambda p, s: csa_mod._fill_random_context(p, dev, seed=s),
                csa_adapter,
                csa_runtime,
                [4],
            ),
            (
                "mega_hca",
                hca_block,
                hca_mod._make_pools,
                lambda p, s: (
                    hca_mod._fill_random_context(p, dev, seed=s),
                    hca_mod._fill_random_state(p, dev, seed=s + 1000),
                ),
                hca_adapter,
                hca_runtime,
                [HCA_COMPRESS_RATIO],
            ),
        ):
            pools_max = make_pools_fn(dev, batch_size=32)
            fill_fn(pools_max, 901)
            specs = {
                at: (
                    pools_max.entries_per_block[at],
                    pools_max.tokens_per_block[at],
                    int(pools_max.block_tables[at].shape[1]),
                )
                for at in pools_max.entries_per_block
            }
            buckets = (1, 4, 8, 32)
            graphs = {}
            holders = {}
            stream = torch.cuda.Stream()
            for bsz in buckets:
                meta_b = allocate_decode_metadata_fp8(
                    max_batch_size=bsz,
                    q_len=1,
                    window_size=128,
                    head_dim=HEAD_DIM,
                    max_seq_len=pools_max.max_seq_len,
                    compress_ratios=ratios,
                    index_topk=PRO_GEOMETRY.index_topk,
                    device=dev,
                    paged_pool_specs=specs,
                )
                sp_b = torch.full((bsz,), 256, dtype=torch.int32, device=dev)
                hid_b = torch.randn(
                    bsz,
                    1,
                    csa_mod.HC,
                    PRO_GEOMETRY.dim,
                    dtype=torch.bfloat16,
                    device=dev,
                )
                bt_b = {
                    at: bt[:bsz].contiguous()
                    for at, bt in pools_max.block_tables.items()
                }

                def _step(mb=meta_b, sp=sp_b, hb=hid_b, btb=bt_b):
                    update_decode_metadata_in_place_fp8(
                        mb,
                        sp,
                        paged_block_tables=btb,
                        paged_pool_entries_per_block=pools_max.entries_per_block,
                        paged_pool_tokens_per_block=pools_max.tokens_per_block,
                        forbid_realloc=True,
                    )
                    runtime.begin_decode(mb)
                    return adapter.forward_attention_sublayer(
                        mod, hb, mb, kv_cache=pools_max.kv_cache
                    )

                with torch.cuda.stream(stream):
                    for _ in range(3):
                        _step()
                torch.cuda.synchronize()
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g, stream=stream):
                    out_b = _step()
                graphs[bsz] = (g, out_b, meta_b, sp_b, hid_b, bt_b)
                holders[bsz] = out_b

            torch.cuda.synchronize()
            for step_idx, bsz in enumerate((32, 1, 8, 4, 32, 1, 8)):
                g, out_b, _, sp_b, hid_b, bt_b = graphs[bsz]
                new_sp = torch.randint(130, 320, (bsz,), dtype=torch.int32, device=dev)
                new_hid = torch.randn_like(hid_b)
                sp_b.copy_(new_sp)
                hid_b.copy_(new_hid)
                g.replay()
                torch.cuda.synchronize()
                got = out_b.clone()

                meta_ref = allocate_decode_metadata_fp8(
                    max_batch_size=bsz,
                    q_len=1,
                    window_size=128,
                    head_dim=HEAD_DIM,
                    max_seq_len=pools_max.max_seq_len,
                    compress_ratios=ratios,
                    index_topk=PRO_GEOMETRY.index_topk,
                    device=dev,
                    paged_pool_specs=specs,
                )
                update_decode_metadata_in_place_fp8(
                    meta_ref,
                    new_sp,
                    paged_block_tables=bt_b,
                    paged_pool_entries_per_block=pools_max.entries_per_block,
                    paged_pool_tokens_per_block=pools_max.tokens_per_block,
                    forbid_realloc=True,
                )
                runtime.begin_decode(meta_ref)
                ref = adapter.forward_attention_sublayer(
                    mod, new_hid, meta_ref, kv_cache=pools_max.kv_cache
                )
                torch.testing.assert_close(
                    got,
                    ref,
                    rtol=1e-2,
                    atol=1e-2,
                    msg=f"{name} interleaved step={step_idx} bs={bsz}",
                )
            for g, *_ in graphs.values():
                g.reset()


if __name__ == "__main__":
    unittest.main()

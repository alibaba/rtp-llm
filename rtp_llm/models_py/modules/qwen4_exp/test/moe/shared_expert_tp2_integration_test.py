"""Two real SM120 ranks; isolated BF16 math and real NVFP4 consumer gates.

Manually assemble ONE model-private layer, using real production modules. The
first phase replaces only routed production with an explicitly named GPU tensor;
the second runs SelectTopk -> FusedMoe -> CUTLASS NVFP4 EP2 -> unified IPC sum.
This is not checkpoint/serving/acceptance/performance evidence.
"""

import json
import os
import socket
import unittest
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F

from rtp_llm.models_py.modules.qwen4_exp.moe import shared_tp2

_ENV = {
    "RTP_QWEN4_BF16_SHARED_TP2": "1",
    "RTP_QWEN4_TARGET_REDUCE": "ipc256",
}


@contextmanager
def _record_installed_tinygemm(rank):
    """Transparent real-API call receipt; never supply a reference/output value.

    Replay executes captured GPU work and does not call this Python recorder.
    Existing independent oracle/output assertions remain the replay gate.
    """
    import flashinfer.gemm as gemm

    installed_api = gemm.tinygemm_bf16
    calls, printed = [], set()

    def record(input, weight, out, bias=None, use_pdl=False):
        assert bias is None and use_pdl is False
        assert input.dtype == weight.dtype == out.dtype == torch.bfloat16
        assert input.device == weight.device == out.device
        assert weight.shape == (640, 2560) and weight.stride() == (2560, 1)
        assert out.shape == (input.shape[0], 640) and out.is_contiguous()
        assert out.data_ptr() not in (input.data_ptr(), weight.data_ptr())
        capture = torch.cuda.is_current_stream_capturing()
        result = installed_api(input, weight, out, bias=bias, use_pdl=use_pdl)
        assert result is None, "pinned TinyGEMM mutates out and returns None"
        receipt = {
            "event": "installed_tinygemm_bf16_returned",
            "rank": rank,
            "rows": input.shape[0],
            "weight_shape": list(weight.shape),
            "weight_stride": list(weight.stride()),
            "out_shape": list(out.shape),
            "capturing": capture,
            "use_pdl": use_pdl,
            "input_ptr": input.data_ptr(),
            "weight_ptr": weight.data_ptr(),
            "out_ptr": out.data_ptr(),
            "return_is_none": True,
        }
        calls.append(receipt)
        key = (input.shape[0], capture)
        if key not in printed:
            print(json.dumps(receipt), flush=True)
            printed.add(key)
        return result

    with patch.object(gemm, "tinygemm_bf16", record):
        yield calls


def _same_on_both(value):
    """Compare EVERY element, including row/position/padding metadata."""
    peers = [
        torch.empty(value.shape, dtype=value.dtype, device=value.device)
        for _ in range(2)
    ]
    dist.all_gather(peers, value.contiguous())
    torch.testing.assert_close(peers[0], peers[1], rtol=0, atol=0)


def _source():
    def marked(rows, cols, salt):
        row = torch.arange(rows, dtype=torch.int32)[:, None]
        col = torch.arange(cols, dtype=torch.int32)[None, :]
        return (((row * (19 + salt) + col * 13 + salt) % 127 - 63) / 1024).to(
            torch.bfloat16
        )

    # KN owners match real source BF16 module strides; no square matrices.
    return {
        "up_kn": marked(2560, 1280, 1).contiguous(),
        "down_kn": marked(640, 2560, 3).contiguous(),
        "gate_kn": marked(2560, 1, 5).contiguous(),
    }


def _pc(rank):
    from rtp_llm.config.server_config_setup import set_parallelism_config
    from rtp_llm.ops import ParallelismConfig, RoleType

    pc = ParallelismConfig()
    pc.tp_size = pc.ep_size = pc.world_size = pc.local_world_size = 2
    pc.dp_size = 1
    pc.ffn_sp_size = 2
    pc.world_rank = rank
    pc.role_type = RoleType.DECODE
    set_parallelism_config(pc)
    assert pc.tp_rank == pc.ep_rank == rank
    assert pc.get_ffn_tp_size() == 1 and pc.get_ffn_tp_rank() == 0
    return pc


def _nvfp4_weights(device):
    """Reuse the existing cutlass_nvfp4_executor_test producer, small E=4."""
    from rtp_llm.device.device_impl import CudaImpl
    from rtp_llm.model_loader.online_modelopt_fp4_quant_weight import (
        quantize_expert_to_nvfp4,
    )
    from rtp_llm.utils.model_weight import W

    generator = torch.Generator().manual_seed(1227)
    values = {}
    for wn, sn, an, ins, shape in (
        (W.moe_w1, W.moe_s1, W.moe_w1_s2, W.moe_w1_i_s, (1280, 2560)),
        (W.moe_w2, W.moe_s2, W.moe_w2_s2, W.moe_w2_i_s, (2560, 640)),
    ):
        packed = [
            quantize_expert_to_nvfp4(
                (torch.randn(shape, generator=generator) * 0.02)
                .to(torch.bfloat16)
                .to(device)
            )
            for _ in range(4)
        ]
        q, s, a = zip(*packed)
        values[wn] = torch.stack(q)
        values[sn] = CudaImpl.swizzle_blockscale(torch.stack(s))
        values[an] = torch.stack(a)
        values[ins] = torch.full((4,), 32 / 2688.0, device=device)
    return values


def _executor(pc, weights, full=False):
    from rtp_llm.models_py.modules.factory.fused_moe.defs.quant_config import (
        FusedMoEQuantConfig,
    )
    from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.cutlass_nvfp4_executor import (
        CutlassNvfp4Executor,
    )

    local = (
        weights
        if full
        else {
            name: value[pc.ep_rank * 2 : (pc.ep_rank + 1) * 2].contiguous()
            for name, value in weights.items()
        }
    )
    config = SimpleNamespace(
        ep_size=1 if full else 2,
        ep_rank=0 if full else pc.ep_rank,
        parallelism_config=pc,
        model_config=SimpleNamespace(quant_config=None),
    )
    executor = CutlassNvfp4Executor(
        config,
        FusedMoEQuantConfig(quant_dtype=torch.uint8, block_shape=[16, 16]),
        local,
    )
    assert executor.tp_size == 1 and executor.tp_rank == 0
    assert executor.sm120_prefill is None, "this gate requires decode CUTLASS"
    return executor


def _layer(pc, source, executor):
    from rtp_llm.config.model_config import ModelConfig
    from rtp_llm.models_py.modules import SelectTopk
    from rtp_llm.models_py.modules.base import FusedSiluAndMul, SigmoidGateScaleAdd
    from rtp_llm.models_py.modules.factory.fused_moe.defs.fused_moe import FusedMoe
    from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.routers.pure_tp_router import (
        PureTpRouterNvfp4Cutlass,
    )
    from rtp_llm.models_py.modules.factory.linear.impl.cuda.f16_linear import (
        CudaF16Linear,
    )
    from rtp_llm.models_py.modules.hybrid.dense_mlp import DenseMLP
    from rtp_llm.models_py.modules.qwen4_exp.moe.shared_overlap import Qwen4MoeLayer
    from rtp_llm.ops import ActivationType

    device = torch.device("cuda", pc.local_rank)
    config = ModelConfig()
    config.is_mtp = False
    config.hidden_size, config.inter_size = 2560, 640
    config.attn_config.head_num = 20
    config.attn_config.size_per_head = 128
    config.num_layers, config.max_seq_len, config.vocab_size = 1, 1024, 5120
    config._qwen4_full_fp8_nvfp4 = True
    config._qwen4_dense_storage = "source_bf16"
    config.expert_num, config.moe_k = 4, 2
    config.has_moe_norm = True
    router_config = SimpleNamespace(
        tp_size=2, ep_size=2, ep_rank=pc.ep_rank, expert_num=4
    )
    router = PureTpRouterNvfp4Cutlass(router_config, executor.quant_config)
    shared = DenseMLP.__new__(DenseMLP)
    torch.nn.Module.__init__(shared)
    shared.parallelism_config = pc
    shared.activation_type = ActivationType.Swiglu
    shared.is_gated = True
    shared.act_fn = FusedSiluAndMul()
    shared.up_proj = CudaF16Linear(source["up_kn"].to(device).clone())
    shared.down_proj = CudaF16Linear(source["down_kn"].to(device).clone())
    layer = Qwen4MoeLayer.__new__(Qwen4MoeLayer)
    torch.nn.Module.__init__(layer)
    layer.config, layer.parallelism_config = config, pc
    layer.ffn_tp_size, layer.ep_size = 1, 2
    layer.shared_expert = shared
    gate_nk = torch.empty((1, 2560), dtype=torch.bfloat16, device=device)
    gate_nk.copy_(source["gate_kn"].t())
    layer.shared_expert_gate = CudaF16Linear(gate_nk.t())
    layer.sigmoid_gate_scale_add = SigmoidGateScaleAdd()
    layer.fused_moe = FusedMoe(router, executor, 4, "nvfp4_cutlass")
    layer.use_ep_shared_allreduce = layer.use_unified_tp_allreduce = False
    layer.fake_balance_expert = layer.correction_bias = None
    layer.top_k = 2
    layer.select_topk = SelectTopk(config)
    router_kn = torch.zeros(2560, 4, dtype=torch.bfloat16, device=device)
    router_kn[:4] = torch.eye(4, device=device, dtype=torch.bfloat16) * 8
    layer.gate = CudaF16Linear(router_kn)
    return layer


def _install(layer, dispatch):
    from rtp_llm.models_py.modules.qwen4_exp.tp_reduce import (
        install_target_graph_all_reduce,
    )

    install_target_graph_all_reduce(SimpleNamespace(), layer, dispatch)

    def forbidden(*args, **kwargs):
        raise AssertionError("router reduced before shared merge")

    layer.fused_moe.router.tp_all_reduce = forbidden


def _equal_bits(actual, expected):
    assert torch.equal(
        actual.detach().cpu().contiguous().view(torch.uint8),
        expected.detach().cpu().contiguous().view(torch.uint8),
    ), "source BF16 storage bits changed"


def _pack_check(layer, source, rank):
    up_before = layer.shared_expert.up_proj.weight
    down_before = layer.shared_expert.down_proj.weight
    assert shared_tp2.maybe_prepare_bf16_shared_tp2(layer), "helper silently fell back"
    up = layer.shared_expert.up_proj.weight
    down = layer.shared_expert.down_proj.weight
    assert type(layer.shared_expert.up_proj) is shared_tp2._Qwen4SharedTp2Up
    assert up is layer.shared_expert.up_proj._packed_weight
    assert up.shape == (640, 2560) and up.stride() == (2560, 1)
    assert down.shape == (2560, 320) and down.stride() == (1, 2560)
    assert up.untyped_storage().nbytes() == 3276800
    assert down.untyped_storage().nbytes() == 3276800  # retained full owner
    assert up.untyped_storage().data_ptr() != up_before.untyped_storage().data_ptr()
    assert down.untyped_storage().data_ptr() == down_before.untyped_storage().data_ptr()
    assert down.storage_offset() == rank * 320 * 2560
    parts = [torch.empty_like(up.contiguous()) for _ in range(2)]
    dist.all_gather(parts, up.contiguous())
    # Reassemble all 1280 non-square rows; gate halves, then up halves.
    rebuilt = torch.cat(
        (parts[0][:320], parts[1][:320], parts[0][320:], parts[1][320:])
    )
    _equal_bits(rebuilt, source["up_kn"].t())
    dp = [torch.empty_like(down.contiguous()) for _ in range(2)]
    dist.all_gather(dp, down.contiguous())
    _equal_bits(torch.cat(dp, dim=1), source["down_kn"].t())
    _equal_bits(up_before, source["up_kn"].t())
    _equal_bits(down_before, source["down_kn"].t())
    _equal_bits(layer.shared_expert_gate.weight, source["gate_kn"].t())
    _same_on_both(layer.shared_expert_gate.weight)
    assert layer.ffn_tp_size == 1 and layer.use_unified_tp_allreduce
    assert not layer.use_ep_shared_allreduce
    assert layer._shared_stream is None and layer._reused_shared_stream is None


def _hidden(rows, generation, device):
    generator = torch.Generator().manual_seed(990 + generation + rows)
    value = (torch.randn(rows, 2560, generator=generator) * 0.2).bfloat16().to(device)
    patterns = ((2, 1, -2, -3), (-3, -2, 2, 1), (2, -2, 1, -3))
    value[:, :4] = torch.tensor(patterns[generation % 3], device=device).bfloat16()
    live = rows - (1 + generation % 3) * 4 if rows > 4 else rows
    value[live:] = 0
    tags = torch.stack(
        (
            torch.arange(rows, device=device) // 4,
            generation * 4 + torch.arange(rows, device=device) % 4,
        ),
        dim=1,
    )
    tags[live:] = -1
    return value, tags, live


def _cpu_stage_oracle(hidden, source, routed_parts):
    """Independent full-width FP64 projection, BF16 boundaries, then partition.

    Does not call DenseMLP, helper, router, GEMM CUDA, or Triton merge. This is a
    numerical budget gate, not an assertion of bitwise old/new equivalence.
    """
    h = hidden.detach().cpu().double()
    up64 = h @ source["up_kn"].double()
    up = up64.bfloat16().float()
    activated = (F.silu(up[:, :640]) * up[:, 640:]).bfloat16()
    gate = (h @ source["gate_kn"].double()).bfloat16().float()
    partials, merged = [], []
    for rank in range(2):
        indices = torch.arange(rank * 320, (rank + 1) * 320)
        partial = (
            activated[:, indices].double() @ source["down_kn"][indices].double()
        ).bfloat16()
        local = (
            torch.sigmoid(gate) * partial.float() + routed_parts[rank].cpu().float()
        ).bfloat16()
        partials.append(partial)
        merged.append(local)
    result = (merged[0].float() + merged[1].float()).bfloat16()
    full_math = F.silu(up64[:, :640]) * up64[:, 640:]
    full_math = torch.sigmoid(h @ source["gate_kn"].double()) * (
        full_math @ source["down_kn"].double()
    )
    full_math += routed_parts[0].cpu().double() + routed_parts[1].cpu().double()
    return result, partials, gate, full_math


def _numerical_check(actual, reference, label):
    # Existing NVFP4 executor comparison uses atol=5e-4, rtol=0.02. Keep that
    # budget; NEVER auto-fit tolerances to observations from a candidate run.
    torch.testing.assert_close(actual.cpu(), reference, atol=0.0005, rtol=0.02)
    error = (actual.cpu().double() - reference.double()).abs()
    print(
        json.dumps({"boundary": label, "max_abs_error": float(error.max())}), flush=True
    )


def _routes(layer, hidden):
    ids = torch.empty(hidden.shape[0], 2, dtype=torch.int32, device=hidden.device)
    weights = torch.empty_like(ids, dtype=torch.float32)
    layer.select_topk(layer.gate(hidden), ids, weights)
    _same_on_both(ids)
    _same_on_both(weights)
    return ids, weights


def _execute(executor, hidden, ids, weights):
    from rtp_llm.models_py.modules.factory.fused_moe.defs.fused_moe import (
        ExpertForwardPayload,
    )

    return executor.execute(
        ExpertForwardPayload(
            expert_x=hidden,
            expert_x_origin_dtype=torch.bfloat16,
            expert_topk_ids=ids,
            expert_topk_weights=weights,
        ),
        "SiGLU",
        None,
        None,
        False,
        None,
    ).fused_expert_output


def _fallback_checks(pc, source, executor):
    from rtp_llm.ops import RoleType

    layer = _layer(pc, source, executor)
    up, down = layer.shared_expert.up_proj.weight, layer.shared_expert.down_proj.weight
    with (
        patch.dict(os.environ, {shared_tp2._ENV: "0"}),
        patch.object(
            shared_tp2,
            "_eligible",
            side_effect=AssertionError("default probed metadata"),
        ),
        patch.object(
            dist, "all_gather_object", side_effect=AssertionError("default collective")
        ),
    ):
        assert not shared_tp2.maybe_prepare_bf16_shared_tp2(layer)
    assert layer.shared_expert.up_proj.weight is up
    assert layer.shared_expert.down_proj.weight is down
    changes = (
        (layer.config, "_qwen4_dense_storage", "fp8"),
        (layer.config, "is_mtp", True),
        (layer.parallelism_config, "role_type", RoleType.PREFILL),
        (layer, "use_ep_shared_allreduce", True),
        (layer, "_shared_stream", object()),
        (layer.shared_expert.up_proj, "weight", up.contiguous()),
        (layer.shared_expert.down_proj, "weight", down[:, :639]),
    )
    for obj, key, replacement in changes:
        old = getattr(obj, key)
        setattr(obj, key, replacement)
        try:
            assert not shared_tp2.maybe_prepare_bf16_shared_tp2(layer)
            assert not layer.use_unified_tp_allreduce
        finally:
            setattr(obj, key, old)
    assert layer.shared_expert.up_proj.weight is up
    assert layer.shared_expert.down_proj.weight is down


def _qualify_local_shared_buckets(layer, source, rank, tiny_calls):
    """Local shared consumer only; no new routed/NVFP4/IPC bucket claim."""
    shared = layer.shared_expert
    begin, end = rank * 320, (rank + 1) * 320
    indices = torch.arange(begin, end)
    selected = torch.cat((indices, indices + 640))
    expected_pack = source["up_kn"][:, selected].t().contiguous()
    _equal_bits(shared.up_proj.weight, expected_pack)
    _equal_bits(shared.down_proj.weight, source["down_kn"][indices].t())
    assert type(shared.up_proj) is shared_tp2._Qwen4SharedTp2Up
    assert shared.up_proj.weight.stride() == (2560, 1)
    assert shared.down_proj.weight.stride() == (1, 2560)
    weight_ptrs = (shared.up_proj.weight.data_ptr(), shared.down_proj.weight.data_ptr())
    stages = {}
    handles = [
        shared.up_proj.register_forward_hook(
            lambda _module, _args, out: stages.__setitem__("up", out)
        ),
        shared.act_fn.register_forward_hook(
            lambda _module, _args, out: stages.__setitem__("activation", out)
        ),
    ]

    def run(hidden):
        # Observe the original DenseMLP; no module forward or return replacement.
        stages.clear()
        before = len(tiny_calls)
        partial = shared(hidden, skip_allreduce=True)
        assert len(tiny_calls) == before + 1
        assert tiny_calls[-1]["rows"] == hidden.shape[0]
        return stages["up"], stages["activation"], partial

    def check_cpu_fp64(hidden, actual, rows, generation):
        # Independent source math on CPU, with an explicit BF16 store per stage.
        h64 = hidden.detach().cpu().double()
        up_bf16 = (h64 @ source["up_kn"].double()).bfloat16()
        up_ref = up_bf16[:, selected]
        activation_ref = (
            F.silu(up_ref[:, :320].double()) * up_ref[:, 320:].double()
        ).bfloat16()
        down_ref = (
            activation_ref.double() @ source["down_kn"][indices].double()
        ).bfloat16()
        label = f"local_shared_ONLY_r{rank}_M{rows}_g{generation}_"
        _numerical_check(actual[0], up_ref, label + "up_vs_CPU_FP64_then_BF16")
        _numerical_check(actual[1], activation_ref, label + "activation_staged_FP64")
        _numerical_check(actual[2], down_ref, label + "partial_staged_FP64")
        # Conditional stage oracles isolate propagation from the observed input;
        # they supplement, and never replace, the independent source path above.
        observed_up = actual[0].detach().cpu().double()
        activation_observed = (
            F.silu(observed_up[:, :320]) * observed_up[:, 320:]
        ).bfloat16()
        down_observed = (
            actual[1].detach().cpu().double() @ source["down_kn"][indices].double()
        ).bfloat16()
        _numerical_check(
            actual[1], activation_observed, label + "activation_given_observed_up"
        )
        _numerical_check(
            actual[2], down_observed, label + "partial_given_observed_activation"
        )

    try:
        for rows in (4, 8, 16, 32, 64, 128, 256):
            hidden, _, _ = _hidden(rows, 0, shared.up_proj.weight.device)
            input_ptr = hidden.data_ptr()
            eager = run(hidden)
            check_cpu_fp64(hidden, eager, rows, 0)
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):
                    run(hidden)
            stream.synchronize()
            graph = torch.cuda.CUDAGraph()
            try:
                with torch.cuda.graph(graph, stream=stream):
                    captured = run(hidden)
                assert tiny_calls[-1]["capturing"]
                # Multiples of3 keep _hidden's small-M live-row geometry valid,
                # while its CPU generator refreshes all non-padding contents.
                for generation in (3, 6):
                    replacement, _, _ = _hidden(rows, generation, hidden.device)
                    assert not torch.equal(hidden, replacement)
                    hidden.copy_(replacement)
                    assert hidden.data_ptr() == input_ptr
                    eager = run(hidden)
                    check_cpu_fp64(hidden, eager, rows, generation)
                    before = len(tiny_calls)
                    graph.replay()
                    assert (
                        len(tiny_calls) == before
                    ), "replay re-entered Python TinyGEMM"
                    for replayed, same_input_eager in zip(captured, eager):
                        torch.testing.assert_close(
                            replayed, same_input_eager, rtol=0, atol=0
                        )
                    assert weight_ptrs == (
                        shared.up_proj.weight.data_ptr(),
                        shared.down_proj.weight.data_ptr(),
                    )
                    print(
                        json.dumps(
                            {
                                "event": "local_shared_bucket_qualified",
                                "scope": "local_Tiny_up_original_act_down_ONLY_not_NVFP4_or_IPC",
                                "rank": rank,
                                "rows": rows,
                                "generation": generation,
                                "input_ptr_unchanged": input_ptr,
                                "up_shape": [rows, 640],
                                "activation_shape": [rows, 320],
                                "partial_shape": [rows, 2560],
                                "dtype": "BF16",
                                "oracle": "CPU_FP64_source_math_with_explicit_BF16_stage_stores",
                                "budget": "UNCHANGED_numerical_check_atol0.0005_rtol0.02",
                                "graph_vs_same_input_eager": "BITEXACT_up_activation_partial",
                                "new_collectives": 0,
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )
            finally:
                torch.cuda.current_stream().synchronize()
                graph.reset()
    finally:
        for handle in handles:
            handle.remove()


def _worker(rank, port):
    from rtp_llm.models_py.distributed import collective_torch as ct
    from rtp_llm.models_py.distributed.flashinfer_all_reduce import (
        get_flashinfer_allreduce,
    )
    from rtp_llm.models_py.modules.qwen4_exp.tp_reduce import Qwen4TargetReduce
    from rtp_llm.ops import NcclCommConfig

    torch.set_num_threads(2)
    torch.cuda.set_device(rank)
    assert torch.cuda.get_device_capability() == (12, 0)
    device = torch.device("cuda", rank)
    pc = _pc(rank)
    # Use the real fresh RTP initializer: it establishes the WORLD/TP map and
    # the real model communicator before installing any layer reduction.
    nccl = NcclCommConfig()
    nccl.nccl_ip = "127.0.0.1"
    ct.init_distributed_environment(
        pc, nccl, port, disable_custom_all_reduce=False, enable_sm120_collectives=True
    )
    communicator = get_flashinfer_allreduce()
    assert communicator is not None and communicator.cpp_small_enabled
    communicator.allow_ipc_capture = True
    reducer = Qwen4TargetReduce(communicator)
    reducer.active = True
    ipc_calls = []
    ipc_all_reduce = reducer.workspace.all_reduce

    def record_target_ipc(value, *args, **kwargs):
        ipc_calls.append((tuple(value.shape), value.data_ptr()))
        return ipc_all_reduce(value, *args, **kwargs)

    reducer.workspace.all_reduce = record_target_ipc
    source = _source()
    # Decode CUTLASS is required for this consumer qualification.
    weights = _nvfp4_weights(device)
    executor, full = _executor(pc, weights), _executor(pc, weights, full=True)
    graphs = []
    # Retain all layer/Graph/input owners until every IPC registration is freed.
    owners = []
    try:
        with (
            patch.dict(os.environ, _ENV),
            _record_installed_tinygemm(rank) as tiny_calls,
        ):
            _fallback_checks(pc, source, executor)
            for producer in ("isolated_gpu_routed_tensor_NOT_NVFP4", "real_nvfp4_ep2"):
                layer = _layer(pc, source, executor)
                _pack_check(layer, source, rank)
                calls = []

                def dispatch(value, group):
                    assert group == ct.Group.TP
                    calls.append((tuple(value.shape), value.dtype, value.data_ptr()))
                    return reducer(value, group)

                _install(layer, dispatch)
                if producer.startswith("isolated"):
                    layer._forward_routed = lambda h: h * (
                        0.375 if rank == 0 else -0.125
                    )
                local_graphs = []
                for rows in (16, 256):  # C4/S4 and C64/S4 target buckets
                    hidden, tags, live = _hidden(rows, 0, device)
                    _same_on_both(hidden)
                    _same_on_both(tags)
                    _same_on_both(layer.shared_expert_gate(hidden))
                    if producer.startswith("isolated"):
                        routed = hidden * (0.375 if rank == 0 else -0.125)
                    else:
                        ids, route_weights = _routes(layer, hidden)
                        assert bool((ids < 2).all()), "rank0-only route case missing"
                        routed = _execute(executor, hidden, ids, route_weights)
                        if rank == 1:
                            assert not bool(
                                routed.any()
                            ), "nonlocal EP output was not zero"
                        rp = [torch.empty_like(routed) for _ in range(2)]
                        dist.all_gather(rp, routed)
                        _numerical_check(
                            rp[0].float() + rp[1].float(),
                            _execute(full, hidden, ids, route_weights).float().cpu(),
                            "actual_EP2_routed_sum_vs_same_quant_EP1",
                        )
                    routed_parts = [torch.empty_like(routed) for _ in range(2)]
                    dist.all_gather(routed_parts, routed)
                    expected, partials, gate, full_math = _cpu_stage_oracle(
                        hidden, source, routed_parts
                    )
                    actual_partial = layer.shared_expert(hidden, skip_allreduce=True)
                    _numerical_check(
                        actual_partial,
                        partials[rank],
                        "partial_down",
                    )
                    _numerical_check(
                        layer.shared_expert_gate(hidden), gate.bfloat16(), "BF16_gate"
                    )
                    before = len(calls)
                    ipc_before = len(ipc_calls)
                    tiny_before = len(tiny_calls)
                    actual = layer(hidden)
                    assert (
                        len(tiny_calls) == tiny_before + 1
                    ), "eager must call real TinyGEMM once"
                    assert not tiny_calls[-1]["capturing"]
                    assert len(calls) == before + 1, "expected one merge-after TP sum"
                    assert len(ipc_calls) == ipc_before + int(
                        rows == 256
                    ), "wrong target IPC dispatch"
                    _numerical_check(
                        actual, expected, producer + "_all_output_elements"
                    )
                    _same_on_both(actual)
                    assert not bool(
                        actual[live:].any()
                    ), "zero dummy/padding rows changed"
                    print(
                        json.dumps(
                            {
                                "producer": producer,
                                "rows": rows,
                                "rank": rank,
                                "max_error_vs_FP64_shared_plus_observed_Rlocal": float(
                                    (actual.cpu().double() - full_math).abs().max()
                                ),
                                "collectives_per_eager_forward": 1,
                                "payload": [rows, 2560],
                                "ffn_tp_size": executor.tp_size,
                                "ep_size": executor.ep_size,
                            }
                        ),
                        flush=True,
                    )
                    stream = torch.cuda.Stream()
                    stream.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(stream):
                        for _ in range(3):
                            layer(hidden)
                    stream.synchronize()
                    graph = torch.cuda.CUDAGraph()
                    before = len(calls)
                    ipc_before = len(ipc_calls)
                    tiny_before = len(tiny_calls)
                    with torch.cuda.graph(graph, stream=stream):
                        output = layer(hidden)
                    assert (
                        len(tiny_calls) == tiny_before + 1
                    ), "capture must call real TinyGEMM once"
                    assert tiny_calls[-1]["capturing"]
                    assert (
                        len(calls) == before + 1
                    ), "capture has more than one TP sum site"
                    assert len(ipc_calls) == ipc_before + int(rows == 256)
                    if rows == 256:
                        assert reducer.workspace.capture_calls == 1
                    else:
                        from rtp_llm.models_py.distributed.flashinfer_ipc_all_reduce import (
                            FlashInferIpcAllReduce,
                        )

                        assert isinstance(
                            communicator._workspace, FlashInferIpcAllReduce
                        )
                        assert communicator._workspace.capture_calls == 1
                    communicator.finish_graph_capture()
                    reducer.finish_graph_capture()
                    local_graphs.append((graph, hidden, tags, output))
                    graphs.append(graph)
                # Both graph objects stay alive; alternate keys and refresh all
                # actual layer inputs. Tags prove peer row mapping, not a hidden
                # metadata input to this stateless MLP.
                refresh = torch.cuda.Stream()
                for generation, key in enumerate((0, 1, 1, 0, 1, 0), 1):
                    graph, hidden, tags, output = local_graphs[key]
                    new_hidden, new_tags, live = _hidden(
                        hidden.shape[0], generation, device
                    )
                    refresh.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(refresh):
                        hidden.copy_(new_hidden)
                        tags.copy_(new_tags)
                    torch.cuda.current_stream().wait_stream(refresh)
                    _same_on_both(hidden)
                    _same_on_both(tags)
                    _same_on_both(layer.shared_expert_gate(hidden))
                    if producer == "real_nvfp4_ep2":
                        ids, route_weights = _routes(layer, hidden)
                        if generation % 3 == 1:
                            assert bool((ids[:live] >= 2).all())
                        if generation % 3 == 2:
                            assert bool((ids < 2).any()) and bool((ids >= 2).any())
                    eager = layer(hidden)
                    graph.replay()
                    torch.testing.assert_close(output, eager, rtol=0, atol=0)
                    _same_on_both(output)
                    assert not bool(output[live:].any())
                # Restart-only rule: a fresh source/pack/module/Graph is needed.
                # No old-source hot-update claim or WeightManager override.
                original_up_ptr = layer.shared_expert.up_proj.weight.data_ptr()
                before = layer.shared_expert.up_proj.weight.clone()
                changed = {name: value.clone() for name, value in source.items()}
                changed["up_kn"][:, :640].mul_(0.5)
                torch.testing.assert_close(
                    layer.shared_expert.up_proj.weight, before, rtol=0, atol=0
                )
                replacement = _layer(pc, changed, executor)
                _pack_check(replacement, changed, rank)
                assert not torch.equal(replacement.shared_expert.up_proj.weight, before)
                assert (
                    replacement.shared_expert.up_proj.weight.data_ptr()
                    != original_up_ptr
                )
                if producer.startswith("isolated"):
                    replacement._forward_routed = lambda h: h * (
                        0.375 if rank == 0 else -0.125
                    )
                _install(replacement, dispatch)
                hidden = local_graphs[0][1]
                for _ in range(3):
                    replacement(hidden)
                new_graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(new_graph):
                    new_output = replacement(hidden)
                communicator.finish_graph_capture()
                reducer.finish_graph_capture()
                graphs.append(new_graph)
                new_graph.replay()
                torch.testing.assert_close(
                    new_output, replacement(hidden), rtol=0, atol=0
                )
                assert not torch.equal(
                    new_output, layer(hidden)
                ), "restart did not consume changed source"
                for graph, old_hidden, _, old_output in local_graphs:
                    graph.replay()
                    torch.testing.assert_close(
                        old_output, layer(old_hidden), rtol=0, atol=0
                    )
                owners.append((layer, local_graphs, replacement, new_graph, new_output))
                assert communicator.cpp_small_enabled and reducer.workspace.handle
            local_bucket_collective_counts = (
                len(calls),
                len(ipc_calls),
                reducer.workspace.capture_calls,
                getattr(communicator._workspace, "capture_calls", None),
            )
            _qualify_local_shared_buckets(layer, source, rank, tiny_calls)
            assert local_bucket_collective_counts == (
                len(calls),
                len(ipc_calls),
                reducer.workspace.capture_calls,
                getattr(communicator._workspace, "capture_calls", None),
            ), "local shared qualification added a collective/capture site"
    finally:
        torch.cuda.synchronize()
        for graph in graphs:
            graph.reset()
        reducer.workspace.destroy()
        ct.destroy_distributed_environment()


class SharedExpertTp2IntegrationTest(unittest.TestCase):
    def test_bf16_oracle_and_real_nvfp4_ep2_two_live_graphs(self):
        # Wrong/missing device reservation is FAIL, never a silent skip.
        self.assertEqual(torch.cuda.device_count(), 2)
        self.assertIsNone(torch.version.hip)
        with socket.socket() as rendezvous:
            rendezvous.bind(("127.0.0.1", 0))
            port = rendezvous.getsockname()[1]
        mp.spawn(_worker, args=(port,), nprocs=2, join=True)


if __name__ == "__main__":
    unittest.main()

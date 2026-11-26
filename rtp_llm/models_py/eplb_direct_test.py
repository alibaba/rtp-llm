"""Real two-process FST reload and module transaction contracts.

This exercises the production Python balancer/module with real stage agreement,
not the C++ engine scheduling loop or a model golden. GPU runs use NCCL and CUDA Graph.
"""

import datetime
import json
import os
import tempfile
import threading
import types
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest import mock

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from safetensors.torch import save_file

from rtp_llm.models_py.checkpoint_reader_test_support import has_native_fst, requires_native_fst
from rtp_llm.models_py.eplb_controller import NewLoaderExpertBalancer
from rtp_llm.models_py.expert_placement import ExpertPlacement
from rtp_llm.models_py.fst_range_reader import iter_fst_range_tensors
from rtp_llm.models_py.layers.moe_experts import BaseMoEExperts
from rtp_llm.models_py.model_loader import (
    NewLoaderConfig,
    NewLoaderLoadMethod,
    NewModelLoader,
    _ExpertRangeFilter,
)
from rtp_llm.models_py.module_base import RtpModule
from rtp_llm.models_py.quant_methods.moe_layout import prepare_moe_weight_layout
from rtp_llm.models_py.quant_methods.unquantized import UnquantizedFusedMoEMethod
from rtp_llm.utils.model_weight import W

PREFIX = "model.layers.0.mlp.experts."
STAGE_VALUE = 0


class _AuxiliaryMethod(UnquantizedFusedMoEMethod):
    """Transport fixture for calibration/derived scales, NOT an FP4 executor."""

    def eplb_aux_weight_names(self):
        return ("test.calibration", "test.derived_scale")

    def process_weights_after_loading(self, layer):
        layer.test_calibration.copy_(layer.w13[:, 0, 0].float())
        layer.test_derived_scale.copy_(layer.test_calibration.reciprocal())

    def add_weight_tensors(self, layer, weights):
        weights["test.calibration"] = layer.test_calibration
        weights["test.derived_scale"] = layer.test_derived_scale


class _AuxiliaryExperts(BaseMoEExperts):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.quant_method = _AuxiliaryMethod()
        self.register_buffer("test_calibration", torch.zeros(self.num_local_experts))
        self.register_buffer("test_derived_scale", torch.zeros(self.num_local_experts))


class _OneLayerModel(RtpModule):
    def __init__(self, experts):
        super().__init__()
        self.experts = experts

    def resolve_checkpoint_load_plan(self, name, shape):
        return None, None


class _LocalExpertOutput(torch.nn.Module):
    """CPU/CUDA reference dispatch: read one known entry from each live down matrix."""

    def __init__(self, live_down, rank):
        super().__init__()
        self.live_down = live_down
        self.rank = rank

    def forward(self, *, hidden_states, topk_weights, topk_ids, **unused):
        self.last_physical = topk_ids
        local = topk_ids.to(torch.int64) - self.rank * self.live_down.shape[0]
        valid = (local >= 0) & (local < self.live_down.shape[0])
        values = self.live_down[:, 0, 0][local.clamp(0, self.live_down.shape[0] - 1)]
        return (values.float() * valid * topk_weights).sum(-1)


def _verify_forward(
    experts, logical, weights, placement, installed, traces, stage, group, stats
):
    logits, loads = stats
    logits.zero_()
    loads.zero_()
    rows = logical.tolist()
    expected = []
    for pos, ident in enumerate(x for row in rows for x in row):
        if ident < 0:
            expected.append(ident)
        elif installed:
            copies = [i for i, source in enumerate(placement) if source == ident]
            expected.append(copies[pos % len(copies)])
        else:
            canonical = (ident // 2) * 3 + ident % 2
            expected.append(
                canonical if placement[canonical] == ident else placement.index(ident)
            )
    expected = torch.tensor(expected, device=logical.device).reshape_as(logical)
    # Run the real module's forward (gating, stats, and routing); the dispatch
    # fixture consumes the actual live checkpoint-backed per-slot down matrix.
    output = experts(
        torch.zeros(len(rows), 32, device=logical.device), weights, logical
    )
    torch.testing.assert_close(experts.fused_moe.last_physical, expected)
    dist.all_reduce(output, group=group)
    oracle = torch.where(logical >= 0, 100 * stage + 10 * logical + 3, 0)
    torch.testing.assert_close(output, (oracle * weights).sum(-1), rtol=0, atol=0)
    torch.testing.assert_close(
        logits[stage].cpu().long(), logical[logical >= 0].cpu().bincount(minlength=4)
    )
    torch.testing.assert_close(
        loads[stage].cpu().long(),
        (expected[expected >= 0].cpu() // 3).bincount(minlength=2),
    )
    # Replay a graph captured before any plan. CPU tracing is not CUDA Graph QA.
    torch.testing.assert_close(traces(logical), expected)


def _checkpoint(dtype):
    return {
        f"{PREFIX}{expert}.{projection}.weight": torch.full(
            (32, 32), float(STAGE_VALUE + 10 * expert + index + 1), dtype=dtype
        )
        for expert in range(4)
        for index, projection in enumerate(("up_proj", "gate_proj", "down_proj"))
    }


def _snapshot(experts):
    return {
        name: (
            id(value),
            value.data_ptr(),
            value.stride(),
            value.detach().cpu().clone(),
        )
        for name, value in experts.eplb_runtime_weight_view().items()
    }


def _unchanged(experts, snapshot):
    for name, value in experts.eplb_runtime_weight_view().items():
        identity, pointer, stride, data = snapshot[name]
        assert (id(value), value.data_ptr(), value.stride()) == (
            identity,
            pointer,
            stride,
        )
        torch.testing.assert_close(value.cpu(), data, rtol=0, atol=0)


def _agree(ok, group=None):
    # Same two-counter contract as native LoadFlags: wait for all terminal
    # peers, then fail the transaction if any peer failed.
    device = "cuda" if dist.get_backend(group) == "nccl" else "cpu"
    status = torch.tensor([1, int(not ok)], dtype=torch.int32, device=device)
    dist.all_reduce(status, group=group)
    assert status[0].item() == dist.get_world_size(group)
    return status[1].item() == 0


def _expected(placement, rank, dtype, device, overlay):
    weights = _checkpoint(dtype)
    if overlay:
        weights[f"{PREFIX}1.up_proj.weight"].fill_(77)
    local = placement[rank * 3 : (rank + 1) * 3]
    w13 = torch.stack(
        [
            torch.cat(
                (
                    weights[f"{PREFIX}{logical}.up_proj.weight"],
                    weights[f"{PREFIX}{logical}.gate_proj.weight"],
                ),
                dim=0,
            )
            for logical in local
        ]
    ).to(device)
    w2 = torch.stack(
        [weights[f"{PREFIX}{logical}.down_proj.weight"] for logical in local]
    ).to(device)
    return {
        W.moe_w1: prepare_moe_weight_layout(w13, projection="gate_up"),
        W.moe_w2: prepare_moe_weight_layout(w2, projection="down"),
    }


@torch.inference_mode()
def _rank_worker(rank, rendezvous, folder, use_gpu, auxiliary, stages=1):
    global PREFIX, STAGE_VALUE
    world_rank = rank
    stage, rank = divmod(world_rank, 2)
    PREFIX = f"model.layers.{stage}.mlp.experts."
    STAGE_VALUE = 100 * stage
    torch.set_num_threads(1)
    device = "cpu"
    if use_gpu:
        torch.cuda.set_device(world_rank)
        device = f"cuda:{world_rank}"
        torch.cuda.reset_peak_memory_stats()
    backend = "nccl" if use_gpu else "gloo"
    dist.init_process_group(
        backend,
        init_method=f"file://{rendezvous}",
        rank=world_rank,
        world_size=2 * stages,
        timeout=datetime.timedelta(seconds=45),
    )
    try:
        stage_group = None
        if stages > 1:
            for index in range(stages):
                group = dist.new_group([2 * index, 2 * index + 1], backend=backend)
                if index == stage:
                    stage_group = group
            # Deliberately same-sized/ranked but spanning stages.
            cross_stage = None
            for members in ([0, 2], [1, 3]):
                group = dist.new_group(members, backend=backend)
                if world_rank in members:
                    cross_stage = group
        for dtype in (torch.float32, torch.bfloat16):
            dtype_name = str(dtype).split(".")[-1]
            checkpoint_path = os.path.join(folder, dtype_name)
            config = types.SimpleNamespace(
                activation_type="SiGLU",
                num_layers=stages,
                is_mtp=False,
                moe_n_group=1,
                ptuning_path=os.path.join(checkpoint_path, "overlay"),
                phy2log_path=(
                    os.path.join(folder, "initial.json") if auxiliary else None
                ),
                eplb_config=types.SimpleNamespace(
                    enable_eplb=True,
                    redundant_expert=2,
                    balance_method="round",
                    eplb_force_repack=1,
                    eplb_stats_window_size=2,
                ),
            )
            parallel = types.SimpleNamespace(
                world_size=2 * stages,
                world_rank=world_rank,
                local_world_size=2,
                dp_size=1,
                dp_rank=0,
                tp_size=2,
                tp_rank=rank,
                ep_size=2,
                ep_rank=rank,
                pp_size=stages,
                pp_rank=stage,
                pp_stage_layer_counts=[1] * stages,
            )
            expert_class = _AuxiliaryExperts if auxiliary else BaseMoEExperts
            experts = expert_class(
                num_experts=4,
                top_k=2,
                hidden_size=32,
                moe_intermediate_size=32,
                tp_size=2,
                tp_rank=rank,
                ep_size=2,
                ep_rank=rank,
                params_dtype=dtype,
                model_config=config,
                parallelism_config=parallel,
                moe_config=types.SimpleNamespace(),
                quant_config=None,
                layer_idx=stage,
            )
            initial = experts._expert_placement.physical_to_logical
            expert_filter = _ExpertRangeFilter(4, 2, rank, config, parallel)
            for name, value in iter_fst_range_tensors(
                [os.path.join(checkpoint_path, "model.safetensors")],
                name_filter=lambda name: name.startswith(PREFIX)
                and expert_filter.should_load(name),
            ):
                experts.load_weights({name[len(PREFIX) :]: value})
            experts.validate_weights_loaded()
            experts.to(device)
            experts.prepare_eplb_reload_weights()
            model = _OneLayerModel(experts)
            loader = NewModelLoader(
                config,
                NewLoaderConfig(
                    device=device,
                    compute_dtype=dtype,
                    ep_size=2,
                    ep_rank=rank,
                    tp_size=2,
                    tp_rank=rank,
                    parallelism_config=parallel,
                    load_method="fastsafetensors",
                ),
                model_path=checkpoint_path,
            )
            loader._selected_load_method = NewLoaderLoadMethod.FASTSAFETENSORS
            balancer = NewLoaderExpertBalancer(loader, model, config)
            if stages > 1:
                from rtp_llm.models_py.fastsafetensors_read_spec import (
                    validate_ep_process_group,
                )

                validate_ep_process_group(stage_group, parallel, 2, rank)
                for invalid in (cross_stage, dist.group.WORLD):
                    try:
                        validate_ep_process_group(invalid, parallel, 2, rank)
                    except RuntimeError:
                        pass
                    else:
                        raise AssertionError("foreign stage communicator accepted")
                # A stage-local planner must never select another stage's layer.
                assert balancer.moe_layer_index == (stage,)
                assert balancer.num_nodes == 1
                plan = balancer.create_balance_plan(
                    torch.ones(stages, 4, dtype=torch.int32),
                    torch.ones(stages, 2, dtype=torch.int32),
                )
                assert plan[0].item() == stage
            logical = torch.tensor([[0, 1], [2, 3], [0, -1]], device=device)
            router_weights = torch.tensor(
                [[0.2, 0.8], [0.25, 0.75], [1.0, 0.0]], device=device
            )
            experts.fused_moe = _LocalExpertOutput(
                experts.eplb_runtime_weight_view()[W.moe_w2], rank
            )
            stats = (
                torch.zeros(stages, 4, dtype=torch.int32, device=device),
                torch.zeros(stages, 2, dtype=torch.int32, device=device),
            )
            model._newloader_eplb_bind_stats(*stats)
            if use_gpu:
                # Capture actual device routing once, before the first install.
                # Plan publication must update the same captured storage.
                graph = torch.cuda.CUDAGraph()
                experts._map_logical_to_physical(logical)
                torch.cuda.synchronize()
                with torch.cuda.graph(graph):
                    captured = experts._map_logical_to_physical(logical)

                def trace_map(ids):
                    assert ids.data_ptr() == logical.data_ptr()
                    graph.replay()
                    return captured

            else:
                from torch.fx.experimental.proxy_tensor import make_fx

                trace_map = make_fx(lambda ids: experts._map_logical_to_physical(ids))(
                    logical
                )

            def verify():
                _verify_forward(
                    experts,
                    logical,
                    router_weights,
                    experts._expert_placement.physical_to_logical,
                    experts._eplb_plan_installed,
                    trace_map,
                    stage,
                    stage_group,
                    stats,
                )

            verify()
            identities = {
                name: (value.data_ptr(), value.stride())
                for name, value in experts.eplb_runtime_weight_view().items()
            }

            for phase, planned in (
                ("success", (2, 2, 1, 3, 0, 0)),
                ("install_failure", (0, 1, 2, 3, 0, 1)),
                ("load_failure", (0, 1, 2, 3, 0, 1)),
                ("success", (0, 1, 2, 3, 0, 1)),
            ):
                # Stage 1 succeeds while stage 0 fails. WORLD acknowledgements
                # would incorrectly roll back both stages (or strand peers).
                if stage > 0:
                    phase = "success"
                before = _snapshot(experts)
                old_placement = experts._expert_placement.physical_to_logical
                phy2log = torch.tensor(planned, dtype=torch.int32)
                placement = ExpertPlacement(4, 2, rank, planned)
                counts, inverse = experts._routing_tensors(placement)
                loaded = None
                load_ok = True
                io_started, continue_io = threading.Event(), threading.Event()

                def scheduled_ranges(*args, **kwargs):
                    io_started.set()
                    if not continue_io.wait(20):
                        raise TimeoutError("inference collective did not finish")
                    if phase == "load_failure" and rank == 1:
                        raise OSError("injected rank-local checkpoint failure")
                    yield from iter_fst_range_tensors(*args, **kwargs)

                with mock.patch(
                    "rtp_llm.models_py.weight_mapper.get_all_weights",
                    side_effect=AssertionError("no scratch fallback"),
                ), mock.patch(
                    "rtp_llm.models_py.fst_range_reader.iter_fst_range_tensors",
                    scheduled_ranges,
                ), ThreadPoolExecutor(
                    max_workers=1
                ) as executor:
                    future = executor.submit(
                        balancer.load_moe_weight,
                        torch.tensor([stage]),
                        rank,
                        2,
                        phy2log,
                    )
                    assert io_started.wait(20)
                    # The inference group progresses while the reload worker
                    # is active. The FST path must never join these collectives.
                    for step in range(4):
                        payload = torch.tensor([rank + 1.0], device=device)
                        dist.all_reduce(payload, group=stage_group)
                        assert payload.item() == 3.0
                        if step == 0:
                            continue_io.set()
                    try:
                        loaded = future.result(timeout=20)
                    except OSError:
                        load_ok = False
                _unchanged(experts, before)
                assert (
                    loader._selected_load_method == NewLoaderLoadMethod.FASTSAFETENSORS
                )
                if not _agree(load_ok, stage_group):
                    assert phase == "load_failure"
                    balancer.rollback_balance_plan(stage)
                    _unchanged(experts, before)
                    assert tuple(balancer.phy2log[stage]) == old_placement
                    verify()
                    continue

                assert loaded == stage
                weights = balancer._staged[1]
                assert ("test.derived_scale" in weights) == auxiliary
                install_ok = True
                try:
                    if phase == "install_failure" and rank == 1:
                        if auxiliary:
                            weights["test.derived_scale"] = weights[
                                "test.derived_scale"
                            ][:-1]
                        else:
                            weights[W.moe_w2] = weights[W.moe_w2][:-1]
                    balancer.install_balance_plan(stage, phy2log, counts, inverse)
                    assert experts._eplb_install_transaction is not None
                    # Re-entry must not discard the rollback snapshot.
                    try:
                        experts.apply_eplb_plan(
                            phy2log, counts, inverse, weights, defer_commit=True
                        )
                    except RuntimeError as error:
                        assert "acknowledgement" in str(error)
                    else:
                        raise AssertionError("pending install unexpectedly re-entered")
                except ValueError:
                    install_ok = False
                if _agree(install_ok, stage_group):
                    assert phase == "success"
                    balancer.commit_balance_plan(stage)
                    for name, expected in _expected(
                        planned, rank, dtype, device, True
                    ).items():
                        torch.testing.assert_close(
                            experts.eplb_runtime_weight_view()[name],
                            expected,
                            rtol=0,
                            atol=0,
                        )
                    if auxiliary:
                        expected_scale = torch.tensor(
                            [
                                (
                                    77.0
                                    if logical == 1
                                    else STAGE_VALUE + 10.0 * logical + 1
                                )
                                for logical in planned[rank * 3 : (rank + 1) * 3]
                            ],
                            device=device,
                        )
                        torch.testing.assert_close(
                            experts.test_calibration, expected_scale, rtol=0, atol=0
                        )
                        torch.testing.assert_close(
                            experts.test_derived_scale,
                            expected_scale.reciprocal(),
                            rtol=0,
                            atol=0,
                        )
                    assert tuple(balancer.phy2log[stage]) == planned
                else:
                    assert phase == "install_failure"
                    balancer.rollback_balance_plan(stage)
                    _unchanged(experts, before)
                    assert tuple(balancer.phy2log[stage]) == old_placement
                verify()
                assert experts._eplb_install_transaction is None
                assert {
                    name: (value.data_ptr(), value.stride())
                    for name, value in experts.eplb_runtime_weight_view().items()
                } == identities
            dist.barrier()
        if use_gpu:
            print(
                f"world_rank={world_rank} stage={stage} ep_rank={rank} peak_allocated={torch.cuda.max_memory_allocated()} peak_reserved={torch.cuda.max_memory_reserved()}",
                flush=True,
            )
    finally:
        dist.destroy_process_group()


class GraphBucketStatsTest(unittest.TestCase):
    def _check_bucket(self, device):
        from rtp_llm.models_py.layers.eplb_stats import ExpertRouteStats

        # Six of eight rows select the same *valid* expert, exactly as a zero
        # padding token can. They must not be counted for a two-token request.
        ids = torch.tensor([[0, 1], [1, -1]] + [[3, 3]] * 6, device=device)
        physical = torch.where(ids == 3, 0, 3)
        logical = torch.zeros(4, dtype=torch.int32, device=device)
        gpu = torch.zeros(2, dtype=torch.int32, device=device)
        bound = torch.zeros((), dtype=torch.int32, device=device)
        stats = ExpertRouteStats(
            logical, gpu, num_experts=4, ep_size=2, device=logical.device, live_rows=bound
        )

        def record():
            stats.record(ids, physical, 4)

        replay = record
        if device == "cuda":
            record()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                record()
            replay = graph.replay
        for n in (2, 5, 0, 8, 2):
            bound.fill_(n)
            logical.zero_()
            gpu.zero_()
            replay()
            valid = ids[:n] >= 0
            self.assertEqual(logical.tolist(), ids[:n][valid].bincount(minlength=4).tolist())
            self.assertEqual(gpu.tolist(), (physical[:n][valid] // 2).bincount(minlength=2).tolist())

    def test_real_rows_only(self):
        self._check_bucket("cpu")

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA Graph")
    def test_changing_real_rows_in_one_captured_graph(self):
        self._check_bucket("cuda")


class DistributedEplbReloadTest(unittest.TestCase):
    def _run(self, use_gpu, auxiliary=False, stages=1):
        with tempfile.TemporaryDirectory() as folder:
            if auxiliary:
                with open(os.path.join(folder, "initial.json"), "w") as stream:
                    json.dump([[2, 2, 1, 3, 0, 0]] * stages, stream)
            for dtype in (torch.float32, torch.bfloat16):
                checkpoint_path = os.path.join(folder, str(dtype).split(".")[-1])
                os.mkdir(checkpoint_path)
                tensors = {}
                for stage in range(stages):
                    tensors.update(
                        {
                            name.replace("layers.0.", f"layers.{stage}."): value
                            + 100 * stage
                            for name, value in _checkpoint(dtype).items()
                        }
                    )
                save_file(
                    tensors,
                    os.path.join(checkpoint_path, "model.safetensors"),
                )
                overlay_path = os.path.join(checkpoint_path, "overlay")
                os.mkdir(overlay_path)
                torch.save(
                    {
                        f"model.layers.{stage}.mlp.experts.1.up_proj.weight": torch.full(
                            (32, 32), 77.0, dtype=dtype
                        )
                        for stage in range(stages)
                    },
                    os.path.join(overlay_path, "pytorch_model.bin"),
                )
            mp.spawn(
                _rank_worker,
                args=(
                    os.path.join(folder, "rendezvous"),
                    folder,
                    use_gpu,
                    auxiliary,
                    stages,
                ),
                nprocs=2 * stages,
                join=True,
            )

    def test_three_logical_experts_on_four_physical_slots(self):
        from rtp_llm.models_py.fst_file_plan import plan_fst_files

        config = types.SimpleNamespace(num_layers=1, eplb_config=types.SimpleNamespace(redundant_expert=1))
        key = PREFIX + "down_proj.weight"
        rows = torch.arange(6).reshape(3, 2)
        with tempfile.TemporaryDirectory() as root:
            path = os.path.join(root, "model.safetensors")
            save_file({key: rows}, path)
            for rank, expected in enumerate(((0, 1), (0, 2))):
                f = _ExpertRangeFilter(3, 2, rank, config)
                self.assertEqual(plan_fst_files([path], root, types.SimpleNamespace(size=lambda: 1),
                                                None, None, expander=f), ([], [path]))
                if not has_native_fst():
                    continue
                got = dict(iter_fst_range_tensors([path], slice_expander=f, catalog=None,
                                                  name_filter=None, read_spec_resolver=None))
                self.assertEqual(set(got), {PREFIX + f"{i}.down_proj.weight" for i in expected})
                for i in expected:
                    torch.testing.assert_close(got[PREFIX + f"{i}.down_proj.weight"], rows[i])

    @requires_native_fst
    def test_two_cpu_ranks_real_fst_checkpoint_commit_and_rollback(self):
        self._run(False)

    @requires_native_fst
    def test_two_cpu_ranks_auxiliary_scale_commit_and_rollback(self):
        self._run(False, auxiliary=True)

    @requires_native_fst
    def test_two_stages_four_cpu_ranks_isolated_fst_aux_commit_and_rollback(self):
        self._run(False, auxiliary=True, stages=2)

    @unittest.skipUnless(
        os.environ.get("NEWLOADER_EPLB_TWO_GPU_TEST") == "1",
        "requires explicitly allocated two GPUs",
    )
    def test_two_gpu_ranks_real_layer_checkpoint_commit_and_rollback(self):
        self._run(True)
        self._run(True, auxiliary=True)

    @unittest.skipUnless(
        os.environ.get("NEWLOADER_EPLB_FOUR_GPU_STAGE_TEST") == "1",
        "requires explicitly allocated four GPUs; not a full-engine golden",
    )
    def test_two_stages_four_gpu_ranks_real_layer_aux_commit_and_rollback(self):
        self._run(True, auxiliary=True, stages=2)


if __name__ == "__main__":
    unittest.main()

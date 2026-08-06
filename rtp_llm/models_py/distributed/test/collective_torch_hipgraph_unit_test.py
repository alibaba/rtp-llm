import inspect
import sys
import unittest
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from rtp_llm.models_py.distributed import collective_torch as ct
from rtp_llm.models_py.distributed import rocm_rccl as rccl


def _record(pg, purpose="tp", generation=7, device=0, ranks=(0, 1)):
    return ct.GroupRecord(
        process_group=pg,
        ranks=ranks,
        backend="nccl" if purpose == "tp" else "gloo",
        device_index=device,
        owned_by_rtp=True,
        purpose=purpose,
        generation=generation,
    )


class ProcessGroupCommAccessorTest(unittest.TestCase):
    def test_dtype_map_and_unsupported_dtype(self):
        expected = {
            torch.int8: 0,
            torch.uint8: 1,
            torch.int32: 2,
            torch.int64: 4,
            torch.float16: 6,
            torch.float32: 7,
            torch.float64: 8,
            torch.bfloat16: 9,
        }
        for dtype, enum_value in expected.items():
            with self.subTest(dtype=dtype):
                self.assertEqual(
                    rccl._get_nccl_dtype(torch.zeros(1, dtype=dtype)), enum_value
                )
        with self.assertRaisesRegex(TypeError, "Unsupported dtype"):
            rccl._get_nccl_dtype(torch.zeros(1, dtype=torch.bool))

    def test_prefers_device_backend_accessor(self):
        backend = SimpleNamespace(_comm_ptr=MagicMock(return_value=1234))
        pg = SimpleNamespace(
            _get_backend=MagicMock(return_value=backend),
            _comm_ptr=MagicMock(return_value=9999),
        )
        record = _record(pg)

        self.assertEqual(rccl.ProcessGroupCommAccessor().extract(record), 1234)
        pg._get_backend.assert_called_once_with(torch.device("cuda", 0))
        pg._comm_ptr.assert_not_called()

    def test_falls_back_to_process_group_accessor(self):
        pg = SimpleNamespace(
            _get_backend=MagicMock(side_effect=RuntimeError("unsupported")),
            _comm_ptr=MagicMock(return_value=4321),
        )
        self.assertEqual(rccl.ProcessGroupCommAccessor().extract(_record(pg)), 4321)

    def test_zero_is_an_error_not_a_bootstrap_signal(self):
        backend = SimpleNamespace(_comm_ptr=MagicMock(return_value=0))
        pg = SimpleNamespace(
            _get_backend=MagicMock(return_value=backend),
            _comm_ptr=MagicMock(return_value=0),
        )
        with self.assertRaisesRegex(RuntimeError, "returned zero"):
            rccl.ProcessGroupCommAccessor().extract(_record(pg))

    def test_materialize_uses_both_process_group_primitives(self):
        accessor = rccl.ProcessGroupCommAccessor()
        record = _record(SimpleNamespace(_comm_ptr=MagicMock()))
        one = torch.ones(1)
        gathered = torch.empty(2)
        with patch("torch.ones", return_value=one), patch(
            "torch.empty", return_value=gathered
        ), patch("torch.distributed.all_reduce") as all_reduce, patch(
            "torch.distributed.all_gather_into_tensor"
        ) as all_gather, patch(
            "torch.cuda.synchronize"
        ) as synchronize:
            accessor.materialize(record)
        all_reduce.assert_called_once_with(one, group=record.process_group)
        all_gather.assert_called_once_with(gathered, one, group=record.process_group)
        synchronize.assert_called_once_with(0)

    def test_world_process_group_omits_device_id_when_policy_returns_none(self):
        with patch.object(rccl, "process_group_device_id", return_value=None), patch(
            "torch.distributed.init_process_group"
        ) as init:
            ct._init_world_process_group(
                "nccl", "tcp://host:1", 2, 0, object(), 0, False
            )
        self.assertNotIn("device_id", init.call_args.kwargs)

    def test_world_process_group_passes_rocm_device_id(self):
        device = torch.device("cuda", 1)
        with patch.object(rccl, "process_group_device_id", return_value=device), patch(
            "torch.distributed.init_process_group"
        ) as init:
            ct._init_world_process_group(
                "nccl", "tcp://host:1", 2, 1, object(), 1, True
            )
        self.assertEqual(init.call_args.kwargs["device_id"], device)

    def test_rocm_device_id_is_scoped_to_graph_required_groups(self):
        with patch.object(rccl, "_is_rocm_runtime", True):
            self.assertIsNone(rccl.process_group_device_id("nccl", 1, False))
            self.assertEqual(
                rccl.process_group_device_id("nccl", 1, True),
                torch.device("cuda", 1),
            )


class RcclGraphCommManagerTest(unittest.TestCase):
    def setUp(self):
        self.accessor = MagicMock()
        self.accessor.extract.return_value = 0x1234
        self.accessor.preflight.return_value = rccl.ProcessGroupPreflight(
            object(),
            object(),
            "device_backend._comm_ptr",
            MagicMock(return_value=0x1234),
        )
        self.manager = rccl.RcclGraphCommManager(self.accessor)
        self.tp = _record(object())
        self.control = _record(object(), purpose="graph_control", device=None)

    def _prepare(self, mutate=None):
        def gather(output, local, group):
            del group
            values = [dict(local), dict(local)]
            values[1]["rank"] = 1
            if mutate is not None:
                mutate(values[1])
            output[:] = values

        fake_lib = SimpleNamespace(ncclAllReduce=MagicMock(), ncclAllGather=MagicMock())
        with patch.object(rccl, "_load_rccl", return_value=fake_lib), patch.object(
            rccl, "_setup_rccl_api"
        ), patch("torch.distributed.get_rank", side_effect=lambda group=None: 0), patch(
            "torch.distributed.all_gather_object", side_effect=gather
        ):
            return self.manager.prepare(self.tp, self.control)

    def test_consensus_publishes_borrowed_descriptor(self):
        descriptor = self._prepare()
        self.assertEqual(descriptor.handle, 0x1234)
        self.assertIs(descriptor.source_group, self.tp)
        self.assertEqual(self.manager.state, rccl.ManagerState.READY)
        self.accessor.preflight.assert_called_once_with(self.tp)
        self.accessor.materialize.assert_called_once()
        self.assertIs(self.accessor.materialize.call_args.args[0], self.tp)
        self.accessor.extract.assert_called_once_with(
            self.tp, self.accessor.preflight.return_value
        )

    def test_prepare_is_idempotent_only_for_identical_records(self):
        descriptor = self._prepare()
        self.assertIs(self.manager.prepare(self.tp, self.control), descriptor)
        with self.assertRaisesRegex(RuntimeError, "different group"):
            self.manager.prepare(_record(object()), self.control)

    def test_one_rank_failure_is_uniform_and_never_published(self):
        with self.assertRaisesRegex(RuntimeError, "Uniform RCCL"):
            self._prepare(lambda remote: remote.update(success=False, error="zero"))
        self.assertEqual(self.manager.state, rccl.ManagerState.FAILED)
        self.assertIsNone(self.manager.descriptor)

    def test_metadata_mismatch_fails_consensus(self):
        with self.assertRaisesRegex(RuntimeError, "Uniform RCCL"):
            self._prepare(lambda remote: remote.update(generation=8))

    def test_rank_order_mismatch_fails_uniform_preflight(self):
        with self.assertRaisesRegex(RuntimeError, "Uniform RCCL"):
            self._prepare(lambda remote: remote.update(tp_ranks=(1, 0)))
        self.accessor.materialize.assert_not_called()

    def test_preflight_collective_failure_marks_manager_failed(self):
        fake_lib = SimpleNamespace(ncclAllReduce=MagicMock(), ncclAllGather=MagicMock())
        with patch.object(rccl, "_load_rccl", return_value=fake_lib), patch.object(
            rccl, "_setup_rccl_api"
        ), patch("torch.distributed.get_rank", return_value=0), patch(
            "torch.distributed.all_gather_object",
            side_effect=RuntimeError("control group failed"),
        ):
            with self.assertRaisesRegex(RuntimeError, "control group failed"):
                self.manager.prepare(self.tp, self.control)
        self.assertEqual(self.manager.state, rccl.ManagerState.FAILED)
        self.assertIsNone(self.manager.descriptor)

    def test_live_owner_blocks_teardown(self):
        self._prepare()
        token = self.manager.acquire_graph_owner(owner_id=88)
        with self.assertRaisesRegex(RuntimeError, "88"):
            self.manager.shutdown()
        self.assertEqual(self.manager.state, rccl.ManagerState.READY)
        self.manager.release_graph_owner(token)
        self.manager.shutdown()
        self.assertEqual(self.manager.state, rccl.ManagerState.EMPTY)

    def test_stale_token_is_rejected(self):
        self._prepare()
        token = self.manager.acquire_graph_owner()
        with self.assertRaisesRegex(RuntimeError, "stale"):
            self.manager.prepare_arena(token.token_id, token.generation + 1)

    def test_capture_revalidates_borrowed_process_group_communicator(self):
        self._prepare()
        token = self.manager.acquire_graph_owner()
        self.accessor.extract.return_value = 0x9999
        with self.assertRaisesRegex(RuntimeError, "communicator changed"):
            self.manager.enter_capture(token.token_id, token.generation)
        self.assertEqual(self.manager.state, rccl.ManagerState.FAILED)

    def test_capture_guard_failures_do_not_strand_capture_state(self):
        self._prepare()
        token = self.manager.acquire_graph_owner(owner_id=88)
        self.manager.enter_capture(token.token_id, token.generation)
        with self.assertRaisesRegex(RuntimeError, "stale"):
            self.manager.exit_capture(token.token_id, token.generation + 1)
        self.assertIsNone(self.manager.current_arena(capture_only=True))
        self.manager.enter_capture(token.token_id, token.generation)
        self.manager.exit_capture(token.token_id, token.generation)
        self.manager.release_graph_owner(token)


class CaptureArenaTest(unittest.TestCase):
    def test_repeated_signatures_use_distinct_occurrence_buffers(self):
        arena = rccl.CaptureArena(generation=3, graph_owner_id=4)
        first = (torch.Size((4, 8)), torch.float16, torch.device("cpu"))
        second = (torch.Size((8, 8)), torch.float16, torch.device("cpu"))
        arena.record(first)
        arena.record(first)
        arena.record(second)
        with patch.object(rccl, "_is_hipgraph_capture_active", return_value=False):
            arena.prepare()
        self.assertEqual(arena.required_signatures, [])
        arena.begin_capture()
        first_occurrence = arena.require(first)
        second_occurrence = arena.require(first)
        other_signature = arena.require(second)
        self.assertIsNot(first_occurrence, second_occurrence)
        self.assertIsNot(first_occurrence, other_signature)
        with self.assertRaisesRegex(RuntimeError, "occurrence 2"):
            arena.require(first)

    def test_generation_and_owner_are_properties_of_whole_arena(self):
        arena = rccl.CaptureArena(generation=3, graph_owner_id=4)
        arena.validate(rccl.GraphOwnerToken(1, 3, 4))
        with self.assertRaisesRegex(RuntimeError, "mismatch"):
            arena.validate(rccl.GraphOwnerToken(1, 4, 4))
        with self.assertRaisesRegex(RuntimeError, "mismatch"):
            arena.validate(rccl.GraphOwnerToken(1, 3, 5))

    def test_missing_capture_signature_never_allocates_lazily(self):
        arena = rccl.CaptureArena(generation=1, graph_owner_id=1)
        signature = (torch.Size((2,)), torch.float32, torch.device("cpu"))
        with self.assertRaisesRegex(RuntimeError, "not planned"):
            arena.require(signature)


class ProcessGroupTopologyOwnershipTest(unittest.TestCase):
    def _config(self, tp_size, dp_size, world_size):
        return SimpleNamespace(
            world_rank=0,
            world_size=world_size,
            local_rank=0,
            tp_size=tp_size,
            dp_size=dp_size,
        )

    def test_topology_ownership_and_teardown_matrix(self):
        topologies = ((1, 4, 4), (4, 1, 4), (2, 2, 4), (2, 1, 2))
        for tp_size, dp_size, world_size in topologies:
            for external_world in (False, True):
                for graph_required in (False, True):
                    case = (
                        f"tp={tp_size},dp={dp_size},world={world_size},"
                        f"external={external_world},graph={graph_required}"
                    )
                    with self.subTest(case=case):
                        self._run_topology_case(
                            tp_size,
                            dp_size,
                            world_size,
                            external_world,
                            graph_required,
                        )

    def _run_topology_case(
        self, tp_size, dp_size, world_size, external_world, graph_required
    ):
        ct._group_records.clear()
        ct._owned_group_creation_order.clear()
        ct._parallelism_config = None
        ct._initialized = False
        ct._world_owned_by_rtp = False
        ct._teardown_failed = False
        ct._graph_required_initialized = False

        config = self._config(tp_size, dp_size, world_size)
        world = torch.distributed.group.WORLD
        initialized = {"value": external_world}
        created_groups = []

        def init_process_group(**kwargs):
            initialized["value"] = True

        def new_group(**kwargs):
            group = SimpleNamespace(
                name=f"{kwargs['backend']}:{tuple(kwargs['ranks'])}",
                ranks=tuple(kwargs["ranks"]),
                backend=kwargs["backend"],
                timeout=kwargs["timeout"],
            )
            created_groups.append(group)
            return group

        def all_gather_object(output, local, group):
            del group
            output[:] = [local] * world_size

        descriptor = SimpleNamespace(
            device_index=0,
            generation=ct._distributed_generation,
        )
        fake_trt = SimpleNamespace(
            ensure_trtllm_comm_initialized=MagicMock(return_value=True),
            cleanup=MagicMock(),
        )
        fake_compute_ops = SimpleNamespace()
        fake_arch = SimpleNamespace(is_cuda=MagicMock(return_value=False))

        destroy = MagicMock()
        try:
            with patch.object(
                torch.distributed,
                "is_initialized",
                side_effect=lambda: initialized["value"],
            ), patch.object(
                torch.distributed,
                "init_process_group",
                side_effect=init_process_group,
            ) as init_world, patch.object(
                torch.distributed, "new_group", side_effect=new_group
            ), patch.object(
                torch.distributed, "get_rank", return_value=0
            ), patch.object(
                torch.distributed, "get_backend", return_value="nccl"
            ), patch.object(
                torch.distributed, "all_gather_object", side_effect=all_gather_object
            ), patch.object(
                torch.distributed, "barrier"
            ), patch.object(
                torch.distributed, "destroy_process_group", destroy
            ), patch.object(
                ct, "init_symm_mem_communicator"
            ), patch.object(
                ct, "destroy_symm_mem_communicator"
            ) as destroy_symm, patch.object(
                ct, "init_user_buffers_environment"
            ), patch.object(
                rccl, "_is_rocm_runtime", True
            ), patch.object(
                rccl, "prepare_distributed_environment"
            ), patch.object(
                ct, "_validate_graph_required_across_ranks"
            ), patch.object(
                rccl._graph_comm_manager, "prepare", return_value=descriptor
            ), patch.object(
                rccl, "assert_graph_comm_can_shutdown"
            ), patch.object(
                rccl, "shutdown_graph_comm"
            ), patch.dict(
                sys.modules,
                {
                    "librtp_compute_ops": fake_compute_ops,
                    "rtp_llm.models_py.utils.arch": fake_arch,
                    "rtp_llm.models_py.modules.base.rocm.trt_allreduce": fake_trt,
                },
            ):
                ct.init_distributed_environment(
                    config,
                    SimpleNamespace(nccl_ip="127.0.0.1"),
                    12345,
                    timeout=77,
                    graph_required=graph_required,
                )

                records = dict(ct._group_records)
                owned_creation_order = list(ct._owned_group_creation_order)
                world_owned = not external_world

                world_record = records[ct.Group.DP_AND_TP]
                self.assertEqual(world_record.purpose, "world")
                self.assertEqual(world_record.ranks, tuple(range(world_size)))
                self.assertEqual(world_record.owned_by_rtp, world_owned)

                isolated_tp = tp_size > 1 and (
                    world_size != tp_size or (external_world and graph_required)
                )
                tp_record = records[ct.Group.TP]
                self.assertEqual(
                    tp_record.ranks,
                    (
                        tuple(range(tp_size))
                        if tp_size > 1
                        else tuple(range(world_size))
                    ),
                )
                self.assertEqual(tp_record.purpose, "tp" if tp_size > 1 else "world")
                self.assertEqual(
                    tp_record.owned_by_rtp, True if isolated_tp else world_owned
                )
                self.assertEqual(tp_record.process_group is not world, isolated_tp)

                separate_dp = dp_size > 1 and world_size != dp_size
                dp_record = records[ct.Group.DP]
                self.assertEqual(
                    dp_record.ranks,
                    (
                        tuple(range(0, world_size, tp_size))
                        if separate_dp
                        else tuple(range(world_size))
                    ),
                )
                self.assertEqual(dp_record.purpose, "dp" if separate_dp else "world")
                self.assertEqual(
                    dp_record.owned_by_rtp, True if separate_dp else world_owned
                )

                has_control = graph_required and tp_size > 1
                self.assertEqual("GRAPH_CONTROL" in records, has_control)
                if has_control:
                    control = records["GRAPH_CONTROL"]
                    self.assertEqual(control.purpose, "graph_control")
                    self.assertEqual(control.ranks, tp_record.ranks)
                    self.assertTrue(control.owned_by_rtp)
                    self.assertIs(owned_creation_order[-1], control.process_group)

                expected_owned = []
                if separate_dp:
                    expected_owned.append(dp_record.process_group)
                if isolated_tp:
                    expected_owned.append(tp_record.process_group)
                if has_control:
                    expected_owned.append(records["GRAPH_CONTROL"].process_group)
                self.assertEqual(owned_creation_order, expected_owned)

                ct.destroy_distributed_environment()
                self.assertFalse(ct.distributed_environment_initialized())

            destroy_symm.assert_called_once_with()
            explicit_destroy = [
                invocation.args[0]
                for invocation in destroy.call_args_list
                if invocation.args
            ]
            self.assertEqual(explicit_destroy, list(reversed(owned_creation_order)))
            self.assertEqual(
                any(not invocation.args for invocation in destroy.call_args_list),
                not external_world,
            )
            if graph_required and tp_size > 1:
                self.assertIs(
                    explicit_destroy[0], records["GRAPH_CONTROL"].process_group
                )
            if external_world:
                init_world.assert_not_called()
            else:
                init_world.assert_called_once()
                self.assertEqual(
                    init_world.call_args.kwargs["timeout"], ct._data_group_timeout
                )
            for created_group in created_groups:
                expected_timeout = (
                    ct._DEFAULT_GRAPH_CONTROL_TIMEOUT
                    if created_group.backend == "gloo"
                    else ct._data_group_timeout
                )
                self.assertEqual(created_group.timeout, expected_timeout)
        finally:
            ct._group_records.clear()
            ct._owned_group_creation_order.clear()
            ct._parallelism_config = None
            ct._initialized = False
            ct._world_owned_by_rtp = False
            ct._teardown_failed = False
            ct._graph_required_initialized = False


class RoutingAndDeletionTest(unittest.TestCase):
    def setUp(self):
        self.old_manager = rccl._graph_comm_manager
        self.old_lib = rccl._rccl_lib
        self.old_graph_communication_required = rccl._graph_communication_required
        rccl._graph_communication_required = None
        self.tensor_guard = patch.object(rccl, "_validate_capture_tensor")
        self.tensor_guard.start()

    def tearDown(self):
        self.tensor_guard.stop()
        rccl._graph_comm_manager = self.old_manager
        rccl._rccl_lib = self.old_lib
        rccl._graph_communication_required = self.old_graph_communication_required

    def _ready_manager(self):
        accessor = MagicMock()
        accessor.extract.return_value = 0x55
        accessor.preflight.return_value = rccl.ProcessGroupPreflight(
            object(), object(), "device_backend._comm_ptr", MagicMock(return_value=0x55)
        )
        manager = rccl.RcclGraphCommManager(accessor)
        tp = _record(object(), ranks=(0, 1))
        control = _record(object(), purpose="graph_control", device=None)
        fake_lib = SimpleNamespace(ncclAllReduce=MagicMock(), ncclAllGather=MagicMock())

        def gather(output, local, group):
            del group
            remote = dict(local)
            remote["rank"] = 1
            output[:] = [dict(local), remote]

        with patch.object(rccl, "_load_rccl", return_value=fake_lib), patch.object(
            rccl, "_setup_rccl_api"
        ), patch("torch.distributed.get_rank", return_value=0), patch(
            "torch.distributed.all_gather_object", side_effect=gather
        ):
            manager.prepare(tp, control)
        token = manager.acquire_graph_owner(1)
        manager.begin_planning(token.token_id, token.generation)
        return manager, token

    def _enter_ready_capture(self):
        manager, token = self._ready_manager()
        rccl._graph_comm_manager = manager
        manager.enter_capture(token.token_id, token.generation)
        return manager, token

    def test_raw_all_gather_uses_preallocated_runner_arena(self):
        manager, token = self._ready_manager()
        rccl._graph_comm_manager = manager
        tensor = torch.zeros((2, 3), dtype=torch.float16)
        signature = (torch.Size((4, 3)), tensor.dtype, tensor.device)
        arena = manager.current_arena()
        arena.record(signature)
        with patch.object(rccl, "_is_hipgraph_capture_active", return_value=False):
            arena.prepare()
        manager.enter_capture(token.token_id, token.generation)
        lib = SimpleNamespace(ncclAllGather=MagicMock(return_value=0))
        rccl._rccl_lib = lib
        with patch("torch.cuda.current_device", return_value=0), patch(
            "torch.cuda.current_stream",
            return_value=SimpleNamespace(cuda_stream=9),
        ):
            output = rccl.hipgraph_capture_all_gather(tensor)
        self.assertIs(output, arena.buffers[(0, signature)])
        lib.ncclAllGather.assert_called_once()
        args = lib.ncclAllGather.call_args.args
        self.assertEqual(args[0], tensor.data_ptr())
        self.assertEqual(args[1], output.data_ptr())
        self.assertEqual(args[2], tensor.numel())
        self.assertEqual(args[3], rccl._NCCL_DTYPE_MAP[tensor.dtype])
        self.assertEqual(args[4].value, 0x55)
        self.assertEqual(args[5], 9)

    def test_trt_failure_does_not_fall_back_to_raw_rccl(self):
        manager, token = self._ready_manager()
        rccl._graph_comm_manager = manager
        manager.enter_capture(token.token_id, token.generation)
        raw = MagicMock(return_value=0)
        rccl._rccl_lib = SimpleNamespace(ncclAllReduce=raw)
        tensor = torch.zeros((1, 1024), dtype=torch.float16)
        fake_module = SimpleNamespace(
            _trtllm_comm_manager=SimpleNamespace(
                dist_env=SimpleNamespace(max_size_in_bytes=1 << 30)
            ),
            allreduce=MagicMock(side_effect=RuntimeError("TRT capture failed")),
            ALLREDUCE_SUPPORTED_HIDDEN_SIZES={1024},
            is_trt_allreduce_ready=lambda: True,
        )
        with patch.dict(
            sys.modules,
            {"rtp_llm.models_py.modules.base.rocm.trt_allreduce": fake_module},
        ), patch("torch.cuda.current_device", return_value=0):
            with self.assertRaisesRegex(RuntimeError, "TRT capture failed"):
                rccl.hipgraph_capture_all_reduce(tensor, process_group=object())
        raw.assert_not_called()

    def test_raw_all_reduce_error_code_is_reported(self):
        self._enter_ready_capture()
        rccl._rccl_lib = SimpleNamespace(ncclAllReduce=MagicMock(return_value=5))
        tensor = torch.zeros((4,), dtype=torch.float16)
        with patch("torch.cuda.current_device", return_value=0), patch(
            "torch.cuda.current_stream", return_value=SimpleNamespace(cuda_stream=9)
        ):
            with self.assertRaisesRegex(RuntimeError, "error code 5"):
                rccl.hipgraph_capture_all_reduce(tensor)

    def test_raw_all_gather_error_code_is_reported(self):
        manager, token = self._ready_manager()
        rccl._graph_comm_manager = manager
        tensor = torch.zeros((2, 3), dtype=torch.float16)
        signature = (torch.Size((4, 3)), tensor.dtype, tensor.device)
        manager.current_arena().record(signature)
        with patch.object(rccl, "_is_hipgraph_capture_active", return_value=False):
            manager.prepare_arena(token.token_id, token.generation)
        manager.enter_capture(token.token_id, token.generation)
        rccl._rccl_lib = SimpleNamespace(ncclAllGather=MagicMock(return_value=7))
        with patch("torch.cuda.current_device", return_value=0), patch(
            "torch.cuda.current_stream", return_value=SimpleNamespace(cuda_stream=9)
        ):
            with self.assertRaisesRegex(RuntimeError, "error code 7"):
                rccl.hipgraph_capture_all_gather(tensor)

    def _trt_module(self, cap, allreduce=None, pending=False):
        return SimpleNamespace(
            _trtllm_comm_manager=SimpleNamespace(
                dist_env=SimpleNamespace(max_size_in_bytes=cap)
            ),
            allreduce=allreduce
            or MagicMock(side_effect=lambda **kw: torch.empty_like(kw["allreduce_in"])),
            ALLREDUCE_SUPPORTED_HIDDEN_SIZES={1024},
            is_trt_allreduce_ready=lambda: True,
            has_pending_capture=MagicMock(return_value=pending),
            consume_capture=MagicMock(),
            finish_capture_session=MagicMock(),
        )

    def test_size_guard_uses_trt_when_tensor_size_equals_cap(self):
        self._enter_ready_capture()
        raw = MagicMock(return_value=0)
        rccl._rccl_lib = SimpleNamespace(ncclAllReduce=raw)
        tensor = torch.zeros((1, 1024), dtype=torch.float16)
        fake_module = self._trt_module(tensor.numel() * tensor.element_size())
        with patch.dict(
            sys.modules,
            {"rtp_llm.models_py.modules.base.rocm.trt_allreduce": fake_module},
        ), patch("torch.cuda.current_device", return_value=0):
            result = rccl.hipgraph_capture_all_reduce(tensor, object())
            self.assertIsNot(result, tensor)
            self.assertEqual(result.shape, tensor.shape)
            self.assertEqual(result.dtype, tensor.dtype)
        fake_module.allreduce.assert_called_once()
        raw.assert_not_called()

    def test_size_guard_uses_raw_rccl_when_tensor_exceeds_cap(self):
        self._enter_ready_capture()
        raw = MagicMock(return_value=0)
        rccl._rccl_lib = SimpleNamespace(ncclAllReduce=raw)
        tensor = torch.zeros((2, 1024), dtype=torch.float16)
        fake_module = self._trt_module(tensor.numel() * tensor.element_size() - 1)
        with patch.dict(
            sys.modules,
            {"rtp_llm.models_py.modules.base.rocm.trt_allreduce": fake_module},
        ), patch("torch.cuda.current_device", return_value=0), patch(
            "torch.cuda.current_stream", return_value=SimpleNamespace(cuda_stream=9)
        ):
            self.assertIs(rccl.hipgraph_capture_all_reduce(tensor, object()), tensor)
        fake_module.allreduce.assert_not_called()
        raw.assert_called_once()
        args = raw.call_args.args
        self.assertEqual(args[0], tensor.data_ptr())
        self.assertEqual(args[1], tensor.data_ptr())
        self.assertEqual(args[2], tensor.numel())
        self.assertEqual(args[3], rccl._NCCL_DTYPE_MAP[tensor.dtype])
        self.assertEqual(args[4], rccl._NCCL_SUM)
        self.assertEqual(args[5].value, 0x55)
        self.assertEqual(args[6], 9)

    def test_finish_capture_session_validates_owner_and_delegates(self):
        manager, token = self._ready_manager()
        rccl._graph_comm_manager = manager
        trt_module = self._trt_module(1)
        with patch.dict(
            sys.modules,
            {"rtp_llm.models_py.modules.base.rocm.trt_allreduce": trt_module},
        ):
            rccl.finish_hipgraph_capture_session(token.token_id, token.generation)
        trt_module.finish_capture_session.assert_called_once_with()

    def test_finish_capture_session_rejects_stale_owner_before_delegating(self):
        manager, token = self._ready_manager()
        rccl._graph_comm_manager = manager
        trt_module = self._trt_module(1)
        with patch.dict(
            sys.modules,
            {"rtp_llm.models_py.modules.base.rocm.trt_allreduce": trt_module},
        ):
            with self.assertRaisesRegex(RuntimeError, "Unknown or stale"):
                rccl.finish_hipgraph_capture_session(
                    token.token_id, token.generation + 1
                )
        trt_module.finish_capture_session.assert_not_called()

    def test_finish_capture_session_propagates_consume_error(self):
        manager, token = self._ready_manager()
        rccl._graph_comm_manager = manager
        fake_module = self._trt_module(1)
        fake_module.finish_capture_session.side_effect = RuntimeError("barrier timeout")
        with patch.dict(
            sys.modules,
            {"rtp_llm.models_py.modules.base.rocm.trt_allreduce": fake_module},
        ):
            with self.assertRaisesRegex(RuntimeError, "barrier timeout"):
                rccl.finish_hipgraph_capture_session(token.token_id, token.generation)

    def test_finish_capture_session_tolerates_missing_trt_module(self):
        manager, token = self._ready_manager()
        rccl._graph_comm_manager = manager
        with patch.dict(
            sys.modules,
            {"rtp_llm.models_py.modules.base.rocm.trt_allreduce": None},
        ):
            rccl.finish_hipgraph_capture_session(token.token_id, token.generation)

    def test_capture_routing_is_limited_to_tp_group(self):
        self._enter_ready_capture()
        rccl._rccl_lib = SimpleNamespace(ncclAllReduce=MagicMock(return_value=0))
        tensor = torch.zeros((4,), dtype=torch.float16)
        group = object()
        get_group = MagicMock(return_value=group)
        with patch.object(rccl, "_is_rocm_runtime", True), patch.object(
            rccl, "_is_hipgraph_capture_active", return_value=True
        ), patch("torch.cuda.current_device", return_value=0), patch(
            "torch.cuda.current_stream", return_value=SimpleNamespace(cuda_stream=9)
        ):
            self.assertIsNone(rccl.try_capture_all_reduce(tensor, False, get_group))
            get_group.assert_not_called()
            self.assertIs(rccl.try_capture_all_reduce(tensor, True, get_group), tensor)
        get_group.assert_called_once()
        rccl._rccl_lib.ncclAllReduce.assert_called_once()

    def test_public_collectives_route_only_tp_through_capture(self):
        manager, token = self._ready_manager()
        rccl._graph_comm_manager = manager
        tensor = torch.zeros((2, 3), dtype=torch.float16)
        signature = (torch.Size((4, 3)), tensor.dtype, tensor.device)
        manager.current_arena().record(signature)
        with patch.object(rccl, "_is_hipgraph_capture_active", return_value=False):
            manager.prepare_arena(token.token_id, token.generation)
        manager.enter_capture(token.token_id, token.generation)

        tp_group = object()
        dp_group = object()
        world_group = object()
        groups = {
            ct.Group.TP: tp_group,
            ct.Group.DP: dp_group,
            ct.Group.DP_AND_TP: world_group,
        }
        raw_reduce = MagicMock(return_value=0)
        raw_gather = MagicMock(return_value=0)
        rccl._rccl_lib = SimpleNamespace(
            ncclAllReduce=raw_reduce, ncclAllGather=raw_gather
        )

        with patch.object(rccl, "_is_rocm_runtime", True), patch.object(
            rccl, "_is_hipgraph_capture_active", return_value=True
        ), patch.object(ct, "_get_group", side_effect=groups.__getitem__), patch.object(
            ct, "get_symm_mem_communicator", return_value=None
        ), patch(
            "torch.cuda.current_device", return_value=0
        ), patch(
            "torch.cuda.current_stream", return_value=SimpleNamespace(cuda_stream=9)
        ), patch(
            "torch.distributed.get_world_size", return_value=2
        ), patch(
            "torch.distributed.all_reduce"
        ) as eager_reduce, patch(
            "torch.distributed.all_gather_into_tensor"
        ) as eager_gather:
            self.assertIs(ct.all_reduce(tensor, ct.Group.TP), tensor)
            gathered = ct.all_gather(tensor, ct.Group.TP)
            self.assertIs(gathered, manager.current_arena().buffers[(0, signature)])
            ct.all_reduce(tensor, ct.Group.DP)
            ct.all_gather(tensor, ct.Group.DP_AND_TP)

        raw_reduce.assert_called_once()
        raw_gather.assert_called_once()
        eager_reduce.assert_called_once_with(
            tensor, op=torch.distributed.ReduceOp.SUM, group=dp_group
        )
        eager_gather.assert_called_once()
        self.assertIs(eager_gather.call_args.kwargs["group"], world_group)

    def test_try_capture_all_gather_is_tp_only_and_requires_planned_arena(self):
        manager, token = self._enter_ready_capture()
        tensor = torch.zeros((2, 3), dtype=torch.float16)
        rccl._rccl_lib = SimpleNamespace(ncclAllGather=MagicMock(return_value=0))

        with patch.object(rccl, "_is_rocm_runtime", True), patch.object(
            rccl, "_is_hipgraph_capture_active", return_value=True
        ), patch("torch.cuda.current_device", return_value=0), patch(
            "torch.cuda.current_stream", return_value=SimpleNamespace(cuda_stream=9)
        ):
            self.assertIsNone(rccl.try_capture_all_gather(tensor, False))
            with self.assertRaisesRegex(RuntimeError, "not planned"):
                rccl.try_capture_all_gather(tensor, True)

        rccl._rccl_lib.ncclAllGather.assert_not_called()
        manager.exit_capture(token.token_id, token.generation)

        signature = (torch.Size((4, 3)), tensor.dtype, tensor.device)
        manager.current_arena().record(signature)
        with patch.object(rccl, "_is_hipgraph_capture_active", return_value=False):
            manager.prepare_arena(token.token_id, token.generation)
        manager.enter_capture(token.token_id, token.generation)
        with patch.object(rccl, "_is_rocm_runtime", True), patch.object(
            rccl, "_is_hipgraph_capture_active", return_value=True
        ), patch("torch.cuda.current_device", return_value=0), patch(
            "torch.cuda.current_stream", return_value=SimpleNamespace(cuda_stream=9)
        ):
            gathered = rccl.try_capture_all_gather(tensor, True)
            self.assertIs(gathered, manager.current_arena().buffers[(0, signature)])
        rccl._rccl_lib.ncclAllGather.assert_called_once()
        manager.exit_capture(token.token_id, token.generation)

    def test_active_tp_capture_fails_fast_when_manager_is_empty(self):
        manager = rccl.RcclGraphCommManager()
        with patch.object(rccl, "_graph_comm_manager", manager), patch.object(
            rccl, "_graph_communication_required", True
        ), patch.object(rccl, "_is_rocm_runtime", True), patch.object(
            rccl, "_is_hipgraph_capture_active", return_value=True
        ), patch(
            "torch.cuda.current_device", return_value=0
        ):
            with self.assertRaisesRegex(RuntimeError, "was not prepared"):
                rccl.try_capture_all_gather(torch.zeros(1), True)

    def test_eager_all_gather_records_signature_before_symm_mem_routing(self):
        tensor = torch.zeros((2, 3), dtype=torch.float16)
        process_group = object()
        gathered = torch.zeros((2, 2, 3), dtype=tensor.dtype)
        recorder = MagicMock()
        symm_mem = SimpleNamespace(
            should_torch_symm_mem_allgather=MagicMock(return_value=True),
            all_gather=MagicMock(return_value=gathered),
        )

        def get_symm_mem():
            recorder.assert_called_once_with(tensor, True, 2)
            return symm_mem

        with patch.object(rccl, "_is_rocm_runtime", True), patch.object(
            rccl, "_is_hipgraph_capture_active", return_value=False
        ), patch.object(ct, "_get_group", return_value=process_group), patch.object(
            rccl, "record_eager_allgather_signature", recorder
        ), patch.object(
            ct, "get_symm_mem_communicator", side_effect=get_symm_mem
        ), patch(
            "torch.distributed.get_world_size", return_value=2
        ):
            output = ct.all_gather(tensor, ct.Group.TP)

        self.assertEqual(output.shape, torch.Size((4, 3)))
        symm_mem.should_torch_symm_mem_allgather.assert_called_once_with(tensor)
        symm_mem.all_gather.assert_called_once_with(tensor)

    def test_real_planning_route_records_collective_torch_all_gather(self):
        manager, token = self._ready_manager()
        tensor = torch.zeros((2, 3), dtype=torch.float16)
        process_group = object()
        with patch.object(rccl, "_graph_comm_manager", manager), patch.object(
            rccl, "_is_rocm_runtime", True
        ), patch.object(
            rccl, "_is_hipgraph_capture_active", return_value=False
        ), patch.object(
            ct, "_get_group", return_value=process_group
        ), patch.object(
            ct, "get_symm_mem_communicator", return_value=None
        ), patch(
            "torch.distributed.get_world_size", return_value=2
        ), patch(
            "torch.distributed.all_gather_into_tensor"
        ):
            output = ct.all_gather(tensor, ct.Group.TP)

        self.assertEqual(output.shape, torch.Size((4, 3)))
        self.assertEqual(
            manager.current_arena().required_signatures,
            [rccl._signature(tensor, 2)],
        )
        manager.release_graph_owner(token)

    def test_cpp_graph_lifecycle_module_contract_and_zero_token_noops(self):
        expected_parameters = {
            "acquire_graph_owner": 1,
            "begin_capture_planning": 2,
            "cancel_capture_planning": 2,
            "prepare_capture_arena": 2,
            "enter_graph_capture_mode": 2,
            "exit_graph_capture_mode": 2,
            "release_graph_owner": 2,
            "release_graph_owner_after_acquire_failure": 1,
            "finish_hipgraph_capture_session": 2,
        }
        for name, parameter_count in expected_parameters.items():
            function = getattr(rccl, name)
            self.assertEqual(
                len(inspect.signature(function).parameters), parameter_count
            )
        if rccl.is_rocm_runtime():
            self.assertTrue(hasattr(rccl.rtp_llm_ops, "is_hipgraph_capture_enabled"))

        manager = rccl.RcclGraphCommManager()
        with patch.object(rccl, "_graph_comm_manager", manager), patch.object(
            manager, "begin_planning"
        ) as begin_planning, patch.object(
            manager, "cancel_planning"
        ) as cancel_planning, patch.object(
            manager, "prepare_arena"
        ) as prepare_arena, patch.object(
            manager, "enter_capture"
        ) as enter_capture, patch.object(
            manager, "exit_capture"
        ) as exit_capture, patch.object(
            manager, "release_owner"
        ) as release_owner:
            token = rccl.acquire_graph_owner(17)
            self.assertEqual(token, (0, 0))
            rccl.begin_capture_planning(*token)
            rccl.cancel_capture_planning(*token)
            rccl.prepare_capture_arena(*token)
            rccl.enter_graph_capture_mode(*token)
            rccl.exit_graph_capture_mode(*token)
            rccl.finish_hipgraph_capture_session(*token)
            rccl.release_graph_owner(*token)

        begin_planning.assert_not_called()
        cancel_planning.assert_not_called()
        prepare_arena.assert_not_called()
        enter_capture.assert_not_called()
        exit_capture.assert_not_called()
        release_owner.assert_not_called()

    def test_required_graph_communication_never_degenerates_to_zero_token(self):
        manager = rccl.RcclGraphCommManager()
        with patch.object(rccl, "_graph_comm_manager", manager), patch.object(
            rccl, "_graph_communication_required", True
        ):
            with self.assertRaisesRegex(RuntimeError, "required but was not prepared"):
                rccl.acquire_graph_owner(17)

    def test_active_capture_with_degenerate_tp_keeps_eager_routing(self):
        manager = rccl.RcclGraphCommManager()
        get_process_group = MagicMock()
        tensor = torch.zeros((2, 3), dtype=torch.float16)
        with patch.object(rccl, "_is_rocm_runtime", True), patch.object(
            rccl, "_graph_comm_manager", manager
        ), patch.object(rccl, "_graph_communication_required", False), patch.object(
            rccl, "_is_hipgraph_capture_active", return_value=True
        ):
            result = rccl.try_capture_all_reduce(
                tensor, is_tp_group=True, get_process_group=get_process_group
            )

        self.assertIsNone(result)
        get_process_group.assert_not_called()

    def test_undeclared_multi_rank_topology_never_degenerates_to_zero_token(self):
        manager = rccl.RcclGraphCommManager()
        with patch.object(rccl, "_graph_comm_manager", manager), patch.object(
            rccl, "_graph_communication_required", None
        ), patch("torch.distributed.is_initialized", return_value=True), patch(
            "torch.distributed.get_world_size", return_value=2
        ):
            with self.assertRaisesRegex(RuntimeError, "topology was not declared"):
                rccl.acquire_graph_owner(17)

    def test_release_of_other_owner_does_not_change_active_planning_arena(self):
        manager, first = self._ready_manager()
        second = manager.acquire_graph_owner(2)
        active = manager.current_arena()
        manager.release_graph_owner(second)
        self.assertIs(manager.current_arena(), active)
        tensor = torch.zeros((2, 3), dtype=torch.float16)
        with patch.object(rccl, "_graph_comm_manager", manager), patch.object(
            rccl, "_is_hipgraph_capture_active", return_value=False
        ):
            rccl.record_allgather_signature(tensor, 2)
        self.assertEqual(active.required_signatures, [rccl._signature(tensor, 2)])
        manager.release_graph_owner(first)

    def test_dedicated_bootstrap_and_destroy_bindings_are_gone(self):
        for obsolete in (
            "ncclGetUniqueId",
            "ncclCommInitRank",
            "ncclCommDestroy",
            "bootstrap_hipgraph_capture_rccl_comm_from_tp_group",
            "_rccl_comm_owned_by_python",
        ):
            self.assertFalse(hasattr(rccl, obsolete))

    def test_cuda_runtime_does_not_change_rocm_process_group_environment(self):
        config = SimpleNamespace(tp_size=2)
        with patch.object(rccl, "_is_rocm_runtime", False), patch.dict(
            "os.environ", {}, clear=True
        ):
            rccl.configure_rocm_pg_for_hipgraph(config)
            self.assertEqual(dict(__import__("os").environ), {})

    def test_rocm_process_group_environment_is_explicit_and_tp_gated(self):
        with patch.object(rccl, "_is_rocm_runtime", True), patch.dict(
            "os.environ", {}, clear=True
        ):
            rccl.configure_rocm_pg_for_hipgraph(SimpleNamespace(tp_size=2))
            self.assertEqual(
                dict(__import__("os").environ), rccl._HIPGRAPH_PROCESS_GROUP_ENV
            )
        with patch.object(rccl, "_is_rocm_runtime", True), patch.dict(
            "os.environ", {}, clear=True
        ):
            rccl.configure_rocm_pg_for_hipgraph(SimpleNamespace(tp_size=1))
            self.assertEqual(dict(__import__("os").environ), {})

    def test_non_graph_rocm_initialization_preserves_process_group_tuning(self):
        config = SimpleNamespace(tp_size=2, local_rank=0)
        with patch.object(rccl, "_is_rocm_runtime", True), patch.dict(
            "os.environ", {}, clear=True
        ), patch("torch.cuda.device_count", return_value=1), patch(
            "torch.cuda.set_device"
        ) as set_device:
            rccl.prepare_distributed_environment(config, graph_required=False)
            self.assertEqual(
                dict(__import__("os").environ), rccl._HIPGRAPH_PROCESS_GROUP_ENV
            )
        set_device.assert_called_once_with(0)

    def test_degenerate_tp_alias_is_not_registered_for_cpp_collectives(self):
        world = SimpleNamespace(size=MagicMock(return_value=2))
        fake_ops = SimpleNamespace(register_comm_ops=MagicMock())
        config = SimpleNamespace(tp_size=1, dp_size=2, world_size=2)
        records = {
            key: _record(world, purpose="world", ranks=(0, 1))
            for key in (ct.Group.DP_AND_TP, ct.Group.TP, ct.Group.DP)
        }
        with patch.dict(sys.modules, {"librtp_compute_ops": fake_ops}), patch.object(
            ct, "_parallelism_config", config
        ), patch.object(ct, "_group_records", records), patch(
            "torch.distributed.get_rank", return_value=0
        ):
            ct._register_process_groups_to_cpp()

        fake_ops.register_comm_ops.assert_called_once()
        cpp_allreduce = fake_ops.register_comm_ops.call_args.args[1]
        tensor = torch.ones(1)
        with patch("torch.cuda.current_device", return_value=0), patch(
            "torch.distributed.all_reduce"
        ) as all_reduce:
            self.assertIs(
                cpp_allreduce(tensor, 0, ct._CPP_PARALLEL_MODE_TP, None), tensor
            )
            all_reduce.assert_not_called()
            self.assertIs(
                cpp_allreduce(tensor, 0, ct._CPP_PARALLEL_MODE_DP, None), tensor
            )
            all_reduce.assert_called_once()
            self.assertIs(all_reduce.call_args.kwargs["group"], world)

    def test_cpp_tp_mode_uses_local_tp_purpose_record(self):
        world = SimpleNamespace(size=MagicMock(return_value=4))
        tp = SimpleNamespace(size=MagicMock(return_value=2))
        fake_ops = SimpleNamespace(register_comm_ops=MagicMock())
        config = SimpleNamespace(tp_size=2, dp_size=2, world_size=4)
        records = {
            ct.Group.DP_AND_TP: _record(world, purpose="world", ranks=(0, 1, 2, 3)),
            ct.Group.TP: _record(tp, purpose="tp", ranks=(2, 3)),
            "TP1": _record(tp, purpose="tp", ranks=(2, 3)),
        }
        with patch.dict(sys.modules, {"librtp_compute_ops": fake_ops}), patch.object(
            ct, "_parallelism_config", config
        ), patch.object(ct, "_group_records", records), patch(
            "torch.distributed.get_rank", return_value=2
        ):
            ct._register_process_groups_to_cpp()

        cpp_allreduce = fake_ops.register_comm_ops.call_args.args[1]
        tensor = SimpleNamespace(is_cuda=True)
        with patch("torch.cuda.current_device", return_value=2), patch(
            "torch.distributed.all_reduce"
        ) as all_reduce:
            self.assertIs(
                cpp_allreduce(tensor, 0, ct._CPP_PARALLEL_MODE_TP, None), tensor
            )
        self.assertIs(all_reduce.call_args.kwargs["group"], tp)


class ProductionGuardAndCleanupTest(unittest.TestCase):
    def test_get_group_never_attempts_partial_auto_initialization(self):
        with patch.object(ct, "_parallelism_config", object()), patch.object(
            ct, "_initialized", False
        ), patch("torch.distributed.is_initialized", return_value=False), patch.object(
            ct, "init_distributed_environment"
        ) as initialize:
            with self.assertRaisesRegex(RuntimeError, "rendezvous configuration"):
                ct._get_group(ct.Group.TP)
        initialize.assert_not_called()

    def test_graph_requirement_consensus_reuses_existing_world_group(self):
        world = torch.distributed.group.WORLD
        with patch.object(rccl, "_is_rocm_runtime", True), patch(
            "torch.distributed.get_world_size", return_value=2
        ), patch("torch.distributed.get_backend", return_value="gloo"), patch(
            "torch.distributed.all_reduce"
        ) as all_reduce, patch(
            "torch.distributed.new_group"
        ) as new_group:
            ct._validate_graph_required_across_ranks(True, 2)

        self.assertEqual(all_reduce.call_count, 2)
        self.assertEqual(
            [invocation.kwargs["op"] for invocation in all_reduce.call_args_list],
            [torch.distributed.ReduceOp.MIN, torch.distributed.ReduceOp.MAX],
        )
        self.assertTrue(
            all(
                invocation.kwargs["group"] is world
                for invocation in all_reduce.call_args_list
            )
        )
        new_group.assert_not_called()

    def test_graph_prepare_reentry_only_validates_existing_descriptor(self):
        descriptor = SimpleNamespace(device_index=0, generation=7)
        tp = _record(object())
        control = _record(object(), purpose="graph_control", device=None)
        registry = SimpleNamespace(get=MagicMock(return_value=control))
        manager = SimpleNamespace(
            state=rccl.ManagerState.READY,
            prepare=MagicMock(return_value=descriptor),
        )
        with patch.object(rccl, "_is_rocm_runtime", True), patch.object(
            rccl, "_graph_comm_manager", manager
        ), patch.object(torch.distributed, "all_gather_object") as all_gather:
            result = rccl.prepare_rocm_graph_communication(
                SimpleNamespace(tp_size=2), tp, registry, object()
            )
        self.assertIs(result, descriptor)
        manager.prepare.assert_called_once_with(tp, control)
        all_gather.assert_not_called()

    def test_graph_requirement_consensus_is_a_noop_outside_rocm(self):
        with patch.object(rccl, "_is_rocm_runtime", False), patch(
            "torch.distributed.all_reduce"
        ) as all_reduce:
            ct._validate_graph_required_across_ranks(True, 2)
        all_reduce.assert_not_called()

    def test_graph_requirement_consensus_detects_mixed_enablement(self):
        def reduce(tensor, op, group):
            self.assertIs(group, torch.distributed.group.WORLD)
            if op == torch.distributed.ReduceOp.MAX:
                tensor.copy_(torch.tensor([1, 2, 0], dtype=torch.int64))

        with patch.object(rccl, "_is_rocm_runtime", True), patch(
            "torch.distributed.get_world_size", return_value=2
        ), patch("torch.distributed.get_backend", return_value="gloo"), patch(
            "torch.distributed.all_reduce", side_effect=reduce
        ):
            with self.assertRaisesRegex(RuntimeError, "identical"):
                ct._validate_graph_required_across_ranks(False, 2)

    def test_missing_or_failing_capture_binding_is_fatal(self):
        with patch.object(rccl, "rtp_llm_ops", SimpleNamespace()):
            with self.assertRaisesRegex(RuntimeError, "runtime binding"):
                rccl._is_hipgraph_capture_active()
        with patch.object(
            rccl,
            "rtp_llm_ops",
            SimpleNamespace(
                is_hipgraph_capture_enabled=MagicMock(
                    side_effect=RuntimeError("binding failed")
                )
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "query HIPGraph"):
                rccl._is_hipgraph_capture_active()

    def test_capture_tensor_requires_cuda_contiguous_matching_device(self):
        descriptor = SimpleNamespace(device_index=1)
        manager = SimpleNamespace(require_ready=MagicMock(return_value=descriptor))
        with patch.object(rccl, "_graph_comm_manager", manager):
            with self.assertRaisesRegex(RuntimeError, "CUDA tensor"):
                rccl._validate_capture_tensor(torch.zeros(2), "input")
            with self.assertRaisesRegex(RuntimeError, "contiguous"):
                rccl._validate_capture_tensor(
                    SimpleNamespace(
                        is_cuda=True,
                        is_contiguous=MagicMock(return_value=False),
                    ),
                    "input",
                )
            with self.assertRaisesRegex(RuntimeError, "does not match"):
                rccl._validate_capture_tensor(
                    SimpleNamespace(
                        is_cuda=True,
                        is_contiguous=MagicMock(return_value=True),
                        device=SimpleNamespace(index=0),
                    ),
                    "input",
                )

    def test_trt_preparation_exception_is_fail_fast(self):
        descriptor = SimpleNamespace(device_index=0, generation=7)
        tp = _record(object())
        control = _record(object(), purpose="graph_control", device=None)
        registry = SimpleNamespace(get=MagicMock(return_value=control))
        fake_trt = SimpleNamespace(
            ensure_trtllm_comm_initialized=MagicMock(
                side_effect=RuntimeError("TRT unavailable")
            ),
            cleanup=MagicMock(),
        )
        with patch.object(rccl, "_is_rocm_runtime", True), patch.object(
            rccl._graph_comm_manager, "prepare", return_value=descriptor
        ), patch.dict(
            sys.modules,
            {"rtp_llm.models_py.modules.base.rocm.trt_allreduce": fake_trt},
        ):
            with self.assertRaisesRegex(RuntimeError, "TRT unavailable"):
                rccl.prepare_rocm_graph_communication(
                    SimpleNamespace(tp_size=2), tp, registry, object()
                )

    def test_trt_readiness_is_consensual_and_uniformly_cleaned_up(self):
        descriptor = SimpleNamespace(device_index=0, generation=7)
        tp = _record(object())
        control = _record(object(), purpose="graph_control", device=None)
        registry = SimpleNamespace(get=MagicMock(return_value=control))
        fake_trt = SimpleNamespace(
            ensure_trtllm_comm_initialized=MagicMock(return_value=True),
            cleanup=MagicMock(),
        )

        def gather(output, local, group):
            self.assertIs(group, control.process_group)
            self.assertTrue(local)
            output[:] = [True, False]

        with patch.object(rccl, "_is_rocm_runtime", True), patch.object(
            rccl._graph_comm_manager, "prepare", return_value=descriptor
        ), patch.object(
            torch.distributed, "all_gather_object", side_effect=gather
        ), patch.dict(
            sys.modules,
            {"rtp_llm.models_py.modules.base.rocm.trt_allreduce": fake_trt},
        ):
            result = rccl.prepare_rocm_graph_communication(
                SimpleNamespace(tp_size=2), tp, registry, object()
            )
        self.assertIs(result, descriptor)
        fake_trt.cleanup.assert_called_once_with()

    def test_graph_shutdown_failure_does_not_forget_manager_state(self):
        manager = SimpleNamespace(assert_can_shutdown=MagicMock(), shutdown=MagicMock())
        fake_trt = SimpleNamespace(
            cleanup=MagicMock(side_effect=RuntimeError("timeout"))
        )
        with patch.object(rccl, "_is_rocm_runtime", True), patch.object(
            rccl, "_graph_comm_manager", manager
        ), patch.dict(
            sys.modules,
            {"rtp_llm.models_py.modules.base.rocm.trt_allreduce": fake_trt},
        ):
            with self.assertRaisesRegex(RuntimeError, "timeout"):
                rccl.shutdown_graph_comm()
        manager.shutdown.assert_not_called()

    def test_clear_comm_ops_failure_does_not_skip_graph_or_registry_cleanup(self):
        old_state = (
            dict(ct._group_records),
            list(ct._owned_group_creation_order),
            ct._parallelism_config,
            ct._initialized,
            ct._world_owned_by_rtp,
            ct._teardown_failed,
        )
        fake_ops = SimpleNamespace(
            clear_comm_ops=MagicMock(side_effect=RuntimeError("clear failed"))
        )
        try:
            ct._group_records[ct.Group.TP] = _record(object())
            ct._initialized = True
            with patch.dict(sys.modules, {"librtp_compute_ops": fake_ops}), patch(
                "torch.distributed.is_initialized", return_value=False
            ), patch.object(rccl, "assert_graph_comm_can_shutdown"), patch.object(
                rccl, "shutdown_graph_comm"
            ) as shutdown, patch(
                "rtp_llm.models_py.utils.arch.is_cuda", return_value=False
            ):
                ct.destroy_distributed_environment()
            shutdown.assert_called_once()
            self.assertFalse(ct._initialized)
            self.assertEqual(ct._group_records, {})
        finally:
            records, owned, config, initialized, world_owned, teardown_failed = (
                old_state
            )
            ct._group_records.clear()
            ct._group_records.update(records)
            ct._owned_group_creation_order[:] = owned
            ct._parallelism_config = config
            ct._initialized = initialized
            ct._world_owned_by_rtp = world_owned
            ct._teardown_failed = teardown_failed

    def test_graph_comm_teardown_failure_marks_environment_terminal(self):
        old_state = (
            ct._parallelism_config,
            ct._initialized,
            ct._teardown_failed,
        )
        try:
            ct._parallelism_config = object()
            ct._initialized = True
            ct._teardown_failed = False
            with patch(
                "torch.distributed.is_initialized", return_value=False
            ), patch.object(rccl, "assert_graph_comm_can_shutdown"), patch.object(
                rccl,
                "shutdown_graph_comm",
                side_effect=RuntimeError("injected graph shutdown failure"),
            ), patch.object(
                ct, "destroy_symm_mem_communicator"
            ) as destroy_symm, patch(
                "rtp_llm.models_py.utils.arch.is_cuda", return_value=False
            ):
                with self.assertRaisesRegex(
                    RuntimeError, "injected graph shutdown failure"
                ):
                    ct.destroy_distributed_environment()

            self.assertFalse(ct._initialized)
            self.assertTrue(ct._teardown_failed)
            destroy_symm.assert_not_called()
            with self.assertRaisesRegex(RuntimeError, "teardown previously failed"):
                ct.init_distributed_environment(
                    SimpleNamespace(tp_size=1), SimpleNamespace(), 1
                )
        finally:
            (
                ct._parallelism_config,
                ct._initialized,
                ct._teardown_failed,
            ) = old_state

    def test_partial_group_teardown_preserves_only_failed_handles_for_retry(self):
        old_state = (
            dict(ct._group_records),
            list(ct._owned_group_creation_order),
            ct._parallelism_config,
            ct._initialized,
            ct._world_owned_by_rtp,
            ct._teardown_failed,
        )
        failed_group = object()
        destroyed_group = object()
        destroy = MagicMock(
            side_effect=lambda group=None: (
                (_ for _ in ()).throw(RuntimeError("injected destroy failure"))
                if group is failed_group
                else None
            )
        )
        try:
            ct._group_records.clear()
            ct._group_records.update(
                {
                    ct.Group.TP: _record(failed_group),
                    "GRAPH_CONTROL": _record(
                        destroyed_group, purpose="graph_control", device=None
                    ),
                }
            )
            ct._owned_group_creation_order[:] = [failed_group, destroyed_group]
            ct._parallelism_config = object()
            ct._initialized = True
            ct._world_owned_by_rtp = False
            ct._teardown_failed = False
            with patch("torch.distributed.is_initialized", return_value=True), patch(
                "torch.distributed.get_rank", return_value=0
            ), patch("torch.distributed.destroy_process_group", destroy), patch.object(
                rccl, "assert_graph_comm_can_shutdown"
            ), patch.object(
                rccl, "shutdown_graph_comm"
            ), patch.object(
                ct, "destroy_symm_mem_communicator"
            ) as destroy_symm, patch(
                "rtp_llm.models_py.utils.arch.is_cuda", return_value=False
            ):
                with self.assertRaisesRegex(RuntimeError, "no longer usable"):
                    ct.destroy_distributed_environment()

            self.assertFalse(ct._initialized)
            self.assertEqual(ct._owned_group_creation_order, [failed_group])
            self.assertEqual(set(ct._group_records), {ct.Group.TP})
            self.assertIs(ct._group_records[ct.Group.TP].process_group, failed_group)
            destroy_symm.assert_called_once_with()
            with self.assertRaisesRegex(RuntimeError, "teardown previously failed"):
                ct.init_distributed_environment(
                    SimpleNamespace(tp_size=1), SimpleNamespace(), 1
                )
        finally:
            records, owned, config, initialized, world_owned, teardown_failed = (
                old_state
            )
            ct._group_records.clear()
            ct._group_records.update(records)
            ct._owned_group_creation_order[:] = owned
            ct._parallelism_config = config
            ct._initialized = initialized
            ct._world_owned_by_rtp = world_owned
            ct._teardown_failed = teardown_failed

    def test_retry_after_post_world_init_failure_retains_world_ownership(self):
        old_state = (
            dict(ct._group_records),
            list(ct._owned_group_creation_order),
            ct._parallelism_config,
            ct._initialized,
            ct._world_owned_by_rtp,
            ct._graph_required_initialized,
        )
        initialized = {"value": False}
        config = SimpleNamespace(
            world_rank=0,
            world_size=1,
            local_rank=0,
            tp_size=1,
            dp_size=1,
        )

        def init_process_group(**kwargs):
            del kwargs
            initialized["value"] = True

        try:
            ct._group_records.clear()
            ct._owned_group_creation_order.clear()
            ct._parallelism_config = None
            ct._initialized = False
            ct._world_owned_by_rtp = False
            ct._graph_required_initialized = False
            with patch(
                "torch.distributed.is_initialized",
                side_effect=lambda: initialized["value"],
            ), patch(
                "torch.distributed.init_process_group",
                side_effect=init_process_group,
            ), patch(
                "torch.distributed.barrier",
                side_effect=[RuntimeError("post-init failure")],
            ), patch(
                "torch.distributed.get_backend", return_value="nccl"
            ), patch.object(
                rccl, "_is_rocm_runtime", False
            ), patch.object(
                rccl, "prepare_distributed_environment"
            ), patch.object(
                ct, "_register_process_groups_to_cpp"
            ):
                with self.assertRaisesRegex(RuntimeError, "post-init failure"):
                    ct.init_distributed_environment(
                        config,
                        SimpleNamespace(nccl_ip="127.0.0.1"),
                        12345,
                    )
                self.assertTrue(ct._world_owned_by_rtp)

                ct.init_distributed_environment(
                    config,
                    SimpleNamespace(nccl_ip="127.0.0.1"),
                    12345,
                )

            self.assertTrue(ct._initialized)
            self.assertTrue(ct._world_owned_by_rtp)
            self.assertTrue(ct._group_records[ct.Group.DP_AND_TP].owned_by_rtp)
        finally:
            (
                records,
                owned,
                parallelism_config,
                was_initialized,
                world_owned,
                graph_initialized,
            ) = old_state
            ct._group_records.clear()
            ct._group_records.update(records)
            ct._owned_group_creation_order[:] = owned
            ct._parallelism_config = parallelism_config
            ct._initialized = was_initialized
            ct._world_owned_by_rtp = world_owned
            ct._graph_required_initialized = graph_initialized

    def test_reused_environment_always_validates_graph_requirement(self):
        old_state = (
            dict(ct._group_records),
            ct._initialized,
            ct._graph_required_initialized,
        )
        try:
            ct._group_records[ct.Group.TP] = _record(object())
            ct._initialized = True
            ct._graph_required_initialized = False
            config = SimpleNamespace(world_size=2, tp_size=2)
            with patch(
                "torch.distributed.is_initialized", return_value=True
            ), patch.object(rccl, "_is_rocm_runtime", True), patch.object(
                ct, "_validate_graph_required_across_ranks"
            ) as validate:
                ct.init_distributed_environment(
                    config, SimpleNamespace(), 1, graph_required=False
                )
            validate.assert_called_once_with(False, 2, False)
        finally:
            records, initialized, graph_initialized = old_state
            ct._group_records.clear()
            ct._group_records.update(records)
            ct._initialized = initialized
            ct._graph_required_initialized = graph_initialized

    def test_reused_environment_rejects_graph_requirement_upgrade(self):
        old_state = (
            dict(ct._group_records),
            ct._initialized,
            ct._graph_required_initialized,
        )
        try:
            ct._group_records[ct.Group.TP] = _record(object())
            ct._initialized = True
            ct._graph_required_initialized = False
            config = SimpleNamespace(world_size=2, tp_size=2)
            with patch(
                "torch.distributed.is_initialized", return_value=True
            ), patch.object(rccl, "_is_rocm_runtime", True), patch.object(
                ct, "_validate_graph_required_across_ranks"
            ):
                with self.assertRaisesRegex(RuntimeError, "destroy.*reinitialize"):
                    ct.init_distributed_environment(
                        config, SimpleNamespace(), 1, graph_required=True
                    )
        finally:
            records, initialized, graph_initialized = old_state
            ct._group_records.clear()
            ct._group_records.update(records)
            ct._initialized = initialized
            ct._graph_required_initialized = graph_initialized

    def test_reused_environment_rejects_graph_requirement_downgrade(self):
        old_state = (
            dict(ct._group_records),
            ct._initialized,
            ct._graph_required_initialized,
        )
        try:
            ct._group_records[ct.Group.TP] = _record(object())
            ct._initialized = True
            ct._graph_required_initialized = True
            config = SimpleNamespace(world_size=2, tp_size=2)
            with patch(
                "torch.distributed.is_initialized", return_value=True
            ), patch.object(rccl, "_is_rocm_runtime", True), patch.object(
                ct, "_validate_graph_required_across_ranks"
            ):
                with self.assertRaisesRegex(RuntimeError, "Cannot disable"):
                    ct.init_distributed_environment(
                        config, SimpleNamespace(), 1, graph_required=False
                    )
        finally:
            records, initialized, graph_initialized = old_state
            ct._group_records.clear()
            ct._group_records.update(records)
            ct._initialized = initialized
            ct._graph_required_initialized = graph_initialized


if __name__ == "__main__":
    unittest.main()

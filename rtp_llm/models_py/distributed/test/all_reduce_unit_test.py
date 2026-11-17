import threading
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from rtp_llm.config.collective_config import CollectiveConfig
from rtp_llm.config.server_config_setup import set_parallelism_config
from rtp_llm.models_py.distributed import all_reduce as all_reduce_module
from rtp_llm.models_py.distributed import collective_torch as collective
from rtp_llm.models_py.distributed.flashinfer_ipc_all_reduce import (
    FlashInferIpcAllReduce,
    IpcInitializationError,
)
from rtp_llm.ops import ParallelismConfig


def topology(size=2):
    pc = ParallelismConfig()
    pc.tp_size = pc.ep_size = pc.world_size = pc.local_world_size = size
    set_parallelism_config(pc)
    return pc


def engine_config(max_tokens=32768, runtime_mb=1024, kv_mb=0):
    return SimpleNamespace(
        runtime_config=SimpleNamespace(
            fifo_scheduler_config=SimpleNamespace(max_batch_tokens_size=max_tokens),
            reserve_runtime_mem_mb=runtime_mb,
        ),
        kv_cache_config=SimpleNamespace(kv_cache_mem_mb=kv_mb),
    )


def model_config(hidden=2560, dtype=torch.bfloat16):
    return SimpleNamespace(hidden_size=hidden, compute_dtype=dtype)


class AllReduceUnitTest(unittest.TestCase):
    def setUp(self):
        self.group = object()
        self.device = torch.device("cuda", 0)
        self.options = CollectiveConfig("sm120_ipc")
        self.patches = [
            patch.object(all_reduce_module.dist, "get_world_size", return_value=2),
            patch.object(all_reduce_module.dist, "get_rank", return_value=0),
            patch.object(
                all_reduce_module, "exchange_group_state", side_effect=self.agree
            ),
        ]
        for item in self.patches:
            item.start()
        self.addCleanup(self.cleanup)

    @staticmethod
    def agree(group, value, tag):
        return [value, value]

    def cleanup(self):
        all_reduce_module.destroy_all_reduce()
        for item in reversed(self.patches):
            item.stop()

    def init(self, options=None):
        return all_reduce_module.init_all_reduce(
            self.group,
            self.device,
            topology(),
            options if options is not None else CollectiveConfig(),
        )

    def test_auto_selects_supported_ipc_without_a_model_name(self):
        with (
            patch.object(all_reduce_module, "_capability_error", return_value=None),
            patch.object(all_reduce_module, "IpcAllReduce") as factory,
        ):
            self.assertIs(self.init(), factory.return_value)
            self.assertEqual(factory.call_args.args[2].all_reduce_backend, "sm120_ipc")

    def test_expandable_allocator_falls_back_before_ipc_allocation(self):
        for backend in ("auto", "sm120_ipc"):
            with self.subTest(backend=backend):
                all_reduce_module.destroy_all_reduce()
                with (
                    patch.object(torch.cuda, "is_available", return_value=True),
                    patch.object(
                        torch.cuda, "get_device_capability", return_value=(12, 0)
                    ),
                    patch.object(torch.cuda, "device_count", return_value=2),
                    patch.object(
                        torch.cuda, "can_device_access_peer", return_value=True
                    ),
                    patch.object(
                        torch.cuda.memory,
                        "_snapshot",
                        return_value={
                            "allocator_settings": {"expandable_segments": True}
                        },
                    ),
                    patch.object(all_reduce_module, "IpcAllReduce") as factory,
                ):
                    self.assertIsNone(self.init(CollectiveConfig(backend)))
                factory.assert_not_called()

    def test_peer_expandable_allocator_keeps_all_ranks_on_nccl(self):
        def peer_failed(group, value, tag):
            if tag == "capabilities":
                return [value, [value[0], value[1], "expandable_segments"]]
            return self.agree(group, value, tag)

        with (
            patch.object(
                all_reduce_module, "exchange_group_state", side_effect=peer_failed
            ),
            patch.object(all_reduce_module, "_capability_error", return_value=None),
            patch.object(all_reduce_module, "IpcAllReduce") as factory,
        ):
            self.assertIsNone(self.init())
        factory.assert_not_called()

    def test_ipc_cleanup_waits_for_every_importer_and_retains_failed_closes(self):
        for fail_close in (False, True):
            with self.subTest(fail_close=fail_close):
                closed = set()
                freed = set()
                errors = []
                rank_zero_closed = threading.Event()
                barrier = threading.Barrier(2, timeout=10)
                verdicts = [False, False]
                workspaces = []
                for rank in range(2):
                    workspace = FlashInferIpcAllReduce.__new__(FlashInferIpcAllReduce)
                    workspace.group = rank
                    workspace.device = self.device
                    workspace.handle = rank + 1
                    workspace.rank_data = object()
                    workspace._opened = [1 - rank]
                    workspace._allocated = [rank]
                    workspace.ops = MagicMock()

                    def close(pointer, rank=rank):
                        if rank == 1:
                            self.assertTrue(rank_zero_closed.wait(5))
                        if rank == 1 and fail_close:
                            raise RuntimeError("injected close failure")
                        closed.add(rank)
                        if rank == 0:
                            rank_zero_closed.set()

                    def free(pointer):
                        self.assertEqual(closed, {0, 1})
                        freed.add(pointer)

                    workspace._close_mapping = close
                    workspace.ipc = SimpleNamespace(
                        cudart=SimpleNamespace(cudaFree=free)
                    )
                    workspaces.append(workspace)

                def agree_closed(group, value, tag):
                    self.assertEqual(tag, "ipc_close_mappings")
                    verdicts[group] = value
                    barrier.wait()
                    return list(verdicts)

                def destroy(workspace):
                    try:
                        workspace.destroy()
                    except BaseException as error:
                        errors.append(error)

                with (
                    patch.object(torch.cuda, "synchronize"),
                    patch.object(
                        all_reduce_module.dist,
                        "barrier",
                        side_effect=lambda **_: barrier.wait(),
                    ),
                    patch(
                        "rtp_llm.models_py.distributed.flashinfer_ipc_all_reduce.exchange_group_state",
                        side_effect=agree_closed,
                    ),
                ):
                    threads = [
                        threading.Thread(target=destroy, args=(w,)) for w in workspaces
                    ]
                    for thread in threads:
                        thread.start()
                    for thread in threads:
                        thread.join(15)
                    self.assertFalse(any(thread.is_alive() for thread in threads))
                self.assertEqual(errors, [])
                self.assertEqual(freed, set() if fail_close else {0, 1})
                self.assertEqual(
                    [len(w._allocated) for w in workspaces],
                    [1, 1] if fail_close else [0, 0],
                )

    def test_explicit_nccl_never_probes_or_allocates(self):
        with (
            patch.object(all_reduce_module, "_capability_error") as probe,
            patch.object(all_reduce_module, "IpcAllReduce") as factory,
            self.assertLogs(level="INFO") as logs,
        ):
            self.assertIsNone(self.init(CollectiveConfig("nccl")))
        probe.assert_not_called()
        factory.assert_not_called()
        self.assertTrue(all(record.levelname == "INFO" for record in logs.records))
        self.assertIn("requested_backend=nccl, effective_backend=nccl", logs.output[-1])

    def test_auto_unsupported_topology_falls_back(self):
        with (
            patch.object(all_reduce_module, "_capability_error") as probe,
            patch.object(all_reduce_module, "IpcAllReduce") as factory,
            self.assertLogs(level="INFO") as logs,
        ):
            self.assertIsNone(
                all_reduce_module.init_all_reduce(
                    self.group, self.device, topology(8), CollectiveConfig()
                )
            )
        probe.assert_not_called()
        factory.assert_not_called()
        self.assertTrue(all(record.levelname == "INFO" for record in logs.records))
        self.assertIn("requested_backend=auto, effective_backend=nccl", logs.output[-1])
        self.assertIn("unsupported topology", logs.output[-1])

    def test_auto_unsupported_capabilities_are_an_informational_selection(self):
        with (
            patch.object(
                all_reduce_module,
                "_capability_error",
                return_value="requires NVIDIA SM120",
            ),
            patch.object(all_reduce_module, "IpcAllReduce") as factory,
            self.assertLogs(level="INFO") as logs,
        ):
            self.assertIsNone(self.init())
        factory.assert_not_called()
        self.assertTrue(all(record.levelname == "INFO" for record in logs.records))
        self.assertIn("effective_backend=nccl", logs.output[-1])
        self.assertIn("requires NVIDIA SM120", logs.output[-1])

    def test_explicit_backend_capability_failure_is_a_warning(self):
        with (
            patch.object(
                all_reduce_module,
                "_capability_error",
                return_value="direct peer access unavailable",
            ),
            self.assertLogs(level="INFO") as logs,
        ):
            self.assertIsNone(self.init(self.options))
        warnings = [record for record in logs.records if record.levelname == "WARNING"]
        self.assertEqual(len(warnings), 1)
        self.assertIn("AllReduce sm120_ipc unavailable", warnings[0].getMessage())
        self.assertIn(
            "requested_backend=sm120_ipc, effective_backend=nccl", logs.output[-1]
        )

    def test_options_reject_invalid_combinations(self):
        for backend, large in (("invalid", False), ("nccl", True), ("sm120_p2p", True)):
            with self.subTest(backend=backend), self.assertRaises(ValueError):
                CollectiveConfig(backend, large)

    def test_explicit_wrong_topology_fails_before_probe(self):
        with patch.object(all_reduce_module, "_capability_error") as probe:
            with self.assertRaisesRegex(ValueError, "TP8"):
                self.init(CollectiveConfig("sm120_p2p"))
        probe.assert_not_called()

    def test_rank_policy_mismatch_fails_before_optional_initialization(self):
        def disagree(group, state, tag):
            return [state, ["nccl", None, None, False]]

        with (
            patch.object(
                all_reduce_module, "exchange_group_state", side_effect=disagree
            ),
            patch.object(all_reduce_module, "_capability_error") as probe,
            self.assertRaisesRegex(ValueError, "differs across TP ranks"),
        ):
            self.init(self.options)
        probe.assert_not_called()

    def test_one_rank_probe_failure_falls_back_before_allocation(self):
        def peer_failed(group, value, tag):
            if tag == "capabilities":
                return [value, [value[0], value[1], "missing peer API"]]
            return self.agree(group, value, tag)

        with (
            patch.object(
                all_reduce_module, "exchange_group_state", side_effect=peer_failed
            ),
            patch.object(all_reduce_module, "_capability_error", return_value=None),
            patch.object(all_reduce_module, "IpcAllReduce") as factory,
        ):
            self.assertIsNone(self.init(self.options))
        factory.assert_not_called()

    def test_auto_rank_topology_disagreement_falls_back_together(self):
        def disagree(group, state, tag):
            if tag == "capabilities":
                return [state, ["nccl", state[1], None]]
            return self.agree(group, state, tag)

        with (
            patch.object(
                all_reduce_module, "exchange_group_state", side_effect=disagree
            ),
            patch.object(all_reduce_module, "_capability_error", return_value=None),
            patch.object(all_reduce_module, "IpcAllReduce") as factory,
        ):
            self.assertIsNone(self.init())
        factory.assert_not_called()

    def test_agreed_local_setup_failure_can_fall_back(self):
        with (
            patch.object(all_reduce_module, "_capability_error", return_value=None),
            patch.object(
                all_reduce_module,
                "IpcAllReduce",
                side_effect=IpcInitializationError("allocation failed"),
            ),
        ):
            self.assertIsNone(self.init(self.options))

    def test_collective_failure_propagates(self):
        with (
            patch.object(all_reduce_module, "_capability_error", return_value=None),
            patch.object(
                all_reduce_module,
                "IpcAllReduce",
                side_effect=RuntimeError("collective failed"),
            ),
            self.assertRaisesRegex(RuntimeError, "collective failed"),
        ):
            self.init(self.options)

    def test_failed_ipc_construction_does_not_commit_policy_and_can_retry(self):
        communicator = MagicMock()
        with (
            patch.object(all_reduce_module, "_capability_error", return_value=None),
            patch.object(
                all_reduce_module,
                "IpcAllReduce",
                side_effect=[RuntimeError("collective failed"), communicator],
            ) as factory,
        ):
            with self.assertRaisesRegex(RuntimeError, "collective failed"):
                self.init(self.options)
            self.assertIsNone(all_reduce_module._group)
            self.assertIsNone(all_reduce_module._options)
            self.assertIsNone(all_reduce_module.get_all_reduce())
            self.assertIs(self.init(self.options), communicator)
            self.assertIs(self.init(self.options), communicator)
        self.assertEqual(factory.call_count, 2)

    def test_failed_capability_agreement_does_not_commit_policy_and_can_retry(self):
        def fail_capabilities(group, state, tag):
            if tag == "capabilities":
                raise RuntimeError("store failed")
            return self.agree(group, state, tag)

        with (
            patch.object(all_reduce_module, "_capability_error", return_value=None),
            patch.object(all_reduce_module, "IpcAllReduce") as factory,
        ):
            with (
                patch.object(
                    all_reduce_module,
                    "exchange_group_state",
                    side_effect=fail_capabilities,
                ),
                self.assertRaisesRegex(RuntimeError, "store failed"),
            ):
                self.init(self.options)
            self.assertIsNone(all_reduce_module._group)
            self.assertIsNone(all_reduce_module._options)
            factory.assert_not_called()
            self.assertIs(self.init(self.options), factory.return_value)
        factory.assert_called_once()

    def test_failed_p2p_construction_does_not_commit_policy_and_can_retry(self):
        communicator = MagicMock()
        policy = CollectiveConfig("sm120_p2p")
        with (
            patch.object(all_reduce_module, "_capability_error", return_value=None),
            patch(
                "rtp_llm.models_py.distributed.canonical_p2p.init_canonical_p2p",
                side_effect=[RuntimeError("P2P setup failed"), communicator],
            ) as factory,
        ):
            with self.assertRaisesRegex(RuntimeError, "P2P setup failed"):
                all_reduce_module.init_all_reduce(
                    self.group, self.device, topology(8), policy
                )
            self.assertIsNone(all_reduce_module._group)
            self.assertIsNone(all_reduce_module._options)
            self.assertIsNone(all_reduce_module.get_canonical_p2p())
            all_reduce_module.init_all_reduce(
                self.group, self.device, topology(8), policy
            )
            self.assertIs(all_reduce_module.get_canonical_p2p(), communicator)
        self.assertEqual(factory.call_count, 2)

    def test_agreed_fallback_remains_idempotent(self):
        with (
            patch.object(all_reduce_module, "_capability_error", return_value=None),
            patch.object(
                all_reduce_module,
                "IpcAllReduce",
                side_effect=IpcInitializationError("allocation failed"),
            ) as factory,
        ):
            self.assertIsNone(self.init(self.options))
            self.assertIsNone(self.init(self.options))
        factory.assert_called_once()
        self.assertIs(all_reduce_module._group, self.group)
        self.assertEqual(all_reduce_module._options, self.options)

    def test_initialization_is_idempotent_and_rebuilds_after_destroy(self):
        with (
            patch.object(all_reduce_module, "_capability_error", return_value=None),
            patch.object(all_reduce_module, "IpcAllReduce") as factory,
        ):
            first = self.init(self.options)
            self.assertIs(self.init(self.options), first)
            factory.assert_called_once()
            all_reduce_module.destroy_all_reduce()
            first.destroy.assert_called_once()
            self.init(self.options)
            self.assertEqual(factory.call_count, 2)

    def test_policy_change_requires_teardown(self):
        self.init(CollectiveConfig("nccl"))
        with self.assertRaisesRegex(RuntimeError, "before changing"):
            self.init(self.options)

    def test_public_initializer_forwards_one_policy(self):
        module = MagicMock()
        module.get_canonical_p2p.return_value = None
        pc = topology()
        with (
            patch.object(collective, "_get_all_reduce_module", return_value=module),
            patch.object(collective, "_get_group", return_value=self.group),
        ):
            collective._init_all_reduce(pc, self.options)
        module.init_all_reduce.assert_called_once_with(
            self.group, self.device, pc, self.options
        )

    def make_ipc(self, large=False, prepare=False):
        def workspace(group, device, max_bytes):
            result = FlashInferIpcAllReduce.__new__(FlashInferIpcAllReduce)
            result.max_bytes = max_bytes
            result.registered_graph_inputs = True
            result.ops = SimpleNamespace(meta_size=lambda: 4096)
            return result

        with (
            patch(
                "rtp_llm.models_py.distributed.flashinfer_ipc_all_reduce.FlashInferIpcAllReduce",
                side_effect=workspace,
            ) as factory,
            patch.object(
                torch.cuda, "mem_get_info", return_value=(10 * 1024**3, 72 * 1024**3)
            ),
            patch.object(torch.cuda, "is_current_stream_capturing", return_value=False),
        ):
            context = all_reduce_module.IpcAllReduce(
                self.group, self.device, CollectiveConfig("sm120_ipc", large)
            )
            if prepare:
                context.prepare_large_prefill([model_config()], engine_config())
        self.assertEqual(factory.call_args_list[0].kwargs["max_bytes"], 32 * 2560 * 2)
        if large:
            self.assertIsNot(context._workspace, context._prefill_workspace)
        return context

    def test_shape_capacity_capture_and_large_prefill_domain(self):
        tensor = SimpleNamespace(
            is_cuda=True,
            device=self.device,
            dtype=torch.bfloat16,
            ndim=2,
            shape=(4, 2560),
            is_contiguous=lambda: True,
            data_ptr=lambda: 16,
        )
        for large in (False, True):
            context = self.make_ipc(large, prepare=large)
            with patch("torch.cuda.is_current_stream_capturing", return_value=False):
                for rows in (1, 4, 16, 32):
                    tensor.shape = (rows, 2560)
                    self.assertTrue(context.should_use_cpp_small(tensor))
                for rows in (4093, 4094, 4095, 4096, 8192, 12288, 16384):
                    tensor.shape = (rows, 2560)
                    self.assertEqual(context.should_use(tensor), large)
                for shape in ((0, 2560), (33, 2560), (4, 5120)):
                    tensor.shape = shape
                    self.assertFalse(context.should_use(tensor))
            tensor.shape = (4, 2560)
            with patch("torch.cuda.is_current_stream_capturing", return_value=True):
                self.assertFalse(context.should_use(tensor))
                context.allow_ipc_capture = True
                self.assertTrue(context.should_use_registered_graph_input(tensor))
                tensor.shape = (4096, 2560)
                self.assertFalse(context.should_use(tensor))

    def test_prefill_budget_failure_preserves_decode_workspace(self):
        context = self.make_ipc()
        context._prefill_policy = None
        small = context._workspace
        with (
            patch.object(
                torch.cuda, "mem_get_info", return_value=(1024**2, 72 * 1024**3)
            ),
            patch.object(torch.cuda, "is_current_stream_capturing", return_value=False),
            patch(
                "rtp_llm.models_py.distributed.flashinfer_ipc_all_reduce.FlashInferIpcAllReduce"
            ) as factory,
        ):
            context.prepare_large_prefill([model_config()], engine_config())
        factory.assert_not_called()
        self.assertFalse(context.large_prefill)
        self.assertIs(context._workspace, small)

    def test_constructor_and_dispatch_never_prepare_large_workspace(self):
        for policy in (None, True):
            context = self.make_ipc(policy)
            self.assertFalse(context.large_prefill)
            tensor = SimpleNamespace(
                is_cuda=True,
                device=self.device,
                dtype=torch.bfloat16,
                ndim=2,
                shape=(4096, 2560),
                is_contiguous=lambda: True,
            )
            with (
                patch.object(
                    torch.cuda, "is_current_stream_capturing", return_value=False
                ),
                patch.object(torch.cuda, "mem_get_info") as probe,
                patch.object(context, "prepare_large_prefill") as prepare,
            ):
                self.assertFalse(context.should_use(tensor))
            probe.assert_not_called()
            prepare.assert_not_called()

    def test_startup_capacity_uses_model_and_scheduler_without_warmup(self):
        for tokens, hidden, dtype, rows in (
            (32768, 2560, torch.bfloat16, 16384),
            (8192, 2560, torch.bfloat16, 8192),
            (4093, 2560, torch.bfloat16, 4093),
            (2048, 2560, torch.bfloat16, 0),
            (32768, 5120, torch.bfloat16, 0),
            (32768, 2560, torch.float16, 0),
        ):
            with self.subTest(tokens=tokens, hidden=hidden, dtype=dtype):
                context = self.make_ipc(None)
                small = context._workspace
                with (
                    patch.object(
                        torch.cuda,
                        "mem_get_info",
                        return_value=(10 * 1024**3, 72 * 1024**3),
                    ),
                    patch(
                        "rtp_llm.models_py.distributed.flashinfer_ipc_all_reduce.FlashInferIpcAllReduce"
                    ) as factory,
                ):
                    context.prepare_large_prefill(
                        [model_config(hidden, dtype)], engine_config(tokens)
                    )
                self.assertIs(context._workspace, small)
                self.assertEqual(context.large_prefill, rows > 0)
                if rows:
                    self.assertEqual(
                        factory.call_args.kwargs["max_bytes"], rows * 2560 * 2
                    )
                else:
                    factory.assert_not_called()

    def test_explicit_kv_budget_and_runtime_floor_are_preserved(self):
        for free_gib, total_gib, runtime_mb, kv_mb, tokens in (
            (10, 72, 4096, 6144, 32768),
            (3, 72, 0, 0, 8192),
            (1, 16, 0, 0, 8192),
        ):
            with self.subTest(free=free_gib, total=total_gib, kv=kv_mb):
                context = self.make_ipc(None)
                with (
                    patch.object(
                        torch.cuda,
                        "mem_get_info",
                        return_value=(free_gib * 1024**3, total_gib * 1024**3),
                    ),
                    patch(
                        "rtp_llm.models_py.distributed.flashinfer_ipc_all_reduce.FlashInferIpcAllReduce"
                    ) as factory,
                ):
                    context.prepare_large_prefill(
                        [model_config()], engine_config(tokens, runtime_mb, kv_mb)
                    )
                factory.assert_not_called()
                self.assertFalse(context.large_prefill)

    def test_startup_accepts_supported_draft_and_is_idempotent(self):
        context = self.make_ipc(None)
        configs = [model_config(5120), model_config()]
        config = engine_config(kv_mb=5120, runtime_mb=4096)
        with (
            patch.object(all_reduce_module, "_communicator", context),
            patch.object(
                torch.cuda, "mem_get_info", return_value=(10 * 1024**3, 72 * 1024**3)
            ) as probe,
            patch(
                "rtp_llm.models_py.distributed.flashinfer_ipc_all_reduce.FlashInferIpcAllReduce"
            ) as factory,
        ):
            all_reduce_module.prepare_collective_workspaces(configs, config)
            all_reduce_module.prepare_collective_workspaces(configs, config)
        probe.assert_called_once()
        factory.assert_called_once()
        self.assertTrue(context.large_prefill)

    def test_startup_rank_capacity_disagreement_keeps_nccl(self):
        context = self.make_ipc(None)
        with (
            patch.object(
                torch.cuda, "mem_get_info", return_value=(10 * 1024**3, 72 * 1024**3)
            ),
            patch.object(
                all_reduce_module,
                "exchange_group_state",
                return_value=[[8192 * 2560 * 2, True], [16384 * 2560 * 2, True]],
            ),
            patch(
                "rtp_llm.models_py.distributed.flashinfer_ipc_all_reduce.FlashInferIpcAllReduce"
            ) as factory,
        ):
            context.prepare_large_prefill([model_config()], engine_config())
        factory.assert_not_called()
        self.assertFalse(context.large_prefill)

    def test_engine_prepares_target_and_draft_before_cpp_engine(self):
        from rtp_llm.async_decoder_engine.engine_creator import create_engine
        from rtp_llm.ops import TaskType

        target_config = model_config()
        target_config.task_type = TaskType.LANGUAGE_MODEL
        target = SimpleNamespace(model_config=target_config)
        draft_config = model_config()
        draft = SimpleNamespace(model=SimpleNamespace(model_config=draft_config))
        config = engine_config()
        events = []
        constructor = MagicMock(side_effect=lambda **kwargs: events.append("construct"))
        with (
            patch("rtp_llm.ops.ensure_engine_ops_loaded"),
            patch.object(
                torch.ops.rtp_llm,
                "init_engine",
                create=True,
                side_effect=lambda _: events.append("init"),
            ),
            patch.object(
                all_reduce_module,
                "prepare_collective_workspaces",
                side_effect=lambda *args: events.append("prepare"),
            ) as prepare,
            patch.dict(
                "sys.modules",
                {
                    "rtp_llm.async_decoder_engine.rpc_engine": SimpleNamespace(
                        LanguageCppEngine=constructor
                    )
                },
            ),
        ):
            create_engine(target, config, "", propose_model=draft)
        prepare.assert_called_once_with([target_config, draft_config], config)
        self.assertEqual(events, ["init", "prepare", "construct"])

    def test_peer_budget_rejection_does_not_allocate(self):
        context = self.make_ipc()
        context._prefill_policy = None
        with (
            patch.object(
                torch.cuda, "mem_get_info", return_value=(10 * 1024**3, 72 * 1024**3)
            ),
            patch.object(torch.cuda, "is_current_stream_capturing", return_value=False),
            patch.object(
                all_reduce_module,
                "exchange_group_state",
                return_value=[[16384 * 2560 * 2, True], [16384 * 2560 * 2, False]],
            ),
            patch(
                "rtp_llm.models_py.distributed.flashinfer_ipc_all_reduce.FlashInferIpcAllReduce"
            ) as factory,
        ):
            context.prepare_large_prefill([model_config()], engine_config())
        factory.assert_not_called()

    def test_capture_registers_and_restores_admission(self):
        communicator = MagicMock()
        communicator.allow_ipc_capture = False
        with patch.object(all_reduce_module, "_communicator", communicator):
            all_reduce_module.begin_graph_capture()
            self.assertTrue(communicator.allow_ipc_capture)
            all_reduce_module.end_graph_capture()
            self.assertFalse(communicator.allow_ipc_capture)
        communicator.finish_graph_capture.assert_called_once()

    def test_capture_failure_restores_without_registration(self):
        communicator = MagicMock()
        with patch.object(all_reduce_module, "_communicator", communicator):
            all_reduce_module.begin_graph_capture()
            all_reduce_module.abort_graph_capture()
            self.assertFalse(communicator.allow_ipc_capture)
        communicator.finish_graph_capture.assert_not_called()

    def test_registration_failure_propagates_after_restoring_admission(self):
        communicator = MagicMock()
        communicator.finish_graph_capture.side_effect = RuntimeError(
            "registration failed"
        )
        with patch.object(all_reduce_module, "_communicator", communicator):
            all_reduce_module.begin_graph_capture()
            with self.assertRaisesRegex(RuntimeError, "registration failed"):
                all_reduce_module.end_graph_capture()
            self.assertFalse(communicator.allow_ipc_capture)
            self.assertFalse(all_reduce_module._capture_active)

    def test_unaligned_views_keep_backend_and_output_contract(self):
        for capturing in (False, True):
            for inplace in (False, True):
                with self.subTest(capturing=capturing, inplace=inplace):
                    source = torch.arange(4 * 2560 + 1, dtype=torch.bfloat16)[1:].view(
                        4, 2560
                    )
                    original = source.clone()
                    target = source if inplace else torch.empty_like(source)
                    workspace = FlashInferIpcAllReduce.__new__(FlashInferIpcAllReduce)
                    workspace.handle = 73
                    workspace.rank = 0
                    workspace.buffer = [912]
                    workspace.max_bytes = 32 * 2560 * 2
                    workspace.registered_graph_inputs = True
                    workspace.capture_calls = 0
                    workspace.ops = MagicMock()

                    def kernel(handle, value, output, staging, capacity, sms):
                        self.assertEqual(value.data_ptr() % 16, 0)
                        self.assertEqual(output.data_ptr() % 16, 0)
                        self.assertEqual(staging, 912)
                        output.copy_(value * 2)

                    workspace.ops.all_reduce.side_effect = kernel
                    with patch.object(
                        torch.cuda,
                        "is_current_stream_capturing",
                        return_value=capturing,
                    ):
                        self.assertIs(workspace.all_reduce(source, target), target)
                    torch.testing.assert_close(target, original * 2)
                    if not inplace:
                        torch.testing.assert_close(source, original)

    def test_collective_inplace_and_out_of_place_contract(self):
        tensor = torch.ones((2, 4))
        fast_path = MagicMock()
        fast_path.should_use.return_value = True
        fast_path.all_reduce_into.side_effect = lambda source, target: target.fill_(2)
        fast_path.all_reduce.return_value = torch.full_like(tensor, 3)
        module = SimpleNamespace(get_all_reduce=lambda: fast_path)
        with (
            patch.object(collective, "_get_all_reduce_module", return_value=module),
            patch.object(collective, "_canonical_p2p", None),
        ):
            self.assertIs(collective.all_reduce(tensor, collective.Group.TP), tensor)
            original = tensor.clone()
            result = collective.all_reduce(tensor, collective.Group.TP, inplace=False)
        torch.testing.assert_close(tensor, original)
        torch.testing.assert_close(result, torch.full_like(tensor, 3))


class CollectiveAgreementTest(unittest.TestCase):
    def test_store_rounds_do_not_reuse_stale_decisions(self):
        from rtp_llm.models_py.distributed import collective_utils

        store = torch.distributed.HashStore()
        group = SimpleNamespace(group_name="agreement_unit_test")
        with (
            patch.object(
                collective_utils.dist.distributed_c10d,
                "_get_default_store",
                return_value=store,
            ),
            patch.object(collective_utils.dist, "get_world_size", return_value=1),
            patch.object(collective_utils.dist, "get_rank", return_value=0),
        ):
            self.assertEqual(
                collective_utils.exchange_group_state(group, False, "probe"), [False]
            )
            self.assertEqual(
                collective_utils.exchange_group_state(group, True, "probe"), [True]
            )

    def test_missing_rank_timeout_propagates_without_gpu_collective(self):
        from rtp_llm.models_py.distributed import collective_utils

        store = MagicMock()
        store.wait.side_effect = RuntimeError("missing rank")
        with (
            patch.object(
                collective_utils.dist.distributed_c10d,
                "_get_default_store",
                return_value=store,
            ),
            patch.object(collective_utils.dist, "get_world_size", return_value=2),
            patch.object(collective_utils.dist, "get_rank", return_value=0),
            patch.object(collective_utils.dist, "all_reduce") as gpu_collective,
            self.assertRaisesRegex(RuntimeError, "missing rank"),
        ):
            collective_utils.exchange_group_state(
                SimpleNamespace(group_name="missing"), True, "probe"
            )
        self.assertLessEqual(store.wait.call_args.args[1].total_seconds(), 120)
        gpu_collective.assert_not_called()


if __name__ == "__main__":
    unittest.main()

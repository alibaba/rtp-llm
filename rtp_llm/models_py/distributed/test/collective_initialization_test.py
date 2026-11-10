"""Exercise policy propagation through every distributed initialization entry."""

import os
import sys
import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch.distributed as dist

from rtp_llm.config.collective_config import CollectiveConfig
from rtp_llm.config.server_config_setup import set_parallelism_config
from rtp_llm.models_py.distributed import all_reduce
from rtp_llm.models_py.distributed import collective_torch as collective
from rtp_llm.models_py.utils import arch
from rtp_llm.ops import ParallelismConfig


class CollectiveInitializationTest(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.group = object()
        self.pc = ParallelismConfig()
        self.pc.tp_size = self.pc.ep_size = 2
        self.pc.world_size = self.pc.local_world_size = 2
        set_parallelism_config(self.pc)
        for module, values in (
            (
                collective,
                dict(
                    _initialized=False,
                    _parallelism_config=None,
                    _group_map={},
                    _canonical_p2p=None,
                    _cpu_tp_broadcaster_base_path=None,
                ),
            ),
            (
                all_reduce,
                dict(
                    _group=None,
                    _options=None,
                    _communicator=None,
                    _canonical_p2p=None,
                    _capture_active=False,
                ),
            ),
        ):
            for name, value in values.items():
                self.stack.enter_context(patch.object(module, name, value))
        self.stack.enter_context(patch.dict(os.environ))
        self.stack.enter_context(
            patch.dict(sys.modules, {"librtp_compute_ops": MagicMock()})
        )
        self.initialized = self.stack.enter_context(
            patch.object(dist, "is_initialized", return_value=False)
        )
        self.stack.enter_context(
            patch.object(dist, "init_process_group", side_effect=self.mark_initialized)
        )
        self.stack.enter_context(
            patch.object(dist, "destroy_process_group", side_effect=self.mark_destroyed)
        )
        self.stack.enter_context(patch.object(dist, "barrier"))
        self.stack.enter_context(patch.object(dist, "get_rank", return_value=0))
        self.stack.enter_context(
            patch.object(collective, "_get_rocm_rccl", return_value=None)
        )
        self.stack.enter_context(
            patch.object(
                collective, "_create_process_groups", side_effect=self.create_groups
            )
        )
        self.stack.enter_context(
            patch.object(collective, "_register_process_groups_to_cpp")
        )
        self.stack.enter_context(
            patch.object(collective, "init_user_buffers_environment")
        )
        self.stack.enter_context(
            patch.object(
                collective,
                "_make_cpu_tp_broadcaster_base_path",
                return_value="/tmp/unused",
            )
        )
        self.stack.enter_context(patch.object(arch, "is_cuda", return_value=False))
        self.stack.enter_context(
            patch.object(
                all_reduce,
                "exchange_group_state",
                side_effect=lambda group, state, tag: [state, state],
            )
        )
        self.probe = self.stack.enter_context(
            patch.object(all_reduce, "_capability_error", return_value=None)
        )
        self.factory = self.stack.enter_context(
            patch.object(all_reduce, "IpcAllReduce")
        )
        self.addCleanup(all_reduce.destroy_all_reduce)

    def mark_initialized(self, **_kwargs):
        self.initialized.return_value = True

    def mark_destroyed(self):
        self.initialized.return_value = False

    def create_groups(self, *_args):
        collective._group_map[collective.Group.DP_AND_TP] = self.group

    def initialize(self, **kwargs):
        collective.init_distributed_environment(
            self.pc, SimpleNamespace(nccl_ip="127.0.0.1"), 29501, **kwargs
        )

    def test_every_initialization_branch_receives_explicit_policy(self):
        policy = CollectiveConfig("nccl", False)
        for owned, external, has_groups in (
            (False, False, False),
            (False, True, False),
            (True, True, True),
            (True, True, False),
        ):
            with self.subTest(owned=owned, external=external, has_groups=has_groups):
                all_reduce.destroy_all_reduce()
                collective._initialized = owned
                collective._parallelism_config = self.pc if owned else None
                collective._group_map.clear()
                if has_groups:
                    self.create_groups()
                self.initialized.return_value = external
                self.initialize(collective_config=policy)
                self.assertIs(all_reduce._options, policy)
        self.probe.assert_not_called()
        self.factory.assert_not_called()

    def test_same_policy_is_idempotent(self):
        policy = CollectiveConfig("sm120_ipc", False)
        self.initialize(collective_config=policy)
        communicator = all_reduce.get_all_reduce()
        self.initialize(collective_config=CollectiveConfig("sm120_ipc", False))
        self.assertIs(all_reduce.get_all_reduce(), communicator)
        self.factory.assert_called_once()

    def test_changing_live_policy_requires_destroy(self):
        self.initialize(collective_config=CollectiveConfig("nccl", False))
        with self.assertRaisesRegex(RuntimeError, "before changing policy"):
            self.initialize(collective_config=CollectiveConfig("sm120_ipc"))
        self.probe.assert_not_called()

    def test_omitted_policy_does_not_inherit_live_override(self):
        self.initialize(collective_config=CollectiveConfig("nccl", False))
        with self.assertRaisesRegex(RuntimeError, "before changing policy"):
            self.initialize()

    def test_destroy_then_omit_restores_default_without_reading_environment(self):
        self.initialize(collective_config=CollectiveConfig("nccl", False))
        collective.destroy_distributed_environment()
        self.assertIsNone(all_reduce._options)
        self.assertFalse(collective._initialized)
        with patch.dict(os.environ, {"RTP_LLM_ALL_REDUCE_BACKEND": "nccl"}):
            self.initialize()
        self.assertEqual(all_reduce._options, CollectiveConfig())
        self.assertEqual(self.factory.call_args.args[2].all_reduce_backend, "sm120_ipc")

    def test_destroy_allows_new_explicit_policy(self):
        self.initialize(collective_config=CollectiveConfig("sm120_ipc", False))
        communicator = all_reduce.get_all_reduce()
        collective.destroy_distributed_environment()
        communicator.destroy.assert_called_once()
        self.initialize(collective_config=CollectiveConfig("nccl", False))
        self.assertIsNone(all_reduce.get_all_reduce())
        self.assertEqual(all_reduce._options, CollectiveConfig("nccl", False))

    def test_single_rank_override_is_not_silently_ignored(self):
        self.pc.tp_size = self.pc.world_size = self.pc.local_world_size = 1
        with self.assertRaisesRegex(ValueError, "requires single-node TP2"):
            collective._init_all_reduce(self.pc, CollectiveConfig("sm120_ipc"))
        self.probe.assert_not_called()
        self.factory.assert_not_called()

    def test_group_lookup_requires_complete_explicit_initialization(self):
        collective._parallelism_config = self.pc
        with (
            patch.object(collective, "init_distributed_environment") as initialize,
            self.assertRaisesRegex(RuntimeError, "complete configuration"),
        ):
            collective._get_group(collective.Group.TP)
        initialize.assert_not_called()


if __name__ == "__main__":
    unittest.main()

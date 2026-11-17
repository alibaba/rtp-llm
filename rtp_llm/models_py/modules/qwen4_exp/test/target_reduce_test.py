import unittest
from types import SimpleNamespace
from unittest.mock import patch

from rtp_llm.models_py.distributed.flashinfer_ipc_all_reduce import (
    IpcInitializationError,
)
from rtp_llm.models_py.modules.qwen4_exp import tp_reduce


class TargetReduceAdmissionTest(unittest.TestCase):
    def setUp(self):
        self.config = SimpleNamespace(is_mtp=False, hidden_size=2560)
        self.parallelism = SimpleNamespace(
            role_type="DECODE", tp_size=2, dp_size=1, local_world_size=2
        )
        self.communicator = SimpleNamespace(
            disabled=False,
            cpp_small_enabled=True,
            group=object(),
            device=object(),
            _workspace=SimpleNamespace(ops=SimpleNamespace(meta_size=lambda: 4096)),
        )

    def prepare(self, *, free=64 * 1024 * 1024, peer_error=None, setup_error=None):
        def vote(group, error, phase):
            self.assertIs(group, self.communicator.group)
            self.assertEqual(phase, "qwen4_target_memory")
            return [error, peer_error]

        with (
            patch.object(tp_reduce, "get_all_reduce", return_value=self.communicator),
            patch.object(
                tp_reduce.torch.cuda, "mem_get_info", return_value=(free, free)
            ),
            patch.object(
                tp_reduce, "exchange_group_state", side_effect=vote
            ) as exchange,
            patch.object(
                tp_reduce, "Qwen4TargetReduce", side_effect=setup_error
            ) as setup,
        ):
            result = tp_reduce.prepare_target_reduce(self.config, self.parallelism)
        return result, exchange, setup

    def test_automatically_extends_admitted_ipc_backend(self):
        result, exchange, setup = self.prepare()
        exchange.assert_called_once()
        setup.assert_called_once_with(self.communicator, max_bytes=256 * 2560 * 2)
        self.assertIs(result, setup.return_value)

    def test_other_roles_shapes_and_backends_allocate_nothing(self):
        for owner, name, value in (
            (self.config, "is_mtp", True),
            (self.config, "hidden_size", 5120),
            (self.parallelism, "role_type", "PREFILL"),
            (self.parallelism, "tp_size", 8),
            (self.parallelism, "dp_size", 2),
            (self.parallelism, "local_world_size", 1),
            (self.communicator, "disabled", True),
            (self.communicator, "cpp_small_enabled", False),
        ):
            with self.subTest(name=name, value=value), patch.object(owner, name, value):
                result, exchange, setup = self.prepare()
                self.assertIsNone(result)
                exchange.assert_not_called()
                setup.assert_not_called()

    def test_missing_communicator_keeps_original_reducer(self):
        self.communicator = None
        with (
            patch.object(tp_reduce, "get_all_reduce", return_value=None),
            patch.object(tp_reduce, "exchange_group_state") as exchange,
            patch.object(tp_reduce, "Qwen4TargetReduce") as setup,
        ):
            self.assertIsNone(
                tp_reduce.prepare_target_reduce(self.config, self.parallelism)
            )
        exchange.assert_not_called()
        setup.assert_not_called()

    def test_either_rank_without_memory_keeps_original_reducer(self):
        for kwargs in ({"free": 0}, {"peer_error": "peer memory unavailable"}):
            with self.subTest(kwargs=kwargs):
                result, exchange, setup = self.prepare(**kwargs)
                self.assertIsNone(result)
                exchange.assert_called_once()
                setup.assert_not_called()

    def test_budget_includes_raw_ipc_allocations_and_rank_data(self):
        required = 8 * 1024 * 1024 + 2 * 256 * 2560 * 2 + 4096
        result, _, setup = self.prepare(free=required - 1)
        self.assertIsNone(result)
        setup.assert_not_called()
        result, _, setup = self.prepare(free=required)
        self.assertIs(result, setup.return_value)

    def test_agreed_setup_failure_keeps_original_reducer(self):
        result, _, setup = self.prepare(
            setup_error=IpcInitializationError("peer failed")
        )
        setup.assert_called_once()
        self.assertIsNone(result)

    def test_unexpected_setup_error_is_not_hidden(self):
        with self.assertRaisesRegex(RuntimeError, "unexpected"):
            self.prepare(setup_error=RuntimeError("unexpected"))


if __name__ == "__main__":
    unittest.main()

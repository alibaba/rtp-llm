import os
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from rtp_llm.models_py.distributed import collective_torch as collective
from rtp_llm.models_py.distributed import symm_mem


class SymmMemConfigurationTest(unittest.TestCase):
    def test_long_socket_path_is_shortened_during_rendezvous_and_restored(self):
        long_dir = "/tmp/" + "x" * 180
        buffer = object()

        def rendezvous(value, group):
            self.assertIs(value, buffer)
            self.assertEqual(group, "test-group")
            self.assertEqual(os.environ["TMPDIR"], "/tmp")
            return "handle"

        with patch.dict(os.environ, {"TMPDIR": long_dir}, clear=True), patch.object(
            symm_mem.torch_symm_mem, "rendezvous", side_effect=rendezvous
        ):
            self.assertEqual(
                symm_mem._rendezvous_with_short_socket_path(buffer, "test-group"),
                "handle",
            )
            self.assertEqual(os.environ["TMPDIR"], long_dir)

    def test_rendezvous_error_restores_absent_tmpdir(self):
        with patch.dict(
            os.environ, {"TEMP": "/tmp/" + "x" * 180}, clear=True
        ), patch.object(
            symm_mem.torch_symm_mem, "rendezvous", side_effect=RuntimeError("failure")
        ):
            with self.assertRaisesRegex(RuntimeError, "failure"):
                symm_mem._rendezvous_with_short_socket_path(object(), "test-group")
            self.assertNotIn("TMPDIR", os.environ)

    def tearDown(self) -> None:
        symm_mem._symm_mem_comm = None

    def test_custom_ar_is_pinned_disabled_communicator_never_constructed(self) -> None:
        sentinel = object()
        symm_mem._symm_mem_comm = sentinel
        with patch.object(
            symm_mem,
            "TorchSymmMemCommunicator",
            side_effect=AssertionError("communicator must not be constructed"),
        ):
            self.assertIsNone(symm_mem.init_symm_mem_communicator(object()))
            self.assertIsNone(symm_mem.get_symm_mem_communicator())

    def test_collective_init_calls_symm_mem_without_custom_ar_config(self) -> None:
        world_group = object()
        communicator_module = MagicMock()
        config = SimpleNamespace(
            world_rank=0, world_size=2, tp_size=2, dp_size=1, pp_size=1
        )

        group_namespace = SimpleNamespace(WORLD=world_group)
        with patch.object(
            collective.torch.distributed, "group", group_namespace
        ), patch.object(collective.torch.distributed, "new_group"), patch.object(
            collective.torch.distributed, "barrier"
        ), patch.dict(
            collective._group_map, {}, clear=True
        ), patch.object(
            collective, "_get_symm_mem", return_value=communicator_module
        ):
            collective._create_process_groups(
                config,
                "nccl",
                None,
            )

        communicator_module.init_symm_mem_communicator.assert_called_once_with(
            world_group,
        )

    def test_unlisted_architecture_votes_to_fall_back_before_allocation(self) -> None:
        group = object()
        for capability in (8, 12, 99):
            with self.subTest(capability=capability), patch.object(
                symm_mem, "torch_symm_mem_available", True
            ), patch.object(symm_mem.torch.cuda, "set_device"), patch.object(
                symm_mem.torch.cuda,
                "get_device_capability",
                return_value=(capability, 0),
            ), patch.object(
                symm_mem.dist, "get_rank", return_value=0
            ), patch.object(
                symm_mem.dist, "get_world_size", return_value=2
            ), patch.object(
                symm_mem, "_agree_across_group", return_value=False
            ) as agree, patch.object(
                symm_mem.torch_symm_mem, "empty"
            ) as allocate, patch.object(
                symm_mem.torch_symm_mem, "rendezvous"
            ) as rendezvous:
                comm = symm_mem.TorchSymmMemCommunicator(group, 0)
                self.assertTrue(comm.disabled)
                self.assertIsNone(comm.buffer)
                agree.assert_called_once_with(group, False, "alloc")
                allocate.assert_not_called()
                rendezvous.assert_not_called()


if __name__ == "__main__":
    unittest.main()

import os
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from rtp_llm.models_py.distributed import collective_torch as collective
from rtp_llm.models_py.distributed import symm_mem


class SymmMemConfigurationTest(unittest.TestCase):
    def tearDown(self) -> None:
        symm_mem._symm_mem_comm = None

    def test_custom_all_reduce_switch_accepts_standard_true_values(self) -> None:
        for value in ("1", "true", "TRUE", "on", "yes"):
            with self.subTest(value=value), patch.dict(
                os.environ, {"FT_DISABLE_CUSTOM_AR": value}, clear=False
            ):
                self.assertFalse(symm_mem._custom_all_reduce_enabled())

    def test_custom_all_reduce_is_enabled_when_switch_is_absent_or_false(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            self.assertTrue(symm_mem._custom_all_reduce_enabled())
        for value in ("0", "false", "off", "no"):
            with self.subTest(value=value), patch.dict(
                os.environ, {"FT_DISABLE_CUSTOM_AR": value}, clear=False
            ):
                self.assertTrue(symm_mem._custom_all_reduce_enabled())

    def test_explicit_config_overrides_environment(self) -> None:
        with patch.dict(os.environ, {"FT_DISABLE_CUSTOM_AR": "0"}, clear=True):
            self.assertFalse(symm_mem._custom_all_reduce_enabled(True))
        with patch.dict(os.environ, {"FT_DISABLE_CUSTOM_AR": "1"}, clear=True):
            self.assertTrue(symm_mem._custom_all_reduce_enabled(False))

    def test_disabled_switch_skips_communicator_construction(self) -> None:
        sentinel = object()
        symm_mem._symm_mem_comm = sentinel
        with patch.dict(
            os.environ, {"FT_DISABLE_CUSTOM_AR": "1"}, clear=False
        ), patch.object(
            symm_mem,
            "TorchSymmMemCommunicator",
            side_effect=AssertionError("communicator must not be constructed"),
        ):
            self.assertIsNone(symm_mem.init_symm_mem_communicator(object()))
            self.assertIsNone(symm_mem.get_symm_mem_communicator())

    def test_explicit_disabled_config_skips_communicator_construction(self) -> None:
        with patch.dict(os.environ, {}, clear=True), patch.object(
            symm_mem,
            "TorchSymmMemCommunicator",
            side_effect=AssertionError("communicator must not be constructed"),
        ):
            self.assertIsNone(
                symm_mem.init_symm_mem_communicator(
                    object(), disable_custom_all_reduce=True
                )
            )

    def test_collective_forwards_explicit_config(self) -> None:
        world_group = object()
        communicator_module = MagicMock()
        config = SimpleNamespace(world_rank=0, world_size=2, tp_size=2, dp_size=1)

        group_namespace = SimpleNamespace(WORLD=world_group)
        with patch.object(
            collective.torch.distributed, "group", group_namespace
        ), patch.object(collective, "_get_symm_mem", return_value=communicator_module):
            collective._create_process_groups(
                config,
                "nccl",
                None,
                disable_custom_all_reduce=True,
            )

        communicator_module.init_symm_mem_communicator.assert_called_once_with(
            world_group,
            disable_custom_all_reduce=True,
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

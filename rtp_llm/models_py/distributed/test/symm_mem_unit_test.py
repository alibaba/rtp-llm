import os
import socket
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from rtp_llm.models_py.distributed import collective_torch as collective
from rtp_llm.models_py.distributed import symm_mem


class SymmMemConfigurationTest(unittest.TestCase):
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

    def test_rendezvous_uses_short_tmpdir_and_restores_environment(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as directory:
            long_directory = str(Path(directory) / ("sandbox-" * 20))
            for previous_tmpdir in (None, directory, long_directory):
                for fail_rendezvous in (False, True):
                    with self.subTest(
                        previous_tmpdir=previous_tmpdir,
                        fail_rendezvous=fail_rendezvous,
                    ):
                        environment = {"TMP": long_directory, "TEMP": long_directory}
                        if previous_tmpdir is not None:
                            environment["TMPDIR"] = previous_tmpdir

                        def rendezvous(
                            buffer: torch.Tensor, *, group: str
                        ) -> SimpleNamespace:
                            socket_directory = next(
                                os.environ[name]
                                for name in ("TMPDIR", "TMP", "TEMP", "TEMPDIR")
                                if name in os.environ
                            )
                            socket_path = (
                                Path(socket_directory) / f"symm_mem-{os.getpid()}"
                            )
                            try:
                                with socket.socket(
                                    socket.AF_UNIX, socket.SOCK_DGRAM
                                ) as channel:
                                    channel.bind(str(socket_path))
                            finally:
                                socket_path.unlink(missing_ok=True)
                            if fail_rendezvous:
                                raise RuntimeError()
                            return SimpleNamespace(multicast_ptr=1)

                        group = SimpleNamespace(group_name="unit-test")
                        with (
                            patch.dict(os.environ, environment, clear=True),
                            patch.object(symm_mem, "torch_symm_mem_available", True),
                            patch.object(torch.cuda, "set_device"),
                            patch.object(
                                torch.cuda, "get_device_capability", return_value=(9, 0)
                            ),
                            patch.object(symm_mem.dist, "get_rank", return_value=0),
                            patch.object(
                                symm_mem.dist, "get_world_size", return_value=2
                            ),
                            patch.object(
                                symm_mem, "_agree_across_group", return_value=True
                            ),
                            patch.object(
                                symm_mem.torch_symm_mem,
                                "empty",
                                return_value=torch.empty(1),
                            ),
                            patch.object(
                                symm_mem.torch_symm_mem,
                                "rendezvous",
                                side_effect=rendezvous,
                            ),
                            patch.object(
                                torch.ops,
                                "symm_mem",
                                SimpleNamespace(multimem_all_gather_out=MagicMock()),
                            ),
                        ):
                            if fail_rendezvous:
                                with self.assertRaises(RuntimeError):
                                    symm_mem.TorchSymmMemCommunicator(group, 0)
                            else:
                                comm = symm_mem.TorchSymmMemCommunicator(group, 0)
                                self.assertFalse(comm.disabled)
                            self.assertEqual(dict(os.environ), environment)


if __name__ == "__main__":
    unittest.main()

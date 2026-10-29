# SPDX-License-Identifier: Apache-2.0

import os
import socket
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from rtp_llm.models_py.distributed import collective_torch as collective
from rtp_llm.ops import NcclCommConfig, ParallelismConfig


class CpuTpBroadcasterUnitTest(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory(prefix="rtp-uds-", dir="/tmp")
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.long_directory = self.root / ("x" * 140)
        environment = patch.dict(
            os.environ,
            {
                "TMPDIR": str(self.long_directory),
                "RTP_LLM_CPU_TP_BROADCASTER_DIR": "",
                "RTP_LLM_CPU_TP_BROADCASTER_ID": self.root.name,
            },
        )
        environment.start()
        self.addCleanup(environment.stop)
        self.config = ParallelismConfig()
        self.config.tp_size = 2
        self.config.local_world_size = 2
        self.config.world_size = 2
        self.config.dp_size = 1
        self.config.dp_rank = 0
        self.config.world_rank = 0
        self.config.tp_rank = 0

    def test_long_tmpdir_uses_a_bindable_short_socket_path(self) -> None:
        base_path = collective._make_cpu_tp_broadcaster_base_path(self.config, 12345)

        self.assertIsNotNone(base_path)
        socket_path = Path(f"{base_path}_0.sock")
        self.addCleanup(socket_path.unlink, missing_ok=True)
        self.assertEqual(socket_path.parent, Path(f"/tmp/rtp_llm_{os.getuid()}"))
        self.assertLess(len(os.fsencode(socket_path)), 108)
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as listener:
            listener.bind(str(socket_path))
        self.assertFalse(self.long_directory.exists())

    def test_explicit_directory_is_preserved(self) -> None:
        directory = self.root / "uds"
        with patch.dict(os.environ, {"RTP_LLM_CPU_TP_BROADCASTER_DIR": str(directory)}):
            base_path = collective._make_cpu_tp_broadcaster_base_path(
                self.config, 12345
            )

        assert base_path is not None
        self.assertEqual(Path(base_path).parent, directory)
        self.assertTrue(directory.is_dir())

    def test_overlong_explicit_directory_is_rejected(self) -> None:
        with patch.dict(
            os.environ,
            {"RTP_LLM_CPU_TP_BROADCASTER_DIR": str(self.long_directory)},
        ):
            with self.assertRaises(ValueError):
                collective._make_cpu_tp_broadcaster_base_path(self.config, 12345)
        self.assertFalse(self.long_directory.exists())

    def test_tp_peers_share_the_same_path(self) -> None:
        with patch.dict(os.environ, {"RTP_LLM_CPU_TP_BROADCASTER_DIR": str(self.root)}):
            rank0_path = collective._make_cpu_tp_broadcaster_base_path(
                self.config, 12345
            )
            self.config.tp_rank = 1
            rank1_path = collective._make_cpu_tp_broadcaster_base_path(
                self.config, 12345
            )

        self.assertEqual(rank0_path, rank1_path)

    def test_sessions_and_dp_groups_have_distinct_paths(self) -> None:
        paths: set[str | None] = set()
        for session_id in (f"{self.root.name}-a", f"{self.root.name}-b"):
            for dp_rank in (0, 1):
                with patch.dict(
                    os.environ,
                    {
                        "RTP_LLM_CPU_TP_BROADCASTER_DIR": str(self.root),
                        "RTP_LLM_CPU_TP_BROADCASTER_ID": session_id,
                    },
                ):
                    self.config.dp_rank = dp_rank
                    paths.add(
                        collective._make_cpu_tp_broadcaster_base_path(
                            self.config, 12345
                        )
                    )

        self.assertNotIn(None, paths)
        self.assertEqual(len(paths), 4)

    def test_initialization_only_requires_paths_for_local_tp(self) -> None:
        nccl_config = NcclCommConfig()
        nccl_config.nccl_ip = "127.0.0.1"
        for tp_size in (1, 2, 4):
            for initialized, torch_initialized in (
                (False, False),
                (False, True),
                (True, True),
            ):
                with self.subTest(
                    tp_size=tp_size,
                    initialized=initialized,
                    torch_initialized=torch_initialized,
                ):
                    self.config.tp_size = tp_size
                    self.config.world_size = max(tp_size, 2)
                    self.config.dp_size = self.config.world_size // tp_size
                    process_group = MagicMock()
                    process_group.size.return_value = self.config.world_size
                    compute_ops = SimpleNamespace(
                        register_comm_ops=MagicMock(),
                        init_cpu_tp_broadcaster=MagicMock(),
                    )
                    directory = self.root if tp_size == 2 else self.long_directory
                    with (
                        patch.dict(
                            os.environ,
                            {"RTP_LLM_CPU_TP_BROADCASTER_DIR": str(directory)},
                        ),
                        patch.dict(sys.modules, {"librtp_compute_ops": compute_ops}),
                        patch.object(collective, "_initialized", initialized),
                        patch.object(collective, "_group_map", {}),
                        patch.object(collective, "_parallelism_config", self.config),
                        patch.object(collective, "_cpu_tp_broadcaster_base_path", None),
                        patch.object(collective, "_get_rocm_rccl", return_value=None),
                        patch.object(collective, "_create_process_groups"),
                        patch.object(collective, "init_user_buffers_environment"),
                        patch.object(collective, "_init_flashinfer_allreduce"),
                        patch.object(
                            torch.distributed,
                            "is_initialized",
                            return_value=torch_initialized,
                        ),
                        patch.object(torch.distributed, "init_process_group"),
                        patch.object(torch.distributed, "barrier"),
                        patch.object(
                            torch.distributed,
                            "group",
                            SimpleNamespace(WORLD=process_group),
                        ),
                    ):
                        collective.init_distributed_environment(
                            self.config, nccl_config, 12345
                        )

                        compute_ops.register_comm_ops.assert_called_once()
                        if tp_size == 2:
                            base_path = collective._cpu_tp_broadcaster_base_path
                            self.assertIsNotNone(base_path)
                            compute_ops.init_cpu_tp_broadcaster.assert_called_once_with(
                                0, 2, base_path
                            )
                        else:
                            self.assertIsNone(collective._cpu_tp_broadcaster_base_path)
                            compute_ops.init_cpu_tp_broadcaster.assert_not_called()
        self.assertFalse(self.long_directory.exists())


if __name__ == "__main__":
    unittest.main()

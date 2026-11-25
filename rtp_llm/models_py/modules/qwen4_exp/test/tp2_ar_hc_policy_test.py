import os
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from rtp_llm.models_py.distributed import collective_torch
from rtp_llm.models_py.modules.qwen4_exp import tp2_ar_hc_fused
from rtp_llm.models_py.modules.qwen4_exp.tp2_ar_hc_fused import (
    DeferredTp2Sublayer,
    prepare_tp2_ar_hc_workspace,
)


class Tp2ArHcPolicyTest(unittest.TestCase):
    def setUp(self):
        self.config = SimpleNamespace(hc_mult=4, hidden_size=2560)
        self.parallelism = SimpleNamespace(tp_size=2, dp_size=1, local_world_size=2)

    def test_default_preserves_standalone_allreduce(self):
        for env in ({}, {"FT_DISABLE_CUSTOM_AR": "0"}):
            with self.subTest(env=env), patch.dict(os.environ, env, clear=True):
                with patch("torch.cuda.is_available") as available:
                    self.assertIsNone(
                        prepare_tp2_ar_hc_workspace(self.config, self.parallelism)
                    )
                    available.assert_not_called()

    def test_explicit_opt_in_reaches_device_validation(self):
        env = {"FT_DISABLE_CUSTOM_AR": "0", "RTP_QWEN4_DISABLE_TP2_AR_HC_FUSED": "0"}
        with patch.dict(os.environ, env, clear=True):
            with patch("torch.cuda.is_available", return_value=False) as available:
                self.assertIsNone(
                    prepare_tp2_ar_hc_workspace(self.config, self.parallelism)
                )
                available.assert_called_once_with()

    def test_custom_allreduce_disable_overrides_opt_in(self):
        env = {"FT_DISABLE_CUSTOM_AR": "1", "RTP_QWEN4_DISABLE_TP2_AR_HC_FUSED": "0"}
        with patch.dict(os.environ, env, clear=True):
            with patch("torch.cuda.is_available") as available:
                self.assertIsNone(
                    prepare_tp2_ar_hc_workspace(self.config, self.parallelism)
                )
                available.assert_not_called()

    def test_materialize_presums_packed_dual_and_avoids_redundant_copy(self):
        routed = torch.full((4, 8), 1.0, dtype=torch.bfloat16)
        shared = torch.full((4, 8), 2.0, dtype=torch.bfloat16)
        packed = torch.cat((routed, shared), dim=0)
        captured = []

        def fake_ar(tensor, group, *, inplace=True):
            captured.append((tuple(tensor.shape), group, inplace))
            return tensor * 2

        with patch.object(collective_torch, "all_reduce", side_effect=fake_ar):
            out_packed = DeferredTp2Sublayer(packed, True, None).materialize()
            out_single = DeferredTp2Sublayer(routed, False, None).materialize()

        self.assertEqual(
            captured,
            [
                ((4, 8), collective_torch.Group.TP, None),
                ((4, 8), collective_torch.Group.TP, False),
            ],
        )
        torch.testing.assert_close(out_packed, torch.full((4, 8), 6.0, dtype=torch.bfloat16))
        torch.testing.assert_close(out_single, torch.full((4, 8), 2.0, dtype=torch.bfloat16))

    def test_workspace_init_failure_falls_back_cleanly_to_flashinfer(self):
        env = {"FT_DISABLE_CUSTOM_AR": "0", "RTP_QWEN4_DISABLE_TP2_AR_HC_FUSED": "0"}
        fake_group = object()
        fake_dev = torch.device("cuda", 0)
        votes = []

        def fake_agree(group, local_ok, tag):
            votes.append((tag, local_ok))
            return bool(local_ok)

        with (
            patch.dict(os.environ, env, clear=True),
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.device_count", return_value=2),
            patch("torch.distributed.is_initialized", return_value=True),
            patch.object(collective_torch, "_initialized", True),
            patch.object(collective_torch, "_get_group", return_value=fake_group),
            patch.object(tp2_ar_hc_fused, "cached_device_capability", return_value=(9, 0)),
            patch.object(tp2_ar_hc_fused, "_tp2_peers_can_access", return_value=True),
            patch(
                "rtp_llm.models_py.distributed.symm_mem._agree_across_group",
                side_effect=fake_agree,
            ),
            patch.object(
                tp2_ar_hc_fused,
                "Qwen4Tp2ArHcWorkspace",
                side_effect=RuntimeError("ipc error"),
            ),
        ):
            # Also mock flashinfer.comm.cuda_ipc and rtp_llm_ops for CPU-only test environments.
            fake_ops = SimpleNamespace(
                tp2_ar_hc_inject_norm=MagicMock(),
                tp2_ar_hc_inject_only=MagicMock(),
            )
            with patch.dict(
                "sys.modules",
                {
                    "flashinfer": SimpleNamespace(comm=SimpleNamespace(cuda_ipc=object())),
                    "flashinfer.comm": SimpleNamespace(cuda_ipc=object()),
                    "rtp_llm.ops.compute_ops": SimpleNamespace(rtp_llm_ops=fake_ops),
                },
            ):
                ws = prepare_tp2_ar_hc_workspace(
                    self.config, self.parallelism, device=fake_dev
                )
        self.assertIsNone(ws)
        self.assertEqual(votes, [("tp2_ar_hc_prereq", True), ("tp2_ar_hc_init", False)])


if __name__ == "__main__":
    unittest.main()

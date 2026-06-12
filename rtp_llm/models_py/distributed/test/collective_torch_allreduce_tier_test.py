import unittest
from unittest.mock import MagicMock, patch

import torch

from rtp_llm.models_py.distributed import collective_torch as ct
from rtp_llm.models_py.distributed import rocm_rccl as hr


class TestAllReduceTierDispatch(unittest.TestCase):
    """Unit tests for the eager all_reduce five-tier fallback logic."""

    def setUp(self):
        self._orig_is_rocm = ct._is_rocm_runtime
        self._orig_strategies = ct._rocm_allreduce_strategies
        self._orig_quick = ct._enable_quick_allreduce
        self._orig_trtllm = ct._enable_trtllm_allreduce
        self._orig_aiter = ct._enable_aiter_custom_ar

    def tearDown(self):
        ct._is_rocm_runtime = self._orig_is_rocm
        ct._rocm_allreduce_strategies = self._orig_strategies
        ct._enable_quick_allreduce = self._orig_quick
        ct._enable_trtllm_allreduce = self._orig_trtllm
        ct._enable_aiter_custom_ar = self._orig_aiter

    def _enable_rocm_tiers(self, quick=False, trtllm=False, aiter=False):
        ct._is_rocm_runtime = True
        ct._rocm_allreduce_strategies = {"quick", "trtllm", "custom"}
        ct._enable_quick_allreduce = quick
        ct._enable_trtllm_allreduce = trtllm
        ct._enable_aiter_custom_ar = aiter

    def _no_capture(self):
        return (
            patch.object(hr, "ensure_capture_comm_ready"),
            patch.object(hr, "should_use_capture_collectives", return_value=False),
        )

    # ---- Test 1: quick allreduce success copies result back in-place ----

    def test_quick_success_finalize_copies_back(self):
        self._enable_rocm_tiers(quick=True)
        tensor = torch.ones(4, dtype=torch.float16)
        quick_result = torch.full((4,), 2.0, dtype=torch.float16)
        mock_group = object()

        with self._no_capture()[0], self._no_capture()[1], patch.object(
            ct, "_get_group", return_value=mock_group
        ), patch.object(ct, "_try_quick_allreduce", return_value=quick_result):
            result = ct.all_reduce(tensor, ct.Group.TP)

        self.assertIs(result, tensor)
        self.assertTrue(torch.equal(tensor, quick_result))

    # ---- Test 2: quick returns None → fallback to trtllm ----

    def test_quick_none_falls_through_to_trtllm(self):
        self._enable_rocm_tiers(quick=True, trtllm=True)
        tensor = torch.ones(4, dtype=torch.float16)
        trtllm_result = torch.full((4,), 3.0, dtype=torch.float16)
        mock_group = object()

        with self._no_capture()[0], self._no_capture()[1], patch.object(
            ct, "_get_group", return_value=mock_group
        ), patch.object(ct, "_try_quick_allreduce", return_value=None), patch.object(
            ct, "_try_trtllm_allreduce", return_value=trtllm_result
        ):
            result = ct.all_reduce(tensor, ct.Group.TP)

        self.assertIs(result, tensor)
        self.assertTrue(torch.equal(tensor, trtllm_result))

    # ---- Test 3: all ROCm tiers fail → NCCL fallback ----

    def test_all_tiers_fail_falls_through_to_nccl(self):
        self._enable_rocm_tiers(quick=True, trtllm=True, aiter=True)
        tensor = torch.ones(4, dtype=torch.float16)
        mock_group = object()

        with self._no_capture()[0], self._no_capture()[1], patch.object(
            ct, "_get_group", return_value=mock_group
        ), patch.object(ct, "_try_quick_allreduce", return_value=None), patch.object(
            ct, "_try_trtllm_allreduce", return_value=None
        ), patch.object(
            ct, "_try_aiter_custom_allreduce", return_value=None
        ), patch.object(
            ct, "get_symm_mem_communicator", return_value=None
        ), patch(
            "torch.distributed.all_reduce"
        ) as mock_nccl:
            result = ct.all_reduce(tensor, ct.Group.TP)

        mock_nccl.assert_called_once()
        self.assertIs(result, tensor)

    # ---- Test 4: group != TP skips ROCm tiers entirely ----

    def test_non_tp_group_skips_rocm_tiers(self):
        self._enable_rocm_tiers(quick=True, trtllm=True, aiter=True)
        tensor = torch.ones(4, dtype=torch.float16)
        mock_group = object()

        with self._no_capture()[0], self._no_capture()[1], patch.object(
            ct, "_get_group", return_value=mock_group
        ), patch.object(ct, "_try_quick_allreduce") as mock_quick, patch.object(
            ct, "_try_trtllm_allreduce"
        ) as mock_trtllm, patch.object(
            ct, "_try_aiter_custom_allreduce"
        ) as mock_aiter, patch.object(
            ct, "get_symm_mem_communicator", return_value=None
        ), patch(
            "torch.distributed.all_reduce"
        ):
            ct.all_reduce(tensor, ct.Group.DP)

        mock_quick.assert_not_called()
        mock_trtllm.assert_not_called()
        mock_aiter.assert_not_called()

    # ---- Test 5: quick success means trtllm/aiter are never tried ----

    def test_quick_success_short_circuits(self):
        self._enable_rocm_tiers(quick=True, trtllm=True, aiter=True)
        tensor = torch.ones(4, dtype=torch.float16)
        quick_result = torch.full((4,), 2.0, dtype=torch.float16)
        mock_group = object()

        with self._no_capture()[0], self._no_capture()[1], patch.object(
            ct, "_get_group", return_value=mock_group
        ), patch.object(
            ct, "_try_quick_allreduce", return_value=quick_result
        ), patch.object(
            ct, "_try_trtllm_allreduce"
        ) as mock_trtllm, patch.object(
            ct, "_try_aiter_custom_allreduce"
        ) as mock_aiter:
            ct.all_reduce(tensor, ct.Group.TP)

        mock_trtllm.assert_not_called()
        mock_aiter.assert_not_called()


class TestFinalizeInplace(unittest.TestCase):
    """Unit tests for _finalize_inplace helper."""

    def test_none_result_returns_none(self):
        tensor = torch.ones(4)
        self.assertIsNone(ct._finalize_inplace(tensor, None))

    def test_different_tensor_copies_back(self):
        tensor = torch.ones(4)
        result = torch.full((4,), 5.0)
        ret = ct._finalize_inplace(tensor, result)
        self.assertIs(ret, tensor)
        self.assertTrue(torch.equal(tensor, result))

    def test_same_tensor_returns_identity(self):
        tensor = torch.ones(4)
        ret = ct._finalize_inplace(tensor, tensor)
        self.assertIs(ret, tensor)


if __name__ == "__main__":
    unittest.main()

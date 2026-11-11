"""CPU contracts for per-layer grouped output weight preparation."""

import sys
import types
import unittest
from unittest.mock import Mock, patch

import torch

from rtp_llm.models_py.modules.dsv4.fp8.attention_backend import SparseAttentionBackend


class GroupedOutputWeightsTest(unittest.TestCase):
    def test_default_preserves_deepgemm_expansion_and_packed_scale_contract(self):
        groups, rank, width = 2, 256, 512
        weight = torch.ones(groups * rank, width, dtype=torch.float8_e4m3fn)
        raw = (
            torch.arange(groups * (rank // 128) * (width // 128))
            .reshape(groups * rank // 128, width // 128)
            .float()
            .exp2()
        )
        packed = torch.zeros(groups, rank, width // 512, dtype=torch.int32)
        pack = Mock(return_value=packed)
        layout = types.SimpleNamespace(
            get_mn_major_tma_aligned_packed_ue8m0_tensor=pack
        )
        backend = SparseAttentionBackend()
        with patch.dict(sys.modules, {"deep_gemm.utils.layout": layout}):
            prepared_w, prepared_s = backend.prepare_grouped_output(
                weight, raw, groups, rank, width
            )
        expected = raw.view(groups, rank // 128, width // 128).repeat_interleave(
            128, dim=1
        )
        torch.testing.assert_close(pack.call_args.args[0], expected, rtol=0, atol=0)
        self.assertEqual(pack.call_count, 1)
        self.assertIs(prepared_s, packed)
        self.assertEqual(prepared_w.shape, (groups, rank, width))
        self.assertEqual(prepared_w.data_ptr(), weight.data_ptr())
        self.assertTrue(prepared_w.is_contiguous())

        kernel = Mock(side_effect=lambda equation, a, b, out, **kw: out.fill_(7))
        activation = torch.ones(3, groups, width, dtype=torch.float8_e4m3fn)
        scales = object()
        with patch.dict(
            sys.modules, {"deep_gemm": types.SimpleNamespace(fp8_einsum=kernel)}
        ):
            for _ in range(2):
                result = backend.grouped_output(
                    activation, scales, prepared_w, prepared_s, rank
                )
                self.assertIs(kernel.call_args.args[2][1], packed)
                self.assertEqual(kernel.call_args.kwargs["recipe"], (1, 1, 128))
                self.assertTrue(torch.all(result == 7))
        self.assertEqual(pack.call_count, 1)


if __name__ == "__main__":
    unittest.main()

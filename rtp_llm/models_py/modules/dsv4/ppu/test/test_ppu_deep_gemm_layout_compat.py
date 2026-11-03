"""PPU linears receive checkpoint scales before CUDA-only repacking."""
import types
import unittest
from unittest.mock import Mock, patch

import torch

from rtp_llm.models_py.modules.dsv4.ppu import utils
from rtp_llm.models_py.modules.dsv4.ppu.attention.fp8 import attention


class PpuDeepGemmLayoutCompatTest(unittest.TestCase):
    def test_attention_inherits_shared_prefill_methods_and_cp_row_mapping(self):
        from rtp_llm.models_py.modules.dsv4.fp8.attention import AttentionFP8 as Shared
        from rtp_llm.models_py.modules.dsv4.ppu.attention.fp8.attention import AttentionFP8 as Ppu

        methods = (
            "_decode_compressor_meta_from_metadata", "_should_overlap_cp_for_prefill",
            "_cp_kv_cache_sharded", "_should_async_workspace_reads_for_prefill",
            "_forward_prefill_swa_only", "_forward_prefill_hca",
            "_forward_prefill_hca_overlapped", "_cp_full_req_ids_and_positions",
            "_cp_local_full_row_indices", "_attn_via_workspace_cp_raw_q_merge",
            "_build_shared_prefill_meta", "_prefill_common_setup", "_prefill_compute_qkv",
            "_materialize_prefill_q", "_ensure_prefill_kv_full",
            "_prefill_write_swa_fp8_paged", "_attn_fp8_swa_via_kv_full",
            "_attn_fp8_swa_via_concat",
        )
        for name in methods:
            with self.subTest(method=name):
                self.assertIs(getattr(Ppu, name), getattr(Shared, name))
        common = types.SimpleNamespace(
            device=torch.device("cpu"),
            cp_ctx=types.SimpleNamespace(
                req_id_per_token=torch.tensor([1, 0, 1, 0]),
                prefix_lengths=torch.tensor([10, 20]),
                cu_seqlens_global=torch.tensor([0, 3, 6]),
                global_positions=torch.tensor([21, 12, 20, 10]),
                seq_len_full=6,
            ),
        )
        # Unequal prefixes and interleaved rank-local requests retain full-row order.
        self.assertEqual(Ppu._cp_local_full_row_indices(common).tolist(), [4, 2, 3, 0])

    def _check_raw_checkpoint_scale_dispatch(self, module):
        weight = torch.zeros((256, 256), dtype=torch.float8_e4m3fn)
        scale = torch.full((2, 2), 127, dtype=torch.uint8).view(torch.float8_e8m0fnu)
        provider = Mock()
        constructed = object()
        with patch.object(module, "_repack_v4_fp8_scale_to_int32",
                          side_effect=AssertionError("premature CUDA layout conversion")):
            with patch.object(provider, "build_fp8_linear",
                              return_value=constructed) as dispatch:
                result = module._v4_fp8_linear(weight, scale, platform_provider=provider)
        self.assertIs(result, constructed)
        self.assertIs(dispatch.call_args.args[1], weight)
        self.assertIs(dispatch.call_args.args[2], scale)

    def test_shared_utility_preserves_checkpoint_scale(self):
        self._check_raw_checkpoint_scale_dispatch(utils)

    def test_attention_preserves_checkpoint_scale(self):
        self._check_raw_checkpoint_scale_dispatch(attention)


if __name__ == "__main__":
    unittest.main()

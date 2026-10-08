import unittest
from types import SimpleNamespace

import torch

from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.mla_page_rr_prefill_params import (
    build_mla_page_rr_prefill_params,
)


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class MlaPageRRPrefillParamsTest(unittest.TestCase):
    def test_kv_indptr_uses_trusted_length_mirrors(self):
        device = torch.device("cuda:0")
        inputs = SimpleNamespace(
            input_lengths=torch.tensor([4, 3], dtype=torch.int32),
            prefix_lengths=torch.tensor([8, 12], dtype=torch.int32),
            input_lengths_device=torch.tensor([4, 3], dtype=torch.int32, device=device),
            prefix_lengths_device=torch.tensor([8, 12], dtype=torch.int32, device=device),
            cu_seqlens_device=torch.tensor([0, 4, 7], dtype=torch.int32, device=device),
            cu_kv_seqlens_device=torch.tensor(
                [0, 2_038_069_513, 2_038_069_513], dtype=torch.int32, device=device
            ),
            padding_offset=torch.zeros(7, dtype=torch.int32, device=device),
            kv_cache_kernel_block_id_device=torch.ones(
                (2, 16), dtype=torch.int32, device=device
            ),
        )

        params = build_mla_page_rr_prefill_params(inputs, page_size=4)

        self.assertEqual(params.prefill_ragged_kv_len_indptr_d.cpu().tolist(), [0, 12, 27])
        self.assertNotEqual(
            params.prefill_ragged_kv_len_indptr_d.untyped_storage().data_ptr(),
            inputs.cu_kv_seqlens_device.untyped_storage().data_ptr(),
        )
        self.assertEqual(params.qo_indptr_d.cpu().tolist(), [0, 4, 7])


if __name__ == "__main__":
    unittest.main()

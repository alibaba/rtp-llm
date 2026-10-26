import unittest

import torch

from rtp_llm.models_py.model_desc.qwen3_next import Qwen3NextMetadata
from rtp_llm.models_py.model_desc.test._linear_sparse_save_common import (
    _exercise_internal_f_last_same_slot,
    _exercise_mixed_prefix_batch,
    _exercise_sparse_save_restore,
    _exercise_tail_boundaries,
    _gdn_forward,
    _make_gdn_module,
)


class LinearSparseSaveGdnTest(unittest.TestCase):
    def test_gdn_sparse_save_and_restore(self):
        device = torch.device("cuda")
        torch.manual_seed(1729)
        torch.cuda.manual_seed_all(1729)
        module = _make_gdn_module(device)
        heads, head_dim = 2, 32
        qkv_size = heads * head_dim * 2 + heads * head_dim
        full_tokens = torch.randn(2000, qkv_size, device=device, dtype=torch.bfloat16)
        aux = (
            torch.randn(2000, heads, device=device, dtype=torch.bfloat16),
            torch.randn(2000, heads, device=device, dtype=torch.bfloat16),
        )

        _exercise_sparse_save_restore(
            module,
            _gdn_forward,
            lambda conv_metadata: Qwen3NextMetadata(prefill_conv1d_meta=conv_metadata),
            full_tokens,
            aux,
            device,
        )
        _exercise_internal_f_last_same_slot(
            module,
            _gdn_forward,
            lambda conv_metadata: Qwen3NextMetadata(prefill_conv1d_meta=conv_metadata),
            full_tokens,
            aux,
            device,
        )
        _exercise_tail_boundaries(
            module,
            _gdn_forward,
            lambda conv_metadata: Qwen3NextMetadata(prefill_conv1d_meta=conv_metadata),
            full_tokens,
            aux,
            device,
        )
        _exercise_mixed_prefix_batch(
            module,
            _gdn_forward,
            lambda conv_metadata: Qwen3NextMetadata(prefill_conv1d_meta=conv_metadata),
            full_tokens,
            aux,
            device,
        )


if __name__ == "__main__":
    unittest.main()

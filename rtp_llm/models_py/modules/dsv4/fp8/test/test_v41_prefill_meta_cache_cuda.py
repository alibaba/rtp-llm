import unittest

import torch

from rtp_llm.models_py.modules.dsv4.fp8.attention_v41 import AttentionV41FP8


@unittest.skipUnless(
    torch.cuda.is_available(), "CUDA required for real metadata builders"
)
class V41PrefillMetaCudaTest(unittest.TestCase):
    def test_cp_ragged_prefix_metadata_matches_independent_builds(self):
        from rtp_llm.models_py.modules.dsv4.fp8.test.test_prefill_common_metadata_cuda import (
            PrefillCommonMetadataCudaTest,
            _assert_exact,
            _MetadataAttention,
        )

        class Owner(_MetadataAttention, AttentionV41FP8):
            _build_shared_prefill_meta = AttentionV41FP8._build_shared_prefill_meta

        for mixed_prefix in (False, True):
            with self.subTest(mixed_prefix=mixed_prefix):
                fixture = PrefillCommonMetadataCudaTest()
                fixture.setUp()
                if mixed_prefix:
                    fixture.prefix_lengths[0] = 0
                    fixture.sp_per_req[0] = 0
                    fixture.position_ids[:3] -= 10
                    fixture.cp_ctx.prefix_lengths_host = (0, 20)
                shared = {}
                owners = []
                for ratio in (0, 2, 1):
                    owner = Owner.__new__(Owner)
                    torch.nn.Module.__init__(owner)
                    _MetadataAttention.__init__(
                        owner,
                        compress_ratio=ratio,
                        freqs_cis=(
                            fixture.base_rope if ratio == 0 else fixture.compressed_rope
                        ),
                        cp_ctx=fixture.cp_ctx,
                        block_table=fixture.block_table,
                        entries_per_block=fixture.entries_per_block,
                    )
                    owner._shared_attention = shared
                    owner._rope_base = 160000
                    owner._rope_max_seq_len = owner._rope_o_seq_len = 64
                    owner._rope_factor = 16.0
                    owner._rope_beta_fast, owner._rope_beta_slow = 32, 1
                    owner._rope_dim = 8
                    if owners:
                        owner._kv_cache = owners[0]._kv_cache
                        owner._block_tables_by_type = owners[0]._block_tables_by_type
                    owners.append(owner)
                references = []
                for owner in owners:
                    shared.clear()
                    references.append(fixture._build(owner))
                shared.clear()
                reused = [fixture._build(owner) for owner in owners]
                self.assertIs(reused[1], reused[2])
                for i, (actual, expected) in enumerate(zip(reused, references)):
                    _assert_exact(self, actual, expected, f"v41_bucket_{i}")


if __name__ == "__main__":
    unittest.main()

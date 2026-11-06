"""Regression for the native empty-position sentinel through both real APIs."""
import os
import unittest
from unittest.mock import patch
import torch
import rtp_arm_bf16_rope as backend


class EmptyPositionsTest(unittest.TestCase):
    def test_empty_sentinel_matches_implicit_positions_and_cache_boundaries(self):
        with patch.dict(os.environ,{'RTP_ARM_BF16_ROPE_ALLOW_CPU_TESTS':'1'}):
            torch.manual_seed(385)
            packed=(torch.randn(3,10*256)*.1).bfloat16()
            step=(torch.randn(1,10*256)*.1).bfloat16()
            table=torch.tensor([[3,1,4]],dtype=torch.int32)
            offsets=backend.convert_offset_to_block_array(table)
            common=dict(batch_size=1,head_num=8,kv_head_num=1,size_per_head=256,
                        tokens_per_block=64,rope_style=1,rope_dim=64,rope_base=10000000)
            cases=[]
            for positions in [None,torch.empty(0,dtype=torch.int32)]:
                cache=torch.full((5,2,1,64,256),.125,dtype=torch.bfloat16)
                query=backend.prefill_fused_rope_kvcache(packed,
                    torch.tensor([0,3],dtype=torch.int32),max_seq_len=3,
                    store_q_no_transpose=True,store_cache=True,kv_cache=cache,
                    kv_cache_offset=offsets,prefix_prompt_lengths=torch.tensor([63],dtype=torch.int32),
                    position_ids=positions,**common)
                decoded=backend.decode_fused_rope_kvcache(step,positions,
                    sequence_lengths=torch.tensor([66],dtype=torch.int32),
                    kv_cache=cache,kv_cache_offset=offsets,**common)
                self.assertTrue(torch.equal(cache[0],torch.full_like(cache[0],.125)))
                cases.append((query,decoded,cache))
            for none,empty in zip(*cases):self.assertTrue(torch.equal(none,empty))
            with self.assertRaisesRegex(ValueError,'one position/token'):
                backend.prefill_fused_rope_kvcache(packed,
                    torch.tensor([0,3],dtype=torch.int32),max_seq_len=3,
                    store_q_no_transpose=True,position_ids=torch.tensor([1,2]),**common)


if __name__=='__main__':unittest.main()

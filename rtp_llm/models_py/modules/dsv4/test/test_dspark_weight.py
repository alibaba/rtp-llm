import unittest

import torch

from rtp_llm.models.deepseek_v4 import DSparkReplicatedLmHeadWeight
from rtp_llm.utils.model_weight import W, identity, sp_id


class DSparkWeightTest(unittest.TestCase):
    def test_lm_head_is_replicated_across_prefill_cp_ranks(self) -> None:
        weight = DSparkReplicatedLmHeadWeight(W.lm_head, [], identity)
        split = weight._get_split_func()
        self.assertIs(split, sp_id)

        full_head = torch.arange(32, dtype=torch.bfloat16).reshape(8, 4)
        rank0 = split(full_head, tp=2, tp_rank=0)
        rank1 = split(full_head, tp=2, tp_rank=1)
        self.assertEqual(tuple(rank0.shape), (8, 4))
        self.assertEqual(tuple(rank1.shape), (8, 4))
        torch.testing.assert_close(rank0, full_head)
        torch.testing.assert_close(rank1, full_head)


if __name__ == "__main__":
    unittest.main()

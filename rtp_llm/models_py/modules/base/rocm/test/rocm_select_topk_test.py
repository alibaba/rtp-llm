import types
from unittest import SkipTest, TestCase, main

import torch
import torch.nn.functional as F

from rtp_llm.models_py.modules.base.rocm.select_topk import SelectTopk


class RocmSelectTopkTest(TestCase):
    def setUp(self) -> None:
        if not torch.cuda.is_available() or torch.version.hip is None:
            raise SkipTest("ROCm is not available")

    def test_qwen2_moe_preserves_renormalize_configuration(self):
        logits = torch.tensor(
            [[1.0, 0.5, -0.25, 2.0], [0.1, 1.5, 0.3, -0.2]],
            dtype=torch.float32,
            device="cuda",
        )
        probabilities = F.softmax(logits, dim=-1)
        expected_weights, expected_ids = torch.topk(probabilities, 2, dim=-1)

        for renormalize in (False, True):
            with self.subTest(renormalize=renormalize):
                config = types.SimpleNamespace(
                    moe_k=2,
                    has_moe_norm=renormalize,
                )
                topk_ids = torch.empty((2, 2), dtype=torch.int32, device=logits.device)
                topk_weights = torch.empty(
                    (2, 2), dtype=torch.float32, device=logits.device
                )
                SelectTopk(config)(logits, topk_ids, topk_weights)

                if renormalize:
                    reference_weights = expected_weights / expected_weights.sum(
                        dim=-1, keepdim=True
                    )
                else:
                    reference_weights = expected_weights
                actual_order = torch.argsort(topk_ids, dim=-1)
                expected_order = torch.argsort(expected_ids, dim=-1)
                self.assertTrue(
                    torch.equal(
                        torch.gather(topk_ids, 1, actual_order),
                        torch.gather(expected_ids.int(), 1, expected_order),
                    )
                )
                self.assertTrue(
                    torch.allclose(
                        torch.gather(topk_weights, 1, actual_order),
                        torch.gather(reference_weights, 1, expected_order),
                        atol=1e-5,
                        rtol=1e-5,
                    )
                )


if __name__ == "__main__":
    main()

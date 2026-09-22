"""Manual GPU correctness for regular and K3-SE MoE input packing."""

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ["GPU_COUNT"] = "1"
import unittest
from functools import partial
from types import SimpleNamespace
from unittest.mock import patch

import torch

from rtp_llm.models_py.modules.dsv4.moe.input_packer import (
    FusedMegaMoeInputPacker,
    TorchMegaMoeInputPacker,
)
from rtp_llm.models_py.modules.kimi_k3.input_packer_se import (
    FusedKimiK3MegaMoeSeInputPacker,
)

assert_exact = partial(torch.testing.assert_close, rtol=0, atol=0)


def buffer(tokens, width=128, topk=1):
    return SimpleNamespace(
        num_max_tokens_per_rank=tokens,
        x=torch.full((tokens, width), 8, device="cuda", dtype=torch.float8_e4m3fn),
        x_sf=torch.full((tokens, width // 128), 9, device="cuda", dtype=torch.int32),
        topk_idx=torch.full((tokens, topk), 9, device="cuda", dtype=torch.int64),
        topk_weights=torch.full((tokens, topk), 9.0, device="cuda"),
    )


class MoeInputPackerGpuTest(unittest.TestCase):
    def setUp(self):
        env = patch.dict(os.environ, {"DSV4_MOE_STRICT_FUSED": "0"})
        env.start()
        self.addCleanup(env.stop)

    def _assert_pack(
        self, output, x, weights, ids, mask, count, shared=None, shared_storage=None
    ):
        tokens, width = x.shape
        reference = buffer(tokens + 2, width, ids.shape[1])
        TorchMegaMoeInputPacker().pack(x, weights, ids, reference, tokens)
        assert_exact(output.x.view(torch.uint8), reference.x.view(torch.uint8))
        assert_exact(output.x_sf, reference.x_sf)

        valid = mask & (torch.arange(tokens, device="cuda") < count)
        assert_exact(output.topk_idx[:tokens], torch.where(valid[:, None], ids, 0))
        assert_exact(
            output.topk_weights[:tokens], torch.where(valid[:, None], weights, 0)
        )
        self.assertTrue(torch.all(output.topk_idx[tokens:] == 9))
        self.assertTrue(torch.all(output.topk_weights[tokens:] == 9))
        self.assertTrue(torch.all(output.x[tokens:].float() == 8))
        self.assertTrue(torch.all(output.x_sf[tokens:] == 9))

        if shared is not None:
            shared_width = shared.shape[1]
            assert_exact(shared_storage[:tokens, :shared_width], shared)
            self.assertTrue(torch.all(shared_storage[tokens:] == 7))
            self.assertTrue(torch.all(shared_storage[:tokens, shared_width:] == 7))

    def test_regular_and_se_pack_mask_count_tails_strides_and_graph_replay(self):
        torch.manual_seed(19)
        tokens, width, topk = 513, 3584, 16
        # The legacy case is the single fallback representative; optimized
        # regular and SE cases exercise graph replay from changed input values.
        cases = (
            ("regular-legacy", FusedMegaMoeInputPacker(), "legacy", None, False),
            ("regular", FusedMegaMoeInputPacker(), "optimized", None, True),
            (
                "se-tp-shard",
                FusedKimiK3MegaMoeSeInputPacker(),
                "optimized",
                896,
                True,
            ),
        )
        for name, packer, implementation, shared_width, replay in cases:
            with self.subTest(case=name), patch.dict(
                os.environ, {"DSV4_MEGA_MOE_INPUT_PACKER_IMPL": implementation}
            ):
                x = torch.randn((tokens, width), device="cuda", dtype=torch.bfloat16)
                weights = torch.rand((tokens, topk), device="cuda")
                ids = torch.randint(-1, 896, (tokens, topk), device="cuda")
                mask = torch.arange(tokens, device="cuda") % 3 != 0
                count = tokens - 1
                output = buffer(tokens + 2, width, topk)
                options = dict(valid_token_count=count, valid_token_mask=mask)
                shared = shared_storage = None
                if shared_width is not None:
                    shared = torch.randn(
                        (tokens, shared_width + 128),
                        device="cuda",
                        dtype=torch.bfloat16,
                    )[:, :shared_width]
                    shared_storage = torch.full(
                        (tokens + 2, shared_width + 128),
                        7,
                        device="cuda",
                        dtype=torch.bfloat16,
                    )
                    options.update(
                        shared_input=shared,
                        shared_out=shared_storage[:, :shared_width],
                    )

                def run():
                    packer.pack(x, weights, ids, output, tokens, **options)

                run()
                torch.cuda.synchronize()
                self._assert_pack(
                    output, x, weights, ids, mask, count, shared, shared_storage
                )
                if not replay:
                    continue

                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    run()
                x.add_(0.5)
                weights.mul_(0.5)
                ids.add_(1)
                mask.logical_not_()
                if shared is not None:
                    shared.add_(1)
                graph.replay()
                self._assert_pack(
                    output, x, weights, ids, mask, count, shared, shared_storage
                )


if __name__ == "__main__":
    unittest.main()

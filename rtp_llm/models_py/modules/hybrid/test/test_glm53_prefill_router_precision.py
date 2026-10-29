"""GLM53 Prefill routing is invariant to batch tails and token shard ownership."""

import unittest
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from torch import nn

from rtp_llm.models_py.distributed.sequence_parallel import token_shard_layout
from rtp_llm.models_py.modules import GroupTopK
from rtp_llm.models_py.modules.dsv4.forward_metadata import scoped_forward_metadata
from rtp_llm.models_py.modules.factory.linear.fixed_m_linear import fixed_m_linear
from rtp_llm.models_py.modules.glm53_router import Glm53FP32Router


class _Model(nn.Module):
    def __init__(self, gate):
        super().__init__()
        self.gate = gate
        self.config = SimpleNamespace(model_type="glm5_3_flash")

    @scoped_forward_metadata
    def forward(self, inputs, x, layout=None, rows=None):
        if rows is not None:
            return fixed_m_linear(self.gate, x, rows)
        return self.gate(x) if layout is None else self.gate.forward_shard(x, layout)


def prefill():
    return SimpleNamespace(
        attention_inputs=SimpleNamespace(is_prefill=True, is_target_verify=False)
    )


def fixed_reference(x, weight, rows=256):
    output = []
    for block in x.split(rows):
        padded = x.new_zeros((rows, x.shape[1]))
        padded[: block.shape[0]].copy_(block)
        output.append(F.linear(padded.float(), weight)[: block.shape[0]])
    return (
        torch.cat(output)
        if output
        else x.new_empty((0, weight.shape[0]), dtype=torch.float32)
    )


def route(logits, bias):
    weights = torch.empty(logits.shape[0], 8, device="cuda")
    ids = torch.empty_like(weights, dtype=torch.int64)
    GroupTopK()(
        topk_weights=weights,
        topk_ids=ids,
        scores=logits,
        correction_bias=bias,
        n_group=1,
        topk_group=1,
        topk=8,
        renormalize=True,
        routed_scaling_factor=2.5,
    )
    return ids, weights


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class Glm53PrefillRouterPrecisionTest(unittest.TestCase):
    def setUp(self):
        self.old_tf32 = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        self.addCleanup(
            setattr, torch.backends.cuda.matmul, "allow_tf32", self.old_tf32
        )
        gen = torch.Generator(device="cuda").manual_seed(533136)
        self.gate = Glm53FP32Router(
            (torch.randn(4096, 288, generator=gen, device="cuda") / 64)
            .bfloat16()
            .float()
        )
        self.x = torch.randn(4, 4096, generator=gen, device="cuda").bfloat16()
        self.bias = torch.randn(288, generator=gen, device="cuda") / 100
        self.model = _Model(self.gate)

    def test_same_logits_and_routes_at_observed_gemm_tail_sizes(self):
        expected = fixed_reference(self.x, self.gate.weight)
        expected_ids, expected_weights = route(expected, self.bias)
        # 368/2576 and 5908/396 are the final global router chunks of
        # two real frozen prefixes, alone and in a batch of seven.
        for count, offset in (
            (368, 364),
            (2576, 2572),
            (5908, 5904),
            (396, 392),
            (8192, 364),
            (8192, 8188),
            (512, 1),
            (512, 17),
            (512, 63),
            (512, 127),
            (512, 252),
        ):
            with self.subTest(count=count, offset=offset):
                block = self.x.new_zeros((count, 4096))
                block[offset : offset + 4].copy_(self.x)
                actual = self.model(prefill(), block)[offset : offset + 4]
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                ids, weights = route(actual, self.bias)
                torch.testing.assert_close(ids, expected_ids, rtol=0, atol=0)
                torch.testing.assert_close(weights, expected_weights, rtol=0, atol=0)

    def test_shard_boundaries_preserve_identical_tokens_and_zero_padding(self):
        expected = fixed_reference(self.x, self.gate.weight)
        for length, batch, occurrence in (
            (33136, 1, 0),
            (33136, 7, 0),
            (33136, 7, 6),
            (38676, 1, 0),
            (38676, 7, 0),
            (38676, 7, 6),
        ):
            total, end = length * batch, length * (occurrence + 1)
            rank = min(7, (end - 1) // ((total + 7) // 8))
            layout = token_shard_layout(total, 8, rank)
            offset = end - 4 - layout.local_start
            block = self.x.new_zeros((layout.local_tokens, 4096))
            block[offset : offset + 4].copy_(self.x)
            block[layout.local_valid_tokens :].fill_(3.5)
            with self.subTest(
                length=length, batch=batch, occurrence=occurrence, rank=rank
            ):
                actual = self.model(prefill(), block, layout)
                torch.testing.assert_close(
                    actual[offset : offset + 4], expected, rtol=0, atol=0
                )
                self.assertEqual(
                    int(torch.count_nonzero(actual[layout.local_valid_tokens :])), 0
                )

    def test_fp32_inputs_and_weights_are_preserved(self):
        self.gate.weight.add_(0.000001)
        x = self.x.float() + 0.000001
        torch.testing.assert_close(
            self.model(prefill(), x),
            fixed_reference(x, self.gate.weight),
            rtol=0,
            atol=0,
        )
        self.assertFalse(
            torch.equal(self.gate.weight, self.gate.weight.bfloat16().float())
        )

    def test_empty_input_and_entirely_padded_shard(self):
        x = self.x[:0]
        self.assertEqual(self.model(prefill(), x).shape, (0, 288))
        layout = token_shard_layout(1, 8, 7)
        actual = self.model(prefill(), self.x[:1], layout)
        torch.testing.assert_close(actual, torch.zeros_like(actual), rtol=0, atol=0)

    def test_explicit_fixed_m_override_keeps_its_requested_reduction(self):
        x = self.x.repeat(99, 1)
        actual = self.model(prefill(), x, rows=64)
        torch.testing.assert_close(
            actual, fixed_reference(x, self.gate.weight, rows=64), rtol=0, atol=0
        )


if __name__ == "__main__":
    unittest.main()

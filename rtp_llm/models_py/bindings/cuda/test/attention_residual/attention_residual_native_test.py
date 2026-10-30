"""Execute the native attention-residual binding against a CPU FP64 oracle."""

import unittest

import torch

from rtp_llm.ops.compute_ops import rtp_llm_ops


def _reference(prefix, delta, blocks, norm, projection, output_norm, active, write):
    # Prefix storage is BF16: the residual addition rounds before selection.
    updated = prefix.clone()
    if delta is not None:
        updated = (prefix.float() + delta.float()).bfloat16()
    bank = blocks.clone()
    if write >= 0:
        bank[:, write] = updated
    candidates = torch.cat((bank[:, :active], updated[:, None]), dim=1).double()
    normalized = candidates / torch.sqrt(
        candidates.square().mean(dim=-1, keepdim=True) + 1e-6
    )
    scores = (normalized * norm.double() * projection.double()).sum(dim=-1)
    probabilities = torch.softmax(scores, dim=1)
    # No intermediate BF16 rounding before the optional output RMSNorm.
    mixed = (candidates * probabilities[:, :, None]).sum(dim=1)
    if output_norm is not None:
        mixed = (
            mixed
            / torch.sqrt(mixed.square().mean(dim=-1, keepdim=True) + 1e-5)
            * output_norm.double()
        )
    return mixed.bfloat16(), updated, bank


class AttentionResidualNativeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise RuntimeError("Native attention-residual tests require a CUDA GPU")
        torch.cuda.set_device(0)
        if torch.cuda.get_device_capability()[0] != 10:
            raise RuntimeError("Native attention-residual tests require SM100 family")

    def _fixture(self, tokens, seed=81):
        generator = torch.Generator().manual_seed(seed)

        def random(*shape, scale=1.0):
            return (torch.randn(*shape, generator=generator) * scale).bfloat16()

        return (
            random(tokens, 7168),
            random(tokens, 7168, scale=0.2),
            random(tokens, 9, 7168),
            random(7168, scale=0.7),
            random(7168, scale=0.01),
            (1 + random(7168, scale=0.1).float()).bfloat16(),
        )

    def _call(
        self, prefix, delta, bank, norm, projection, output_norm, output, active, write
    ):
        rtp_llm_ops.attn_res(
            prefix,
            delta,
            bank,
            norm,
            projection,
            output_norm,
            output,
            active,
            write,
            1e-6,
            1e-5,
        )

    def test_native_selection_residual_updates_and_fused_rmsnorm(self):
        # Include every source-count specialization and a multi-CTA token batch.
        for active in range(9):
            for has_delta, has_norm, write in (
                (False, False, -1),
                (False, True, active),
                (True, True, -1),
                (True, False, active),
            ):
                with self.subTest(
                    active=active, delta=has_delta, norm=has_norm, write=write
                ):
                    prefix, delta, bank, norm, projection, output_norm = (
                        self._fixture(5)
                    )
                    delta = delta if has_delta else None
                    output_norm = output_norm if has_norm else None
                    expected, expected_prefix, expected_bank = _reference(
                        prefix,
                        delta,
                        bank,
                        norm,
                        projection,
                        output_norm,
                        active,
                        write,
                    )
                    operands = [
                        value.cuda() if value is not None else None
                        for value in (
                            prefix, delta, bank, norm, projection, output_norm
                        )
                    ]
                    output = torch.empty_like(operands[0])
                    self._call(*operands, output, active, write)
                    # Fast FP32 reductions/exp2 and final BF16 rounding need not
                    # be bit-identical to the independent FP64 softmax oracle.
                    torch.testing.assert_close(
                        output.cpu(), expected, rtol=1e-2, atol=1e-2
                    )
                    torch.testing.assert_close(
                        operands[0].cpu(), expected_prefix, rtol=0, atol=0
                    )
                    torch.testing.assert_close(
                        operands[2].cpu(), expected_bank, rtol=0, atol=0
                    )

    def test_graph_replay_consumes_new_inputs_on_current_stream(self):
        initial = self._fixture(3)
        operands = [value.cuda() for value in initial]
        output = torch.empty_like(operands[0])
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                self._call(*operands, output, 4, -1)
        stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            self._call(*operands, output, 4, -1)
        for seed in (83, 89):
            fresh = self._fixture(3, seed)
            expected, expected_prefix, expected_bank = _reference(*fresh, 4, -1)
            with torch.cuda.stream(stream):
                for destination, source in zip(operands, fresh):
                    destination.copy_(source)
                graph.replay()
            stream.synchronize()
            torch.testing.assert_close(output.cpu(), expected, rtol=1e-2, atol=1e-2)
            torch.testing.assert_close(
                operands[0].cpu(), expected_prefix, rtol=0, atol=0
            )
            torch.testing.assert_close(operands[2].cpu(), expected_bank, rtol=0, atol=0)

    def test_empty_tokens_and_invalid_active_count(self):
        operands = [value.cuda() for value in self._fixture(0)]
        self._call(*operands, torch.empty_like(operands[0]), 0, -1)
        operands = [value.cuda() for value in self._fixture(1)]
        with self.assertRaisesRegex(RuntimeError, "num_blocks must be"):
            self._call(*operands, torch.empty_like(operands[0]), 9, -1)


if __name__ == "__main__":
    unittest.main()

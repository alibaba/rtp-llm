import itertools
import unittest

import torch

from rtp_llm.multimodal.multimodal_mixins.qwen3_5_moe.gpu_video import nv12_to_rgb


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class VisionKernelsTest(unittest.TestCase):
    def test_nv12_matches_integer_reference_for_both_color_matrices(self):
        torch.manual_seed(27)
        for n, h, w in [(1, 2, 2), (3, 32, 48), (2, 720, 1280)]:
            video = torch.randint(0, 256, (n, h * 3 // 2, w), dtype=torch.uint8)
            for color_space in (1, 2, 5, 6):
                expected = nv12_to_rgb(video, h, color_space)
                actual = nv12_to_rgb(video.cuda(), h, color_space)
                torch.testing.assert_close(actual.cpu(), expected, atol=0, rtol=0)


@unittest.skipUnless(
    torch.cuda.is_available() and torch.cuda.get_device_capability() == (10, 3),
    "requires SM103",
)
class VisionDenseAttentionTest(unittest.TestCase):
    def _inputs(self, lengths, dtype=torch.bfloat16, dim=72):
        torch.manual_seed(917)
        packed = torch.randn(sum(lengths), 3, 16, dim, device="cuda", dtype=dtype)
        q, k, v = packed.unbind(1)
        # RoPE materializes Q/K; V retains the interleaved QKV storage.
        q, k = q.contiguous(), k.contiguous()
        cu = torch.tensor(
            [0] + list(itertools.accumulate(lengths)),
            device="cuda",
            dtype=torch.int32,
        )
        return q, k, v, cu

    def test_dense_dispatch_preserves_segments_strides_and_reference(self):
        from unittest.mock import patch

        from rtp_llm.multimodal.multimodal_mixins.qwen3_5_moe import (
            qwen3_5_moe_vit as vm,
        )

        lengths = [1025, 1025]
        q, k, v, cu = self._inputs(lengths)
        fa4, _ = vm._flash_attention_backends()
        dense = vm._dense_flash_attention_backend()
        with torch.inference_mode(), patch.object(
            vm, "_dense_flash_attention_backend", return_value=dense
        ) as selected:
            actual = vm._fa4_vision_attention(q, k, v, cu, lengths, 72**-0.5)
            expected = fa4(
                q,
                k,
                v,
                cu_seqlens_q=cu,
                cu_seqlens_k=cu,
                max_seqlen_q=1025,
                max_seqlen_k=1025,
                causal=False,
                softmax_scale=72**-0.5,
            )
            if isinstance(expected, tuple):
                expected = expected[0]
            self.assertEqual(selected.call_count, 1)
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
            self.assertEqual(v.view(2, 1025, 16, 72).data_ptr(), v.data_ptr())
            self.assertEqual(v.stride(0), 3 * 16 * 72)
            before = actual.clone()
            v[1025:].add_(1)
            changed = vm._fa4_vision_attention(q, k, v, cu, lengths, 72**-0.5)
            torch.testing.assert_close(changed[:1025], before[:1025], atol=0, rtol=0)
            self.assertFalse(torch.equal(changed[1025:], before[1025:]))

    def test_mixed_short_and_other_shapes_retain_varlen(self):
        from unittest.mock import patch

        from rtp_llm.multimodal.multimodal_mixins.qwen3_5_moe import (
            qwen3_5_moe_vit as vm,
        )

        cases = [
            ([1025, 129], torch.bfloat16, 72),
            ([129, 129], torch.bfloat16, 72),
            ([1025, 1025], torch.float16, 72),
            ([1025, 1025], torch.bfloat16, 64),
        ]
        fa4, _ = vm._flash_attention_backends()
        for lengths, dtype, dim in cases:
            with self.subTest(lengths=lengths, dtype=dtype, dim=dim):
                q, k, v, cu = self._inputs(lengths, dtype, dim)
                with torch.inference_mode(), patch.object(
                    vm,
                    "_dense_flash_attention_backend",
                    side_effect=AssertionError("unexpected dense dispatch"),
                ):
                    actual = vm._fa4_vision_attention(q, k, v, cu, lengths, dim**-0.5)
                    expected = fa4(
                        q,
                        k,
                        v,
                        cu_seqlens_q=cu,
                        cu_seqlens_k=cu,
                        max_seqlen_q=max(lengths),
                        max_seqlen_k=max(lengths),
                        causal=False,
                        softmax_scale=dim**-0.5,
                    )
                    torch.testing.assert_close(actual, expected, atol=0, rtol=0)

    def test_attention_preserves_native_head_size_full_output_and_graph(self):
        from unittest.mock import patch

        from rtp_llm.multimodal.multimodal_mixins.qwen3_5_moe import (
            qwen3_5_moe_vit as vm,
        )

        torch.manual_seed(918)
        lengths = [1025, 1025]
        module = vm.Qwen3_5MoeVisionAttention(vm.Qwen3_5MoeVisionConfig()).cuda()
        module = module.to(torch.bfloat16).eval()
        with torch.no_grad():
            for parameter in module.parameters():
                parameter.normal_(mean=0.0, std=0.02)
        module.attn.backend = "fa4"
        hidden = torch.randn(2050, 1, 1152, device="cuda", dtype=torch.bfloat16)
        angles = torch.randn(2050, 36, device="cuda", dtype=torch.bfloat16)
        cu = torch.tensor([0, 1025, 2050], device="cuda", dtype=torch.int32)
        kwargs = dict(
            cu_seqlens=cu,
            rotary_pos_emb_cos=angles.cos(),
            rotary_pos_emb_sin=angles.sin(),
            sequence_lengths=tuple(lengths),
            max_seqlen=torch.tensor(max(lengths), dtype=torch.int32),
        )
        dense = vm._dense_flash_attention_backend()
        calls = []

        def checked_dense(q, k, v, **attention_kwargs):
            calls.append((tuple(q.shape), tuple(k.shape), tuple(v.shape)))
            self.assertEqual(attention_kwargs["softmax_scale"], 72**-0.5)
            return dense(q, k, v, **attention_kwargs)

        with torch.inference_mode():
            with patch.object(vm, "_use_dense_fa4", return_value=False):
                expected = module(hidden, **kwargs)
            with patch.object(
                vm, "_dense_flash_attention_backend", return_value=checked_dense
            ):
                actual = module(hidden, **kwargs)
            self.assertEqual(
                calls, [((2, 1025, 16, 72), (2, 1025, 16, 72), (2, 1025, 16, 72))]
            )
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):
                    module(hidden, **kwargs)
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                captured = module(hidden, **kwargs)
            graph.replay()
            first = captured.clone()
            hidden[1025:].mul_(0.5)
            graph.replay()
            with patch.object(vm, "_use_dense_fa4", return_value=False):
                expected = module(hidden, **kwargs)
            torch.testing.assert_close(captured, expected, atol=0, rtol=0)
            torch.testing.assert_close(captured[:1025], first[:1025], atol=0, rtol=0)
            self.assertFalse(torch.equal(captured[1025:], first[1025:]))

    def test_dense_cuda_graph_replay_uses_updated_input(self):
        from rtp_llm.multimodal.multimodal_mixins.qwen3_5_moe import (
            qwen3_5_moe_vit as vm,
        )

        lengths = [1025, 1025]
        q, k, v, cu = self._inputs(lengths)
        with torch.inference_mode():
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):
                    vm._fa4_vision_attention(q, k, v, cu, lengths, 72**-0.5)
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                captured = vm._fa4_vision_attention(q, k, v, cu, lengths, 72**-0.5)
            graph.replay()
            first = captured.clone()
            v.mul_(0.5)
            graph.replay()
            expected = vm._fa4_vision_attention(q, k, v, cu, lengths, 72**-0.5)
            torch.testing.assert_close(captured, expected, atol=0, rtol=0)
            self.assertFalse(torch.equal(first, captured))


if __name__ == "__main__":
    unittest.main()

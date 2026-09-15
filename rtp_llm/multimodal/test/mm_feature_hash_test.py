import unittest

import torch

from rtp_llm.ops import get_multimodal_feature_hash


class FeatureHashTest(unittest.TestCase):
    def test_row_hashes_respect_storage_and_layout(self):
        hash_rows = get_multimodal_feature_hash
        # Odd row byte counts exercise the partial final 64-bit word.
        for dtype in (torch.uint8, torch.int32, torch.float32, torch.bfloat16):
            data = torch.arange(105).reshape(7, 15).to(dtype)
            reference = hash_rows(data)
            self.assertEqual(reference.dtype, torch.int32)
            self.assertEqual(reference.device.type, "cpu")
            self.assertEqual(reference.shape, (7,))
            torch.testing.assert_close(
                reference, hash_rows(data.clone()), atol=0, rtol=0
            )
            reordered = data[[5, 2, 1]]
            torch.testing.assert_close(
                hash_rows(reordered), reference[[5, 2, 1]], atol=0, rtol=0
            )
            transposed = data.t()
            torch.testing.assert_close(
                hash_rows(transposed),
                hash_rows(transposed.contiguous()),
                atol=0,
                rtol=0,
            )
            changed = data.clone()
            changed[2, -1] += 1
            actual = hash_rows(changed)
            self.assertNotEqual(actual[2], reference[2])
            torch.testing.assert_close(
                actual[[0, 1, 3, 4, 5, 6]],
                reference[[0, 1, 3, 4, 5, 6]],
                atol=0,
                rtol=0,
            )
        for invalid in (torch.empty(0, 8), torch.empty(1, 0), torch.tensor(1)):
            with self.assertRaises(RuntimeError):
                hash_rows(invalid)

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_cuda_matches_cpu_on_nondefault_stream(self):
        hash_rows = get_multimodal_feature_hash
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            for dtype in (torch.uint8, torch.int32, torch.float32, torch.bfloat16):
                for shape in ((17, 3), (32, 2048), (9, 2, 7)):
                    data = (
                        torch.arange(torch.tensor(shape).prod().item())
                        .reshape(shape)
                        .to(dtype)
                    )
                    torch.testing.assert_close(
                        hash_rows(data.cuda()), hash_rows(data), atol=0, rtol=0
                    )
                    view = data.cuda().transpose(0, 1)
                    torch.testing.assert_close(
                        hash_rows(view), hash_rows(view.cpu()), atol=0, rtol=0
                    )

    def test_shared_spans_preserve_separators_and_reject_overlap(self):
        from libth_transformer_config import get_multimodal_token_spans

        spans = get_multimodal_token_spans
        tokens = [1, 90, 5, 91, 2, 99, 3]
        self.assertEqual(spans(tokens, [[90, 91], [99]], False), [(2, 3), (5, 6)])
        self.assertEqual(spans(tokens, [[90, 91], [99]], True), [(1, 4), (5, 6)])
        for bad, separators in (
            ([90, 90, 91], [[90, 91]]),
            ([90, 1], [[90, 91]]),
            ([90, 99, 91], [[90, 91], [99]]),
            ([1], [[]]),
        ):
            with self.assertRaises(ValueError):
                spans(bad, separators, False)


if __name__ == "__main__":
    unittest.main()

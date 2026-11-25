import unittest

import torch

from rtp_llm.ops import get_multimodal_feature_hash, get_multimodal_feature_hashes


class FeatureHashBatchTest(unittest.TestCase):
    def assert_hashes_match(self, embeddings, expected_embeddings):
        actual = get_multimodal_feature_hashes(embeddings)
        self.assertEqual(len(actual), len(embeddings))
        for value, expected in zip(actual, expected_embeddings):
            torch.testing.assert_close(value, get_multimodal_feature_hash(expected))
            self.assertEqual(value.device.type, "cpu")
            self.assertEqual(value.dtype, torch.int32)
            self.assertEqual(value.untyped_storage().nbytes(), value.numel() * 4)
        self.assertEqual(
            len({value.untyped_storage().data_ptr() for value in actual}), len(actual)
        )
        return actual

    @staticmethod
    def embeddings(device):
        return [
            torch.arange(39, device=device, dtype=torch.uint8).reshape(3, 13),
            torch.arange(24, device=device, dtype=torch.float32).reshape(4, 6).t(),
            torch.arange(128, device=device, dtype=torch.bfloat16).reshape(4, 32),
        ]

    def test_cpu_empty_single_and_heterogeneous_batch(self):
        self.assertEqual(get_multimodal_feature_hashes([]), [])
        values = self.embeddings("cpu")
        self.assert_hashes_match(values[:1], values[:1])
        self.assert_hashes_match(values, values)

    def test_rejects_empty_tensors_empty_rows_and_scalars(self):
        for invalid, message in (
            (torch.empty(0, 4), "tensor is empty"),
            (torch.empty(3, 0), "row is empty"),
            (torch.tensor(1.0), "tensor is empty"),
        ):
            with self.subTest(shape=invalid.shape):
                with self.assertRaisesRegex(RuntimeError, message):
                    get_multimodal_feature_hashes([torch.ones(2, 4), invalid])

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_cuda_byte_parity_and_heterogeneous_devices(self):
        cuda = self.embeddings("cuda")
        cpu = self.embeddings("cpu")
        self.assert_hashes_match(cuda, cpu)
        self.assert_hashes_match([cuda[0], cpu[1], cuda[2]], cpu)
        if torch.cuda.device_count() > 1:
            other = self.embeddings("cuda:1")
            self.assert_hashes_match([cuda[0], other[1], cuda[2]], cpu)

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_cuda_uses_the_callers_stream_and_owns_each_cpu_result(self):
        stream = torch.cuda.Stream()
        expected = [torch.full((184, 32), float(index + 2)) for index in range(4)]
        with torch.cuda.stream(stream):
            values = [torch.ones(184, 32, device="cuda") for _ in expected]
            for index, value in enumerate(values):
                value.mul_(index + 2)
            # Inspect CPU bytes immediately without synchronizing the test stream.
            results = self.assert_hashes_match(values, expected)
            snapshots = [result.clone() for result in results]
            for value in values:
                value.zero_()
            get_multimodal_feature_hashes(values)
        for result, snapshot in zip(results, snapshots):
            torch.testing.assert_close(result, snapshot)

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_cuda_rejects_late_invalid_embedding(self):
        with self.assertRaisesRegex(RuntimeError, "row is empty"):
            get_multimodal_feature_hashes(
                [torch.ones(3, 4, device="cuda"), torch.empty(2, 0, device="cuda")]
            )


if __name__ == "__main__":
    unittest.main()

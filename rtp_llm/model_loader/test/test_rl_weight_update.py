"""Transaction failure and source-lifetime tests using the real HF converter."""

import unittest

import torch

from rtp_llm.model_loader.model_weight_info import ModelWeights
from rtp_llm.model_loader.rl_weight_update import RlWeightUpdate
from rtp_llm.model_loader.test.test_tensor_weight_stream import inputs, make_loader


class RlWeightUpdateTest(unittest.TestCase):
    def setUp(self):
        self.loader = make_loader(generated=True)
        self.sources = [(name, value.bfloat16()) for name, value in inputs()]
        self.shapes = {name: tuple(value.shape) for name, value in self.sources}
        self.weights = ModelWeights.__new__(ModelWeights)
        self.weights.global_weights = {}
        self.weights.weights = [{}]
        for layer, name, value in self.loader.prepare_weights_from_tensors(
            self.sources, "cpu", max_buffer_bytes=4096
        ):
            target = (
                self.weights.global_weights
                if layer is None
                else self.weights.weights[layer]
            )
            target[name] = torch.zeros_like(value)

    def transaction(self):
        return RlWeightUpdate(self.loader, self.weights, "cpu", self.shapes, 4096)

    def test_ack_allows_immediate_source_reuse_and_preserves_storage(self):
        expected = {
            (layer, name): value.clone()
            for layer, name, value in self.loader.prepare_weights_from_tensors(
                self.sources, "cpu", max_buffer_bytes=4096
            )
        }
        session = self.transaction()
        pointers = {key: value.data_ptr() for key, value in session.targets.items()}
        for name, value in self.sources:
            source = value.clone()
            session.push(name, source, timeout=2)
            source.fill_(999)
        result = session.finish(timeout=2)
        self.assertEqual(result["hf_sources"], len(self.sources))
        self.assertEqual(result["internal_tensors"], len(expected))
        for key, value in session.targets.items():
            self.assertTrue(torch.equal(value, expected[key]), key)
            self.assertEqual(value.data_ptr(), pointers[key])
        with self.assertRaises(RuntimeError):
            session.push(*self.sources[0])
        with self.assertRaises(RuntimeError):
            session.finish()

    def test_partial_stream_cannot_commit(self):
        session = self.transaction()
        session.push(*self.sources[0], timeout=2)
        with self.assertRaisesRegex(ValueError, "missing HF tensors"):
            session.finish(timeout=2)

    def test_bad_source_fails_push_and_commit(self):
        name, value = self.sources[0]
        for label, source_name, tensor, pattern in (
            ("unknown", "unknown.weight", value, "unknown HF"),
            ("shape", name, value.flatten(), "shape mismatch"),
            ("dtype", name, value.float(), "must be BF16"),
        ):
            with self.subTest(label=label):
                session = self.transaction()
                with self.assertRaisesRegex(ValueError, pattern):
                    session.push(source_name, tensor, timeout=2)
                with self.assertRaisesRegex(ValueError, pattern):
                    session.finish(timeout=2)

    def test_duplicate_source_fails_before_commit(self):
        session = self.transaction()
        session.push(*self.sources[0], timeout=2)
        with self.assertRaisesRegex(ValueError, "duplicate HF"):
            session.push(*self.sources[0], timeout=2)
        with self.assertRaisesRegex(ValueError, "duplicate HF"):
            session.finish(timeout=2)

    def test_missing_internal_target_cannot_commit(self):
        session = self.transaction()
        # A target that the converter never emits must prevent publication.
        session.targets[(None, "unwritten")] = torch.zeros(1, dtype=torch.bfloat16)
        for source in self.sources:
            session.push(*source, timeout=2)
        with self.assertRaisesRegex(ValueError, "target coverage"):
            session.finish(timeout=2)


if __name__ == "__main__":
    unittest.main()

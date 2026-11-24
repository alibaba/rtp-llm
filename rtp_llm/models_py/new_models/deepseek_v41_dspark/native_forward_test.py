"""Cold V4.1 draft forward contract with the real PyModelInputs/Outputs binding."""

import tempfile
import unittest

import torch

from rtp_llm.models_py.new_models.deepseek_v41_dspark import service_test as fixtures
from rtp_llm.ops.compute_ops import PyModelInputs


class V41DraftNativeForwardTest(unittest.TestCase):
    def test_actual_graph_metadata_and_cold_native_forward_interfaces(self):
        for commit in (False, True):
            with self.subTest(
                commit=commit
            ), tempfile.TemporaryDirectory() as directory:
                model, _, _ = fixtures.V41DraftServiceTest().staged(
                    directory, commit=commit
                )
                # Exercise the actual ABI-presence check without allocating a GPU.
                model.validate_runtime_device(torch.device("cuda"))
                self.assertEqual(model.cuda_graph_input_hidden_size(), 192)
                metadata = model.prepare_fmha_impl(None, is_cuda_graph=True)
                self.assertEqual(metadata.sched_meta_cache, {})
                self.assertTrue(metadata.support_cuda_graph())
                inputs = PyModelInputs()
                inputs.input_ids = torch.ones(6, dtype=torch.int32)
                # Main's documented no-KV warmup only; no attention/projection
                # backend or CUDA Graph execution is replaced/claimed here.
                result = (
                    model.forward_commit(inputs)
                    if commit
                    else model.forward_propose(inputs)
                )
                self.assertEqual(
                    tuple(result.hidden_states.shape), (0 if commit else 6, 64)
                )
                with self.assertRaisesRegex(RuntimeError, "fixed forward_propose"):
                    model.forward(inputs)


if __name__ == "__main__":
    unittest.main()

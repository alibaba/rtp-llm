"""CUDA graph integration; the Bazel target reserves a real GPU."""

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import torch
from safetensors.torch import save_file

from rtp_llm.models_py.model_loader import NewModelLoader
from rtp_llm.models_py.new_models.qwen3_dspark.checkpoint_test import checkpoint, config, load_config


class DSparkGraphTest(unittest.TestCase):
    def test_capture_replay_live_context_projection(self):
        # This is a real capture of the draft feature -> per-layer K/V
        # projection, not a substitute for graph replay on a serving engine.
        with tempfile.TemporaryDirectory() as root:
            save_file(checkpoint(), str(Path(root) / "model.safetensors"))
            load = replace(load_config(), device="cuda", compute_dtype=torch.bfloat16)
            owner = NewModelLoader(config(), load, model_path=root).load()
        self.assertEqual(owner.cuda_graph_input_hidden_size(), 8)
        features = torch.zeros((3, 8), device="cuda", dtype=torch.bfloat16)

        def context_rows():
            hidden = owner.hidden_norm(owner.combine_hidden_states(features))
            return torch.stack(
                [torch.stack((k, v)) for k, v in owner._context_kv_rows(hidden, 1, 2)]
            )

        warm_stream = torch.cuda.Stream()
        warm_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(warm_stream):
            for _ in range(3):
                context_rows()
        torch.cuda.current_stream().wait_stream(warm_stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = context_rows()
        for x in (1, 3):
            features.copy_(torch.arange(24, device=features.device).reshape(3, 8) * x)
            graph.replay()
            torch.testing.assert_close(out, context_rows())


if __name__ == "__main__":
    unittest.main()

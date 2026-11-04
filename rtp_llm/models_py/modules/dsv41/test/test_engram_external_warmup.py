"""CPU regression for external Engram warmup; actual orchestration, no CUDA.

Install beside the other dsv41 tests. The test deliberately has no skip when
torch is absent: its CPU Bazel target must supply the real torch/numpy wheels.
GPU hashing, lookup and gated-residual execution are replaced at their kernel
boundaries; cache-key creation, metadata access and the warmup loop are real.
"""

from contextlib import ExitStack
import os
import sys
from types import SimpleNamespace
import unittest
from unittest import mock

import torch

from rtp_llm.models_py.modules.dsv41 import dsv41_kernel_jit_warmup as warmup
from rtp_llm.models_py.modules.dsv41 import engram


class ExternalEmbeddingForWarmup(engram.HostEngramEmbedding):
    """Keep the production metadata property; replace only the lookup kernel."""

    def __init__(self, table_shape):
        # Match from_storage's public/metadata state without a CUDA allocation.
        # There is deliberately no full host tensor, even for these huge shapes.
        self.weight = self.scales = None
        self._uses_external_storage = True
        self._table_shape = table_shape
        self._num_sms = 2
        self._pinned = None
        self.lookups = []

    def __call__(self, indices, device):
        if indices.device.type != "cpu" or not indices.is_contiguous():
            raise AssertionError("test kernel boundary expects contiguous CPU IDs")
        self.lookups.append((indices.clone(), str(device)))


class LocalEmbeddingForWarmup(ExternalEmbeddingForWarmup):
    """Same kernel boundary with real small CPU tensors and local capabilities."""

    def __init__(self, table_shape):
        engram.HostEngramEmbedding.__init__(
            self, torch.empty(table_shape, dtype=torch.uint8),
            torch.empty((table_shape[0], table_shape[1] // 32), dtype=torch.uint8),
        )
        self._num_sms = 2
        self.lookups = []


class ExternalEngramWarmupTest(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(mock.patch.dict(os.environ, {
            "WARM_UP": "1", "MODEL_WARM_UP": "1",
            "DSV41_CPU_ENGRAM_LOOKUP": "1", "DSV41_ASYNC_ENGRAM_LOOKUP": "0",
        }))
        self.stack.enter_context(mock.patch.object(warmup, "_ENGRAM_WARMED", set()))
        for name in ("_lazy_init", "init", "synchronize", "current_device"):
            self.stack.enter_context(mock.patch.object(
                torch.cuda, name, side_effect=AssertionError("GPU call in CPU regression")
            ))
        # This CPU test exercises orchestration with CUDA placement replaced at
        # tensor/kernel boundaries. Hidden GPUs must not skip that orchestration
        # at the hardware-availability guard before the real lookup loop runs.
        self.stack.enter_context(mock.patch.object(
            warmup.common, "_is_cuda_device",
            side_effect=lambda device: torch.device(device).type == "cuda",
        ))
        self.stack.enter_context(mock.patch.object(warmup.common, "_assert_not_capturing"))
        self.sync = self.stack.enter_context(mock.patch.object(warmup.common, "_sync_cuda"))
        self.stack.enter_context(mock.patch.object(
            warmup.common, "_run_triton_warmup_launch_with_retry",
            side_effect=lambda label, description, launch, **kwargs: launch(),
        ))
        self.gate = self.stack.enter_context(mock.patch.object(engram, "gated_engram_residual"))
        self.cpu_runtime = self.stack.enter_context(mock.patch.object(
            engram, "EngramCPULookupRuntime",
            side_effect=AssertionError("CPU runtime must not use external host views"),
        ))

        # Only CUDA placement is changed. Real torch tensors exercise slicing,
        # contiguity and masking in the production loop. Large allocations fail.
        self.allocation_numels = []
        for name in ("zeros", "ones"):
            factory = getattr(torch, name)

            def cpu_factory(*args, _factory=factory, **kwargs):
                kwargs.pop("device", None)
                size = args[0] if isinstance(args[0], (tuple, list)) else args
                numel = 1
                for value in size:
                    numel *= value
                if numel > 65536:
                    raise AssertionError("warmup attempted a full-table allocation")
                self.allocation_numels.append(numel)
                return _factory(*args, **kwargs)

            self.stack.enter_context(mock.patch.object(torch, name, side_effect=cpu_factory))

        self.embeddings = [
            ExternalEmbeddingForWarmup((1_000_000_000, 32)),
            ExternalEmbeddingForWarmup((2_000_000_000, 32)),
        ]
        self.hash_outputs = []

        def hash_windows(windows, *, dead_mask):
            self.assertEqual(windows.shape, dead_mask.shape)
            self.assertEqual(dead_mask.dtype, torch.bool)
            hashes = torch.arange(windows.shape[0] * 2 * 4).reshape(windows.shape[0], 2, 4)
            self.hash_outputs.append(hashes)
            return hashes

        self.state = SimpleNamespace(
            layout=SimpleNamespace(layer_ids=(1, 14), n_hash_cols=4, head_dim=32, max_ngram_size=3),
            hash_token_windows=mock.Mock(side_effect=hash_windows),
        )
        self.active_layers = [SimpleNamespace(engram=SimpleNamespace(
            embed_tokens=embedding, q_weight=torch.ones((4, 8)),
            k_weight=torch.ones((4, 8)), eps=1e-20, layer_hash_index=index,
            prepare_lookup_prefetch=mock.Mock(), _lookup_stream=None,
        )) for index, embedding in enumerate(self.embeddings)]
        layers = [SimpleNamespace(engram=None) for _ in range(15)]
        for layer_id, layer in zip(self.state.layout.layer_ids, self.active_layers):
            layers[layer_id] = layer
        self.model = SimpleNamespace(
            _engram_hash_state=self.state, v4=SimpleNamespace(layers=layers),
            _engram_layers=((1, 0), (14, 1)), _engram_cpu_runtime=None,
            _is_decode_role=False,
        )

    def run_warmup(self):
        warmup.warmup_v41_engram_jit(self.model, max_m=3, device="cuda:0")

    def test_external_none_weight_runs_real_loop_without_materializing_table(self):
        self.run_warmup()
        self.assertEqual(self.state.hash_token_windows.call_count, 3)
        for layer_index, embedding in enumerate(self.embeddings):
            self.assertIsNone(embedding.weight)
            self.assertIsNone(embedding.scales)
            self.assertEqual(len(embedding.lookups), 3)
            for (indices, device), hashes in zip(embedding.lookups, self.hash_outputs):
                torch.testing.assert_close(indices, hashes[:, layer_index])
                self.assertEqual(device, "cuda:0")
        self.assertEqual(self.gate.call_count, 4)  # Both masks, both layers at row=1.
        self.sync.assert_called_once_with("cuda:0")
        self.assertTrue(self.allocation_numels)
        self.assertLess(max(self.allocation_numels), 65536)
        self.assertEqual(len(warmup._ENGRAM_WARMED), 1)
        self.cpu_runtime.assert_not_called()
        for layer in self.active_layers:
            layer.engram.prepare_lookup_prefetch.assert_not_called()

    def test_cache_uses_each_external_table_shape(self):
        self.run_warmup()
        self.run_warmup()
        self.assertEqual(self.state.hash_token_windows.call_count, 3)
        # Simulate another complete generation with a different second table.
        # Identical q/k shapes alone must not make this warmup key a cache hit.
        old_embedding = self.embeddings[1]
        replacement = ExternalEmbeddingForWarmup((2_000_000_001, 32))
        self.embeddings[1] = replacement
        self.model.v4.layers[14].engram.embed_tokens = replacement
        self.run_warmup()
        self.assertEqual(self.state.hash_token_windows.call_count, 6)
        self.assertEqual([len(e.lookups) for e in self.embeddings], [6, 3])
        self.assertEqual(len(old_embedding.lookups), 3)
        self.assertEqual(len(warmup._ENGRAM_WARMED), 2)
        self.assertEqual(self.sync.call_count, 3)

    def test_external_async_flag_skips_special_kernel_and_stream_on_cache_hit(self):
        with mock.patch.dict(os.environ, {
            "DSV41_CPU_ENGRAM_LOOKUP": "0", "DSV41_ASYNC_ENGRAM_LOOKUP": "1",
        }), mock.patch.object(
            ExternalEmbeddingForWarmup, "warmup_prefetch",
            side_effect=AssertionError("external prefetch warmup must not run"),
        ) as prefetch:
            self.run_warmup()
            self.run_warmup()
        self.assertEqual(self.state.hash_token_windows.call_count, 3)
        prefetch.assert_not_called()
        self.cpu_runtime.assert_not_called()
        for layer in self.active_layers:
            layer.engram.prepare_lookup_prefetch.assert_not_called()

    def test_cache_distinguishes_same_geometry_backend_capabilities(self):
        for layer in self.active_layers:
            layer.engram.embed_tokens=ExternalEmbeddingForWarmup((3,32))
        with mock.patch.dict(os.environ, {
            "DSV41_CPU_ENGRAM_LOOKUP": "0", "DSV41_ASYNC_ENGRAM_LOOKUP": "0",
        }):
            self.run_warmup()
            for layer in self.active_layers:
                layer.engram.embed_tokens=LocalEmbeddingForWarmup((3,32))
            self.run_warmup()
            self.run_warmup()
        self.assertEqual(self.state.hash_token_windows.call_count,6)
        self.assertEqual(len(warmup._ENGRAM_WARMED),2)
        for layer in self.active_layers:
            self.assertEqual(layer.engram.prepare_lookup_prefetch.call_count,2)
        self.cpu_runtime.assert_not_called()

    def test_mixed_warmup_prefetches_only_capable_layer(self):
        local=LocalEmbeddingForWarmup((3,32))
        local.warmup_prefetch=mock.Mock()
        self.model.v4.layers[1].engram.embed_tokens=local
        name="rtp_llm.models_py.modules.dsv41._engram_triton"
        kernel=SimpleNamespace(PREFETCH_LOOKUP_CONFIG=(4,4),PREFETCH_LOOKUP_CARVEOUT=100)
        with mock.patch.dict(os.environ, {
            "DSV41_CPU_ENGRAM_LOOKUP": "0", "DSV41_ASYNC_ENGRAM_LOOKUP": "1",
        }), mock.patch.dict(sys.modules,{name:kernel}):
            self.run_warmup()
            self.run_warmup()
        self.assertEqual(local.warmup_prefetch.call_count,3)
        self.assertEqual(self.model.v4.layers[1].engram.prepare_lookup_prefetch.call_count,2)
        self.model.v4.layers[14].engram.prepare_lookup_prefetch.assert_not_called()
        self.assertEqual(len(self.embeddings[1].lookups),3)
        self.cpu_runtime.assert_not_called()


if __name__ == "__main__":
    unittest.main()

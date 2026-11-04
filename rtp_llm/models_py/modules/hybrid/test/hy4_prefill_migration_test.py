import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from rtp_llm.models_py.model_desc.hy_v4_model import Hy4LayerOutput, Hy4Model
from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl import (
    flashmla_sparse_cp_impl as cp,
)


class _PrefillCache:
    def __init__(self):
        self.active = False
        self.events = []

    def begin_forward(self, fmha):
        self.active = True
        self.events.append("begin")

    def before_layer(self, layer):
        self.events.append(layer)

    def end_forward(self, failed=False):
        self.events.append(("end", failed))
        self.active = False


class _Layer(nn.Module):
    def __init__(self, cache, layer, fail=False):
        super().__init__()
        self.cache = cache
        self.layer = layer
        self.fail = fail

    def forward(self, channels, fmha, kv_cache=None, prev_topk_indices=None):
        if self.cache.events[-1] != self.layer:
            raise AssertionError("layer ran before its cache was prepared")
        if fmha.layered_prefill_cache is not self.cache:
            raise AssertionError("attention did not receive the cache owner")
        if self.fail:
            raise RuntimeError("layer failed")
        return Hy4LayerOutput(channels + 1, None)


class _ChannelMean(nn.Module):
    def forward(self, channels):
        return channels.mean(dim=1)


class Hy4PrefillMigrationTest(unittest.TestCase):
    def make_model(self, fail=False):
        model = Hy4Model.__new__(Hy4Model)
        nn.Module.__init__(model)
        model.hc_mult = 4
        model.hidden_size = 4
        model.layer_num = 2
        model.kv_cache = None
        model.pinned_mla_groups = {}
        model.layered_prefill_cache = _PrefillCache()
        model.embed_tokens = nn.Embedding.from_pretrained(
            torch.arange(12.0).reshape(3, 4)
        )
        model.layers = nn.ModuleList(
            [
                _Layer(model.layered_prefill_cache, 0),
                _Layer(model.layered_prefill_cache, 1, fail),
            ]
        )
        model.ihc_head = _ChannelMean()
        model.norm = nn.Identity()
        inputs = SimpleNamespace(
            input_ids=torch.tensor([1, 2]),
            attention_inputs=SimpleNamespace(
                kv_cache_kernel_block_id_device_by_group=None
            ),
        )
        return model, inputs, SimpleNamespace(fmha_params=None)

    def test_hy4_prepares_each_layer_and_releases_cache_after_forward(self):
        model, inputs, fmha = self.make_model()

        output = model(inputs, fmha)

        torch.testing.assert_close(
            output.hidden_states, model.embed_tokens(inputs.input_ids) + 2
        )
        self.assertEqual(
            model.layered_prefill_cache.events, ["begin", 0, 1, ("end", False)]
        )
        self.assertFalse(model.layered_prefill_cache.active)

    def test_hy4_releases_cache_after_layer_failure(self):
        model, inputs, fmha = self.make_model(fail=True)

        with self.assertRaisesRegex(RuntimeError, "layer failed"):
            model(inputs, fmha)

        self.assertEqual(
            model.layered_prefill_cache.events, ["begin", 0, 1, ("end", True)]
        )
        self.assertFalse(model.layered_prefill_cache.active)

    def test_cp_gather_preserves_sink_and_removes_padding(self):
        op = cp.SparseMlaFp8CPOp.__new__(cp.SparseMlaFp8CPOp)
        op._gather = SimpleNamespace()
        op.num_heads = op.bf16_num_heads = 64
        op.kv_lora_rank = 4
        op.scale = 1.0
        op.precomputed_kv_offsets = torch.zeros(2, dtype=torch.int32)
        query = torch.arange(24.0).reshape(2, 3, 4)
        sink = torch.tensor([0.5, -1.0, 2.0])
        topk = torch.zeros(2, 2, dtype=torch.int32)
        history = torch.zeros(2, 4)
        seen = []

        def attend(q, kv, indices, scale, d_v, attn_sink):
            seen.append((q.clone(), attn_sink.clone()))
            return q + attn_sink.reshape(1, -1, 1), None, None

        with patch.dict("os.environ", {"RTP_CP_FUSED_TOPK_OFFSETS": "0"}), patch.object(
            cp, "flash_mla_sparse_fwd", side_effect=attend, create=True
        ):
            output = op._attend_gather(
                query, None, topk, prepared_kv=history, attn_sink=sink
            )

        torch.testing.assert_close(output, query + sink.reshape(1, -1, 1))
        self.assertEqual(seen[0][0].shape, (2, 64, 4))
        torch.testing.assert_close(seen[0][1][:3], sink)
        self.assertTrue(torch.isneginf(seen[0][1][3:]).all())


if __name__ == "__main__":
    unittest.main()

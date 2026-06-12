import types
import unittest
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from rtp_llm.models_py.model_desc.module_base import GptModelBase


class _FakeEmbedding(nn.Module):
    def __init__(self, vocab_size=100, hidden_dim=8):
        super().__init__()
        self.weight = torch.randn(vocab_size, hidden_dim)

    def forward(self, input_ids):
        return F.embedding(input_ids, self.weight)


class _SimpleModel(GptModelBase):
    def __init__(self):
        nn.Module.__init__(self)
        self.embed_tokens = _FakeEmbedding()

    def forward(self, inputs: Any, fmha_impl=None):
        input_ids = inputs.input_ids
        inputs_embeds = self.get_inputs_embeds(input_ids, inputs)
        return inputs_embeds


def _make_inputs(input_ids, input_embeddings=None, input_embeddings_locs=None):
    ns = types.SimpleNamespace(
        input_ids=input_ids,
        input_embeddings=input_embeddings,
        input_embeddings_locs=input_embeddings_locs,
    )
    return ns


class GetInputsEmbedsTest(unittest.TestCase):

    def test_no_embeddings_returns_normal_lookup(self):
        model = _SimpleModel()
        input_ids = torch.tensor([1, 2, 3])
        inputs = _make_inputs(input_ids)
        result = model.get_inputs_embeds(input_ids, inputs)
        expected = F.embedding(input_ids, model.embed_tokens.weight)
        self.assertTrue(torch.equal(result, expected))

    def test_none_embeddings_returns_normal_lookup(self):
        model = _SimpleModel()
        input_ids = torch.tensor([1, 2, 3])
        inputs = _make_inputs(input_ids, input_embeddings=None)
        result = model.get_inputs_embeds(input_ids, inputs)
        expected = F.embedding(input_ids, model.embed_tokens.weight)
        self.assertTrue(torch.equal(result, expected))

    def test_empty_embeddings_returns_normal_lookup(self):
        model = _SimpleModel()
        input_ids = torch.tensor([1, 2, 3])
        inputs = _make_inputs(
            input_ids, input_embeddings=[], input_embeddings_locs=torch.tensor([])
        )
        result = model.get_inputs_embeds(input_ids, inputs)
        expected = F.embedding(input_ids, model.embed_tokens.weight)
        self.assertTrue(torch.equal(result, expected))

    def test_single_embedding_overlay(self):
        model = _SimpleModel()
        hidden_dim = model.embed_tokens.weight.size(1)
        input_ids = torch.tensor([1, 2, 3, 4, 5])
        emb = torch.ones(2, hidden_dim) * 99.0
        locs = torch.tensor([1])
        inputs = _make_inputs(
            input_ids, input_embeddings=[emb], input_embeddings_locs=locs
        )

        result = model.get_inputs_embeds(input_ids, inputs)
        self.assertEqual(list(result.shape), [5, hidden_dim])
        self.assertTrue(torch.equal(result[1:3], emb))
        expected_0 = F.embedding(torch.tensor([1]), model.embed_tokens.weight)
        self.assertTrue(torch.equal(result[0:1], expected_0))

    def test_multiple_embeddings_overlay(self):
        model = _SimpleModel()
        hidden_dim = model.embed_tokens.weight.size(1)
        input_ids = torch.tensor([1, 2, 3, 4, 5, 6])
        emb1 = torch.ones(1, hidden_dim) * 11.0
        emb2 = torch.ones(2, hidden_dim) * 22.0
        locs = torch.tensor([0, 4])
        inputs = _make_inputs(
            input_ids, input_embeddings=[emb1, emb2], input_embeddings_locs=locs
        )

        result = model.get_inputs_embeds(input_ids, inputs)
        self.assertTrue(torch.equal(result[0:1], emb1))
        self.assertTrue(torch.equal(result[4:6], emb2))

    def test_overlay_preserves_non_overlaid_positions(self):
        model = _SimpleModel()
        hidden_dim = model.embed_tokens.weight.size(1)
        input_ids = torch.tensor([10, 20, 30])
        emb = torch.ones(1, hidden_dim) * 55.0
        locs = torch.tensor([1])
        inputs = _make_inputs(
            input_ids, input_embeddings=[emb], input_embeddings_locs=locs
        )

        result = model.get_inputs_embeds(input_ids, inputs)
        expected_0 = F.embedding(torch.tensor([10]), model.embed_tokens.weight)
        expected_2 = F.embedding(torch.tensor([30]), model.embed_tokens.weight)
        self.assertTrue(torch.equal(result[0:1], expected_0))
        self.assertTrue(torch.equal(result[2:3], expected_2))

    def test_forward_uses_get_inputs_embeds(self):
        model = _SimpleModel()
        hidden_dim = model.embed_tokens.weight.size(1)
        input_ids = torch.tensor([1, 2, 3])
        emb = torch.ones(1, hidden_dim) * 77.0
        locs = torch.tensor([0])
        inputs = _make_inputs(
            input_ids, input_embeddings=[emb], input_embeddings_locs=locs
        )

        result = model.forward(inputs)
        self.assertTrue(torch.equal(result[0:1], emb))


if __name__ == "__main__":
    unittest.main()

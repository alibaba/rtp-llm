import unittest
from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F

from rtp_llm.models_py.model_desc.generic_moe_mtp import GenericMoeMTPModel
from rtp_llm.models_py.model_desc.hy_v4_mtp_model import Hy4MtpModel
from rtp_llm.models_py.model_desc.module_base import GptModelBase


class SharedDecoderInputsTest(unittest.TestCase):
    def test_layer_selection_preserves_group_tables_and_layer_caches(self):
        tables = [torch.tensor([[10]]), torch.tensor([[20]])]
        caches = [torch.tensor([layer]) for layer in range(3)]
        model = GptModelBase.__new__(GptModelBase)
        nn.Module.__init__(model)
        model.kv_cache = SimpleNamespace(get_layer_cache=lambda layer: caches[layer])
        inputs = SimpleNamespace(
            kv_cache_kernel_block_id_device_by_group=tables,
            kv_cache_layer_to_group=torch.tensor([1, 0, 1]),
            kv_cache_kernel_block_id_device=tables[0],
        )

        for layer, group in enumerate((1, 0, 1)):
            with self.subTest(layer=layer):
                cache = model.prepare_decoder_layer(inputs, layer)
                self.assertIs(cache, caches[layer])
                self.assertIs(inputs.kv_cache_kernel_block_id_device, tables[group])

    def test_layer_selection_without_kv_cache_keeps_block_mapping(self):
        table = torch.tensor([[17]])
        model = GptModelBase.__new__(GptModelBase)
        nn.Module.__init__(model)
        model.kv_cache = None
        inputs = SimpleNamespace(
            kv_cache_kernel_block_id_device_by_group=[table],
            kv_cache_layer_to_group=None,
            kv_cache_kernel_block_id_device=None,
        )

        self.assertIsNone(model.prepare_decoder_layer(inputs, 0))
        self.assertIs(inputs.kv_cache_kernel_block_id_device, table)

    def test_plain_block_table_is_preserved(self):
        table = torch.tensor([[31]])
        model = GptModelBase.__new__(GptModelBase)
        nn.Module.__init__(model)
        model.kv_cache = None
        inputs = SimpleNamespace(
            kv_cache_kernel_block_id_device_by_group=None,
            kv_cache_kernel_block_id_device=table,
        )

        self.assertIsNone(model.prepare_decoder_layer(inputs, 0))
        self.assertIs(inputs.kv_cache_kernel_block_id_device, table)

    def test_glm5_and_hy4_mtp_frontends_preserve_mask_norm_and_projection(self):
        ids = torch.tensor([1, 2], dtype=torch.int32)
        hidden = torch.arange(1, 9, dtype=torch.float32).reshape(2, 4)
        embedding = torch.arange(1, 13, dtype=torch.float32).reshape(3, 4)
        projection = torch.arange(1, 33, dtype=torch.float32).reshape(4, 8) / 32
        e_weight = torch.tensor([1.0, 0.5, 2.0, 1.5])
        h_weight = torch.tensor([0.5, 1.5, 1.0, 2.0])
        for model_type in (GenericMoeMTPModel, Hy4MtpModel):
            for positions in (torch.tensor([0, 2]), None):
                with self.subTest(model=model_type.__name__, positions=positions):
                    model = model_type.__new__(model_type)
                    nn.Module.__init__(model)
                    model.embed_tokens = nn.Embedding.from_pretrained(embedding)
                    model.pre_fc_norm_embedding = nn.RMSNorm(4, eps=1e-6)
                    model.pre_fc_norm_hidden = nn.RMSNorm(4, eps=1e-6)
                    model.pre_fc_norm_embedding.weight = nn.Parameter(e_weight)
                    model.pre_fc_norm_hidden.weight = nn.Parameter(h_weight)
                    model.fc = nn.Linear(8, 4, bias=False)
                    model.fc.weight = nn.Parameter(projection)
                    model.pinned_mla_groups = {}
                    model._mtp_frontend_native = None
                    inputs = SimpleNamespace(input_ids=ids, input_hiddens=hidden)
                    fmha = SimpleNamespace(
                        fmha_params=SimpleNamespace(positions_d=positions)
                    )
                    embedded = embedding[ids.long()].clone()
                    if positions is not None:
                        embedded[positions == 0] = 0
                    expected = F.linear(
                        torch.cat(
                            (
                                F.rms_norm(embedded, (4,), e_weight, 1e-6),
                                F.rms_norm(hidden, (4,), h_weight, 1e-6),
                            ),
                            dim=-1,
                        ),
                        projection,
                    )

                    actual, residual = model._prepare_mtp_inputs(inputs, fmha)

                    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
                    torch.testing.assert_close(
                        residual, torch.zeros_like(expected), atol=0, rtol=0
                    )


if __name__ == "__main__":
    unittest.main()

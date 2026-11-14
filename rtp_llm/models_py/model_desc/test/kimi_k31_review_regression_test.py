"""Checkpoint and renderer boundaries used by the K3.1 runtime."""

import unittest
from types import SimpleNamespace

import torch

from rtp_llm.model_loader.load_config import LoadConfig
from rtp_llm.models.kimi_k3.kimi_k31_weight import KimiK31Weight
from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.dsa_kv_workspace import (
    DsaWorkspaceOwner,
)
from rtp_llm.openai.renderer_factory_register import (
    _renderer_factory,
    ensure_renderer_registered,
)
from rtp_llm.ops import RoleType
from rtp_llm.utils.model_weight import W


def _manifest(query_cp, role=RoleType.PREFILL):
    manifest = object.__new__(KimiK31Weight)
    manifest.prefill_query_cp = query_cp
    manifest.role_type = role
    manifest.tp_size, manifest.tp_rank = 2, 0
    manifest._head_num = 4
    manifest.nope_head_dim = manifest.v_head_dim = 8
    manifest.rope_head_dim = 0
    manifest.kv_lora_rank = 16
    manifest.model_config = SimpleNamespace(
        attn_config=SimpleNamespace(use_mla=True, q_lora_rank=16),
        indexer_projection_quant_config=None,
    )
    return manifest


def _load_config(rank):
    return LoadConfig.model_construct(
        tp_size=2,
        tp_rank=rank,
        dp_size=1,
        dp_rank=0,
        ep_size=1,
        ep_rank=0,
        ffn_tp_size=2,
        ffn_tp_rank=rank,
        hidden_size=12,
        head_num=4,
        head_num_kv=1,
        size_per_head=8,
        moe_pure_tp_mode=True,
    )


class KimiK31ReviewRegressionTest(unittest.TestCase):
    def test_mla_manifest_builds_for_query_cp_and_regular_roles(self):
        for query_cp, role in (
            (True, RoleType.PREFILL),
            (False, RoleType.PDFUSION),
            (False, RoleType.DECODE),
        ):
            with self.subTest(query_cp=query_cp, role=role):
                manifest = _manifest(query_cp, role)
                names = {weight.name for weight in manifest._mla_weights()}
                self.assertIn(W.attn_o_w, names)
                self.assertIn(W.mla_indexer_qb_w, names)

    def test_bf16_query_cp_keeps_complete_output_projection(self):
        checkpoint = torch.arange(12 * 32, dtype=torch.float32).reshape(12, 32)
        expected = checkpoint.T.contiguous()
        for query_cp in (False, True):
            with self.subTest(query_cp=query_cp):
                manifest = _manifest(query_cp)
                output = next(
                    weight
                    for weight in manifest._mla_weights()
                    if weight.name == W.attn_o_w
                )
                prepared = output.process_fun([checkpoint])
                for rank in range(2):
                    actual = output._split({output.name: prepared}, _load_config(rank))[
                        output.name
                    ]
                    reference = expected if query_cp else expected.chunk(2, dim=0)[rank]
                    torch.testing.assert_close(actual, reference, rtol=0, atol=0)

    def test_query_cp_value_weight_matches_workspace_consumer(self):
        manifest = _manifest(query_cp=True)
        kv_weights = next(
            weight for weight in manifest._mla_weights() if weight.name == W.mla_kv_b_w
        )
        checkpoint = (
            torch.arange(4 * 16 * 16, dtype=torch.float32).reshape(64, 16).bfloat16()
        )
        expected = checkpoint.view(4, 16, 16)[:, 8:, :].permute(0, 2, 1).contiguous()
        prepared = {
            name: source.process_fun([checkpoint])
            for name, source in kv_weights.sub_weights.items()
        }
        for rank in range(2):
            with self.subTest(rank=rank):
                loaded = kv_weights._split(prepared, _load_config(rank))
                for name in (W.mla_kv_b_w, W.mla_kc, W.mla_vc):
                    torch.testing.assert_close(
                        loaded[name], prepared[name], rtol=0, atol=0
                    )
                owner = object.__new__(DsaWorkspaceOwner)
                owner.heads, owner.latent_dim, owner.value_dim = 4, 16, 8
                owner.tp_size, owner.prefill_query_cp = 2, True
                owner.value_weights = {}
                owner.bind_value_weights([(0, loaded)])
                torch.testing.assert_close(
                    owner.value_weights[0], expected, rtol=0, atol=0
                )

    def test_k31_lazy_renderer_registration_selects_k3_protocol(self):
        # Do not import the renderer before asking the lazy registry for it.
        self.assertTrue(ensure_renderer_registered("kimi_k31"))
        self.assertTrue(ensure_renderer_registered("kimi_k3"))
        self.assertIs(_renderer_factory["kimi_k31"], _renderer_factory["kimi_k3"])


if __name__ == "__main__":
    unittest.main()

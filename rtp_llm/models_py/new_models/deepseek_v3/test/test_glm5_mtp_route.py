"""GLM-5 draft through BaseModel.load and both NewLoader backends."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from safetensors.torch import save_file

from rtp_llm.model_factory_register import ModelDict
from rtp_llm.models.glm5 import Glm5Mtp
from rtp_llm.models_py.new_models.deepseek_v3.test.test_nextn_route import (
    _CpuCheckpoint,
    fixture,
    load_service,
)
from rtp_llm.models_py.new_models.glm5.language import (
    Glm5ForCausalLM,
    Glm5MTPForCausalLM,
)
from rtp_llm.models_py.registry import get_model_class


class _Draft(_CpuCheckpoint, Glm5Mtp):
    pass


class Glm5DraftRouteTest(unittest.TestCase):
    def test_mixed_checkpoint_keeps_fp4_nibbles_and_group32_scales(self):
        cfg, data, prefix = fixture(False)
        cfg.update(
            architectures=["GlmMoeDsaMtpForCausalLM"],
            model_type="glm_moe_dsa",
            quantization_config={"expert_dtype": "fp4"},
        )
        for expert in range(4):
            for i, proj in enumerate(("gate_proj", "up_proj", "down_proj")):
                base = prefix + f"mlp.experts.{expert}.{proj}"
                data[base + ".weight"] = torch.full(
                    (32, 16), expert * 3 + i, dtype=torch.int8
                )
                data[base + ".weight_scale"] = torch.full((32, 1), 2.0 ** (expert + i))
        with tempfile.TemporaryDirectory() as root:
            (Path(root) / "config.json").write_text(json.dumps(cfg))
            save_file(data, str(Path(root) / "model.safetensors"))
            for method in ("scratch", "fastsafetensors"):
                with self.subTest(method=method):
                    service = load_service(root, method, cls=_Draft, tag="glm_5_mtp")
                    experts = service.py_model.layers[0].mlp.experts
                    self.assertTrue(experts.checkpoint_load_complete())
                    torch.testing.assert_close(
                        experts.w13[2, :32],
                        data[prefix + "mlp.experts.2.up_proj.weight"],
                    )
                    torch.testing.assert_close(
                        experts.w13[2, 32:],
                        data[prefix + "mlp.experts.2.gate_proj.weight"],
                    )
                    torch.testing.assert_close(
                        experts.w2_scale[2].float(),
                        data[prefix + "mlp.experts.2.down_proj.weight_scale"],
                    )

    def test_service_owner_and_both_vocab_layouts(self):
        self.assertIs(get_model_class("glm_5"), Glm5ForCausalLM)
        tag = ModelDict.get_ft_model_type_by_config(
            {"architectures": ["GlmMoeDsaMtpForCausalLM"]}
        )
        self.assertIs(get_model_class(tag), Glm5MTPForCausalLM)
        for combined in (True, False):
            cfg, data, prefix = fixture(combined)
            cfg.update(
                model_type="glm_moe_dsa",
                index_topk=4,
                index_n_heads=2,
                index_head_dim=128,
                swiglu_limit=5.0,
                architectures=[
                    "GlmMoeDsaForCausalLM" if combined else "GlmMoeDsaMtpForCausalLM"
                ],
            )
            for key, shape in (
                ("wq_b.weight", (256, 32)),
                ("wk.weight", (128, 32)),
                ("k_norm.weight", (128,)),
                ("k_norm.bias", (128,)),
                ("weights_proj.weight", (2, 32)),
            ):
                data[prefix + "self_attn.indexer." + key] = torch.ones(shape)
            embed_src = (
                "model.embed_tokens.weight"
                if combined
                else prefix + "embed_tokens.weight"
            )
            head_src = (
                "lm_head.weight" if combined else prefix + "shared_head.head.weight"
            )
            with tempfile.TemporaryDirectory() as root:
                (Path(root) / "config.json").write_text(json.dumps(cfg))
                save_file(data, str(Path(root) / "model.safetensors"))
                for method in ("scratch", "fastsafetensors"):
                    with self.subTest(combined=combined, method=method):
                        service = load_service(root, method, cls=_Draft, tag=tag)
                        model = service.py_model
                        self.assertEqual(
                            service.model_weights_loader.loaded_method.value, method
                        )
                        self.assertEqual(model._checkpoint_layer, 2 if combined else 0)
                        self.assertEqual(len(model.layers), 1)
                        self.assertIsNotNone(model.layers[0].indexer)
                        torch.testing.assert_close(
                            model.embed_tokens.weight,
                            data[embed_src].to(model.embed_tokens.weight),
                        )
                        torch.testing.assert_close(
                            model.lm_head.weight,
                            data[head_src].to(model.lm_head.weight),
                        )
                        self.assertTrue(
                            model.initialize(SimpleNamespace(kv_cache=None))
                        )
                        self.assertIsNone(model.layers[0].cmp)
                        model.py_hw_kernel_config.enable_glm5_cmp = True
                        self.assertTrue(
                            model.initialize(SimpleNamespace(kv_cache=None))
                        )
                        # Enabling CMP binds the actual operator adapter; BF16
                        # test weights cannot select the FP8 hardware fast path.
                        cmp = model.layers[0].cmp
                        self.assertTrue(cmp.has_indexer)
                        self.assertIsNotNone(cmp._disabled_reason)
                        self.assertIs(
                            cmp.self_attn.o_proj.weight,
                            model.layers[0].self_attn.o_proj.weight,
                        )
                        ids = torch.tensor([1, 2])
                        hidden = (
                            torch.arange(64)
                            .reshape(2, 32)
                            .to(model.embed_tokens.weight)
                        )
                        positions = torch.tensor([0, 7], dtype=torch.int32)
                        default = SimpleNamespace(
                            fmha_params=SimpleNamespace(positions_d=positions)
                        )
                        embed = model.embed_tokens(ids)
                        embed[0].zero_()

                        def norm(x, name):
                            y = x.float()
                            return (
                                y
                                * torch.rsqrt(
                                    y.square().mean(-1, keepdim=True)
                                    + cfg["rms_norm_eps"]
                                )
                                * data[prefix + name]
                            ).to(x)

                        expected = torch.nn.functional.linear(
                            torch.cat(
                                [
                                    norm(embed, "enorm.weight"),
                                    norm(hidden, "hnorm.weight"),
                                ],
                                dim=1,
                            ),
                            data[prefix + "eh_proj.weight"].to(hidden),
                        )
                        result = SimpleNamespace(
                            hidden_states=expected, residual=torch.zeros_like(expected)
                        )
                        with patch.object(
                            model, "_forward_decoder_layers", return_value=result
                        ) as run:
                            model(
                                SimpleNamespace(input_ids=ids, input_hiddens=hidden),
                                {"default": default},
                            )
                        torch.testing.assert_close(run.call_args.args[0], expected)
                        clone = model.clone_for_cuda_graph()
                        self.assertIsNot(clone.layers[0], model.layers[0])
                        self.assertIs(
                            clone.layers[0].self_attn, model.layers[0].self_attn
                        )
                        self.assertTrue(
                            clone.initialize(SimpleNamespace(kv_cache=None))
                        )
                        self.assertIs(
                            clone._mtp_shared_topk_indices,
                            model._mtp_shared_topk_indices,
                        )
                        seed = torch.tensor([[1, 2, 3, -1]], dtype=torch.int32)
                        clone.load_mtp_indexer_topk(seed)
                        torch.testing.assert_close(
                            model.snapshot_mtp_indexer_topk(1), seed
                        )
                        reused = model.layers[0].self_attn._run_sparse_indexer(
                            None,
                            None,
                            None,
                            None,
                            SimpleNamespace(is_sparse=lambda: True),
                            seed,
                            force_reuse_topk_indices=True,
                        )
                        self.assertIs(reused, seed)
                        experts = model.layers[0].mlp.experts
                        adapter = SimpleNamespace(
                            moe_strategy="mega_moe", physical_expert_num=4
                        )
                        experts._configure_factory_adapter(adapter)
                        self.assertEqual(adapter.mega_moe_policy.activation_clamp, 5.0)
                        self.assertEqual(adapter.max_generate_batch_size, 1)


if __name__ == "__main__":
    unittest.main()

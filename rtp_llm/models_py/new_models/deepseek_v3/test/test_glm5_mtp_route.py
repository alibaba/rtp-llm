"""GLM-5 draft through BaseModel.load and both NewLoader backends."""

import json
import os
import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from safetensors.torch import save_file

from rtp_llm.model_factory_register import ModelDict
from rtp_llm.models.glm5 import Glm5, Glm5Mtp
from rtp_llm.models_py.new_models.deepseek_v3.test.test_nextn_route import (
    _CpuCheckpoint,
    fixture,
    load_service,
)
from rtp_llm.models_py.new_models.glm5.language import (
    Glm5ForCausalLM,
    Glm5MTPForCausalLM,
    _Glm5Mapper,
)
from rtp_llm.models_py.read_spec import make_sharded_read_spec, unwrap_checkpoint_tensor
from rtp_llm.models_py.registry import get_model_class
from rtp_llm.models_py.weight_mapper import WeightsMapper


class _Draft(_CpuCheckpoint, Glm5Mtp):
    pass


class _Score(_CpuCheckpoint, Glm5):
    pass


_READERS = (
    ("scratch",)
    if os.environ.get("NEWLOADER_TEST_NATIVE_FST") == "0"
    else (
        "scratch",
        "fastsafetensors",
    )
)


class _ReachedDecoder(Exception):
    """The driverless test stops after the actual embedding/norm/projection."""


class _DraftServiceFixture:
    def check_service_owner_and_both_vocab_layouts(self, *, native_output):
        self.assertIs(get_model_class("glm_5"), Glm5ForCausalLM)
        tag = ModelDict.get_ft_model_type_by_config(
            {"architectures": ["GlmMoeDsaMtpForCausalLM"]}
        )
        self.assertIs(get_model_class(tag), Glm5MTPForCausalLM)
        for combined, draft_layer in ((True, 2), (False, 0), (True, 78)):
            cfg, data, prefix = fixture(combined)
            if draft_layer == 78:
                # Cropped score, original draft layer number retained on disk.
                original = prefix
                prefix = f"model.layers.{draft_layer}."
                data = {
                    key.replace(original, prefix, 1) if key.startswith(original) else key: value
                    for key, value in data.items()
                }
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
                if draft_layer == 78:
                    for i in range(cfg["num_hidden_layers"]):
                        data[f"model.layers.{i}.self_attn.indexer." + key] = torch.ones(shape)
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
                if draft_layer == 78:
                    (Path(root) / "model.safetensors.index.json").write_text(
                        json.dumps({"weight_map": dict.fromkeys(data, "model.safetensors")})
                    )
                for method in _READERS:
                    with self.subTest(combined=combined, draft_layer=draft_layer, method=method):
                        service = load_service(root, method, cls=_Draft, tag=tag)
                        model = service.py_model
                        self.assertEqual(
                            service.model_weights_loader.loaded_method.value, method
                        )
                        self.assertEqual(model._checkpoint_layer, draft_layer)
                        if draft_layer == 78:
                            target = load_service(root, method, cls=_Score, tag="glm_5").py_model
                            owns = target.checkpoint_weight_name_filter()
                            self.assertFalse(owns(prefix + "self_attn.q_a_proj.weight"))
                            # A stray, undeclared layer must still reach strict dispatch.
                            self.assertTrue(owns("model.layers.79.mlp.bad.weight"))
                            torch.testing.assert_close(
                                target.norm.weight, data["model.norm.weight"].to(target.norm.weight)
                            )
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
                            model,
                            "_forward_decoder_layers",
                            return_value=result,
                            side_effect=None if native_output else _ReachedDecoder,
                        ) as run:
                            with (
                                nullcontext()
                                if native_output
                                else self.assertRaises(_ReachedDecoder)
                            ):
                                output = model(
                                    SimpleNamespace(
                                        input_ids=ids, input_hiddens=hidden
                                    ),
                                    {"default": default},
                                )
                        torch.testing.assert_close(run.call_args.args[0], expected)
                        if native_output:
                            from rtp_llm.ops.compute_ops import PyModelOutputs

                            self.assertIsInstance(output, PyModelOutputs)
                            torch.testing.assert_close(
                                output.hidden_states,
                                norm(expected, "shared_head.norm.weight").to(expected),
                            )
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


class Glm5DraftRouteTest(_DraftServiceFixture, unittest.TestCase):
    def test_dense_scale_mapping_preserves_tp_shard(self):
        mapper = _Glm5Mapper(WeightsMapper(prefix_mapping={"model.": ""}), packed=True)
        name = "model.layers.0.self_attn.q_b_proj.scale"
        spec = make_sharded_read_spec(name, (4, 1), 0, 2, 1)
        source = torch.tensor([[1.0], [2.0], [4.0], [8.0]])
        dtypes = [torch.float32]
        if hasattr(torch, "float8_e8m0fnu"):
            dtypes.append(torch.float8_e8m0fnu)
        for dtype in dtypes:
            with self.subTest(dtype=dtype):
                local = spec.materialize(source.to(dtype))
                key, value = next(mapper.apply([(name, local)]))
                tensor, proof = unwrap_checkpoint_tensor(value)
                self.assertEqual(key, "layers.0.self_attn.q_b_proj.weight_scale_inv")
                self.assertIs(proof, spec)
                torch.testing.assert_close(tensor, source[2:])

    @unittest.skipUnless(hasattr(torch, "float8_e8m0fnu"), "requires UE8M0 torch dtype")
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
            for method in _READERS:
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
        self.check_service_owner_and_both_vocab_layouts(native_output=False)


if __name__ == "__main__":
    unittest.main()

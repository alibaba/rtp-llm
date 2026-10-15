import os
import unittest
from types import SimpleNamespace

from rtp_llm.model_factory_register import ModelDict, ensure_model_registered
from rtp_llm.models.xing4_0 import Xing4, Xing4Mtp, Xing4Weight, Xing4MtpWeight

TESTDATA_DIR = os.path.join(
    os.getcwd(), "rtp_llm/test/model_test/fake_test/testdata/xing4_0"
)


def _fake_parallelism():
    ffn_disagg = SimpleNamespace(
        enable_ffn_disaggregate=False, is_ffn_service=lambda: False
    )
    return SimpleNamespace(
        role_type=None,
        get_attn_tp_size=lambda: 1,
        get_attn_tp_rank=lambda: 0,
        ep_size=1,
        ep_rank=0,
        dp_size=1,
        dp_rank=0,
        world_size=1,
        local_world_size=1,
        get_ffn_tp_rank=lambda: 0,
        get_ffn_tp_size=lambda: 1,
        ffn_disaggregate_config=ffn_disagg,
        tp_size=1,
        tp_rank=0,
    )


def _fake_hw_kernel():
    return SimpleNamespace(use_swizzleA=False)


class Xing4ConfigTest(unittest.TestCase):
    """Factory-order config tests for the Xing4.0 registration.

    Covers the module-recipes config lifecycle contract: lazy registration
    resolves without importing every model, the architecture mapping infers
    the model type, and _create_config maps the real HF config fields into
    ModelConfig with non-default values.
    """

    def test_architecture_mapping(self):
        model_type = ModelDict.get_ft_model_type_by_config(
            {"architectures": ["Xing4_0ForCausalLM"]}
        )
        self.assertEqual(model_type, "xing4_0")
        mtp_type = ModelDict.get_ft_model_type_by_config(
            {"architectures": ["Xing4_0ForCausalLMNextN"]}
        )
        self.assertEqual(mtp_type, "xing4_0_mtp")

    def test_lazy_import_resolves_classes(self):
        from rtp_llm.model_factory import ModelFactory

        self.assertTrue(ensure_model_registered("xing4_0"))
        self.assertTrue(ensure_model_registered("xing4_0_mtp"))
        self.assertIs(ModelFactory.get_model_cls("xing4_0"), Xing4)
        self.assertIs(ModelFactory.get_model_cls("xing4_0_mtp"), Xing4Mtp)

    def test_create_config_from_real_checkpoint_config(self):
        config = Xing4._create_config(TESTDATA_DIR)

        # mHC hyper-connection fields
        self.assertEqual(config.hc_mult, 4)
        self.assertEqual(config.hc_sinkhorn_iters, 20)
        self.assertAlmostEqual(config.hc_eps, 1e-6)
        self.assertAlmostEqual(config.mhc_h_res_clamp_min, -30.0)
        self.assertAlmostEqual(config.mhc_h_res_clamp_max, 30.0)

        # MLA geometry
        self.assertTrue(config.attn_config.use_mla)
        self.assertEqual(config.attn_config.head_num, 32)
        self.assertEqual(config.attn_config.kv_head_num, 32)
        self.assertEqual(config.attn_config.q_lora_rank, 768)
        self.assertEqual(config.attn_config.kv_lora_rank, 512)
        self.assertEqual(config.attn_config.nope_head_dim, 128)
        self.assertEqual(config.attn_config.rope_head_dim, 64)
        self.assertEqual(config.attn_config.v_head_dim, 128)
        self.assertEqual(config.attn_config.size_per_head, 192)
        # Xing4.0 ships rope_interleave=True (default, not overridden in
        # config.json): interleaved pairs, i.e. NOT neox style.
        self.assertFalse(config.attn_config.rope_config.is_neox_style)
        self.assertEqual(config.attn_config.rope_config.dim, 64)
        self.assertEqual(config.attn_config.rope_config.base, 10000)
        self.assertEqual(config.attn_config.rope_config.scale, 64)
        self.assertEqual(config.attn_config.rope_config.max_pos, 4096)

        # layer / size geometry
        self.assertEqual(config.num_layers, 40)
        self.assertEqual(config.hidden_size, 3584)
        self.assertEqual(config.vocab_size, 131072)
        self.assertEqual(config.dense_inter_size, 9216)
        self.assertEqual(config.inter_size, 1024)  # n_shared * moe_inter
        self.assertEqual(config.moe_inter_size, 1024)
        self.assertEqual(config.tie_word_embeddings, False)
        self.assertEqual(config.config_dtype, "bfloat16")

        # lm_head stays in checkpoint dtype (bf16): the framework default
        # enable_fp32_lm_head=True would force fp32 storage, doubling the
        # [131072,3584] lm_head bytes and missing the C++
        # cublas_gemm_bf16_bf16_fp32 fast path.
        self.assertFalse(config.enable_fp32_lm_head)

        # MoE routing
        self.assertEqual(config.moe_k, 4)
        self.assertEqual(config.expert_num, 64)
        self.assertEqual(config.scoring_func, 1)  # sigmoid
        self.assertAlmostEqual(config.routed_scaling_factor, 2.0)
        self.assertTrue(config.has_moe_norm)
        self.assertEqual(config.moe_style, 2)
        self.assertEqual(config.n_shared_experts, 1)
        self.assertEqual(config.moe_n_group, 1)
        self.assertEqual(config.moe_topk_group, 1)
        self.assertEqual(
            config.moe_layer_index, [i for i in range(40) if i >= 2]
        )

        self.assertEqual(Xing4.get_weight_cls(), Xing4Weight)

    def test_mtp_create_config(self):
        config = Xing4Mtp._create_config(TESTDATA_DIR)
        self.assertTrue(config.is_mtp)
        # Xing4.0's eh_proj is e-first ([enorm(embed), hnorm(hidden)]); the
        # reversed V3 order was empirically rejected by the offline draft
        # order diagnosis (zero draft acceptance with True).
        self.assertFalse(config.reverse_e_h_norm)
        # Xing4.0 keeps the nextn layer inside the main checkpoint
        # (model.layers.40.*); the draft model sees exactly one layer.
        self.assertEqual(config.num_layers, 1)
        self.assertEqual(config.moe_layer_index, [0])
        self.assertEqual(config.hc_mult, 1)
        self.assertEqual(config.hc_sinkhorn_iters, 0)
        # shared_head.head.weight feeds the same C++ lm_head exit; keep the
        # draft lm_head bf16 like the target model.
        self.assertFalse(config.enable_fp32_lm_head)
        self.assertEqual(Xing4Mtp.get_weight_cls(), Xing4MtpWeight)

    def test_lm_head_keeps_checkpoint_bf16_dtype(self):
        """lm_head must load as bf16 so the C++ lm_head mm takes the
        cublas_gemm_bf16_bf16_fp32 fast path (bf16 inputs, fp32 accumulate)
        instead of the fp32 gemv fallback.

        The Xing4.0 checkpoint stores lm_head.weight /
        model.layers.40.shared_head.head.weight as BF16 [131072, 3584] and
        lists lm_head in modules_to_not_convert. Loading chain under test:
        ModelDeployWeightInfo.get_weight_info() -> _fix_fp32_lm_head is
        skipped when enable_fp32_lm_head=False, so the AtomicWeight
        data_type stays None and AtomicWeight._load_raw_tensor casts to
        load_config.compute_dtype (bf16 for this model).
        """
        import torch

        from rtp_llm.model_loader.tensor_source import TensorSource
        from rtp_llm.utils.model_weight import W

        class _Bf16LmHeadSource(TensorSource):
            """Minimal checkpoint source returning a bf16 lm_head shard."""

            def load_tensor(self, name, data_type=torch.float16):
                return [torch.randn(16, 8, dtype=torch.bfloat16)]

            def has_tensor(self, name):
                return True

            def get_database(self):
                return None

        load_config = SimpleNamespace(compute_dtype=torch.bfloat16)

        def _lm_head_descriptor(weight_cls, config):
            weight = weight_cls(
                config, _fake_parallelism(), _fake_hw_kernel(), SimpleNamespace()
            )
            info = weight.get_weight_info()
            matches = [w for w in info.weights if w.name == W.lm_head]
            self.assertEqual(len(matches), 1)
            return matches[0]

        for weight_cls, config_factory in (
            (Xing4Weight, Xing4._create_config),
            (Xing4MtpWeight, Xing4Mtp._create_config),
        ):
            with self.subTest(model=weight_cls.__name__):
                config = config_factory(TESTDATA_DIR)
                # Model-side default: no forced fp32 storage dtype.
                self.assertFalse(config.enable_fp32_lm_head)
                descriptor = _lm_head_descriptor(weight_cls, config)
                self.assertIsNone(descriptor.data_type)

                loaded = descriptor._load_raw_tensor(
                    _Bf16LmHeadSource(), None, "cpu", load_config
                )
                self.assertEqual(loaded[W.lm_head].dtype, torch.bfloat16)

        # Contrast: the framework default (enable_fp32_lm_head=True) is the
        # operative switch — flipping it back re-forces fp32 storage, which is
        # the slow fp32 gemv path this model opts out of.
        config = Xing4._create_config(TESTDATA_DIR)
        config.enable_fp32_lm_head = True
        descriptor = _lm_head_descriptor(Xing4Weight, config)
        self.assertEqual(descriptor.data_type, torch.float32)
        loaded = descriptor._load_raw_tensor(
            _Bf16LmHeadSource(), None, "cpu", load_config
        )
        self.assertEqual(loaded[W.lm_head].dtype, torch.float32)

    def test_layer_weight_info_contains_hc_weights(self):
        config = Xing4._create_config(TESTDATA_DIR)
        from rtp_llm.utils.model_weight import W

        weight = Xing4Weight(
            config, _fake_parallelism(), _fake_hw_kernel(), SimpleNamespace()
        )
        layer_weights = weight._get_hf_layer_weight_info(5)
        names = {module.name for module in layer_weights}
        self.assertIn(W.v4_hc_attn_base, names)
        self.assertIn(W.v4_hc_attn_fn, names)
        self.assertIn(W.v4_hc_attn_scale, names)
        self.assertIn(W.v4_hc_ffn_base, names)
        self.assertIn(W.v4_hc_ffn_fn, names)
        self.assertIn(W.v4_hc_ffn_scale, names)
        # mHC checkpoint key templates use the attn_hc/ffn_hc module names.
        key_templates = set()
        for module in layer_weights:
            for info in getattr(module, "weights", None) or []:
                key_templates.add(info.name)
        for sub in ("base", "fn", "scale"):
            self.assertIn(f"model.layers.{{i}}.attn_hc.hc_{sub}", key_templates)
            self.assertIn(f"model.layers.{{i}}.ffn_hc.hc_{sub}", key_templates)

    def test_mtp_weight_info_remapped_to_nextn_layer(self):
        config = Xing4Mtp._create_config(TESTDATA_DIR)
        weight = Xing4MtpWeight(
            config, _fake_parallelism(), _fake_hw_kernel(), SimpleNamespace()
        )
        # Production resolves these in _process_meta from real checkpoint keys
        # (Xing4.0 uses q_a_lora and ships e_score_correction_bias).
        weight.q_use_lora = True
        weight.has_e_score_correction_bias = True
        info = weight._get_weight_info()

        def _collect(mod, out):
            for ckpt_info in getattr(mod, "weights", None) or []:
                out.add(ckpt_info.name)
            subs = getattr(mod, "sub_weights", None)
            if subs:
                for sub in (subs.values() if isinstance(subs, dict) else subs):
                    _collect(sub, out)

        templates = set()
        for layer_modules in info.layer_weights:
            for module in layer_modules:
                _collect(module, templates)
        for module in info.weights:
            _collect(module, templates)
        self.assertIn("model.layers.40.embed_tokens.weight", templates)
        self.assertIn("model.layers.40.shared_head.head.weight", templates)
        self.assertIn("model.layers.40.eh_proj.weight", templates)
        self.assertIn("model.layers.40.enorm.weight", templates)
        self.assertIn("model.layers.40.hnorm.weight", templates)
        self.assertIn("model.layers.40.self_attn.kv_a_proj_with_mqa.weight", templates)
        # CompositeWeight children keep sub_weights as dicts; the MoE/FFN
        # descriptors must be remapped too (a dict-unaware walker silently
        # skipped them and the draft then looked up dense-layer 0 weights).
        self.assertIn(
            "model.layers.40.mlp.shared_experts.down_proj.weight", templates
        )
        self.assertIn(
            "model.layers.40.mlp.experts.{expert_id}.down_proj.weight", templates
        )
        self.assertIn("model.layers.40.mlp.gate.weight", templates)
        # no backbone layer templates survive the rewrite
        self.assertFalse(
            any("model.layers.{i}." in name for name in templates),
            f"unrewritten template remains: {sorted(templates)[:5]}",
        )


if __name__ == "__main__":
    unittest.main()

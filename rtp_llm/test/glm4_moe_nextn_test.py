import json
import os
import tempfile
import unittest

from rtp_llm.models.glm4_moe import (
    Glm4Moe,
    Glm4MoeNextN,
    Glm4MoeNextNWeight,
    _retarget_layer,
    find_nextn_layer_id,
    resolve_config_dtype,
)


def _glm47_config_json(num_hidden_layers: int = 92) -> dict:
    """The fields Glm4Moe._from_config_json reads, with GLM-4.7's real values."""
    return {
        "architectures": ["Glm4MoeForCausalLM"],
        "model_type": "glm4_moe",
        "hidden_size": 5120,
        "intermediate_size": 12288,
        "num_attention_heads": 96,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "num_hidden_layers": num_hidden_layers,
        "vocab_size": 151552,
        "rms_norm_eps": 1e-5,
        "rope_theta": 10000,
        "num_experts_per_tok": 8,
        "n_routed_experts": 160,
        "moe_intermediate_size": 1536,
        "n_shared_experts": 1,
        "routed_scaling_factor": 2.5,
        "first_k_dense_replace": 3,
        "n_group": 1,
        "topk_group": 1,
        "norm_topk_prob": True,
        "use_qk_norm": True,
        "num_nextn_predict_layers": 1,
        "max_position_embeddings": 202752,
        # The real GLM-4.7 checkpoint carries the new transformers key only.
        "dtype": "bfloat16",
    }


class _FakeCkpt:
    def __init__(self, name):
        self.name = name


class _FakeAtomic:
    def __init__(self, names):
        self.weights = [_FakeCkpt(n) for n in names]


class _FakeComposite:
    def __init__(self, sub):
        self.sub_weights = sub


class FindNextnLayerIdTest(unittest.TestCase):
    def test_finds_the_marker_layer(self):
        keys = [
            "model.embed_tokens.weight",
            "model.layers.0.input_layernorm.weight",
            "model.layers.92.enorm.weight",
            "model.layers.92.hnorm.weight",
            "model.layers.92.eh_proj.weight",
        ]
        self.assertEqual(find_nextn_layer_id(keys), 92)

    def test_index_is_read_off_the_checkpoint_not_assumed(self):
        # A re-exported checkpoint that renumbers the NextN layer must be followed,
        # not overridden by a hardcoded 92.
        self.assertEqual(find_nextn_layer_id(["model.layers.61.enorm.weight"]), 61)

    def test_rejects_a_checkpoint_without_a_nextn_layer(self):
        with self.assertRaises(ValueError):
            find_nextn_layer_id(["model.layers.0.input_layernorm.weight"])

    def test_rejects_more_than_one_nextn_layer(self):
        with self.assertRaises(ValueError):
            find_nextn_layer_id(
                ["model.layers.92.enorm.weight", "model.layers.93.enorm.weight"]
            )

    def test_ignores_non_numeric_and_foreign_prefixes(self):
        with self.assertRaises(ValueError):
            find_nextn_layer_id(
                ["model.layers.x.enorm.weight", "draft.layers.92.enorm.weight"]
            )


class RetargetLayerTest(unittest.TestCase):
    def test_rewrites_the_layer_index(self):
        module = _FakeAtomic(["model.layers.{i}.self_attn.q_proj.weight"])
        _retarget_layer(module, 92)
        self.assertEqual(
            module.weights[0].name, "model.layers.92.self_attn.q_proj.weight"
        )

    def test_preserves_the_expert_placeholder(self):
        # {expert_id} is resolved later by the MoE weight itself; retargeting must
        # not consume it. Losing it would load one expert 160 times.
        module = _FakeAtomic(
            ["model.layers.{i}.mlp.experts.{expert_id}.down_proj.weight"]
        )
        _retarget_layer(module, 92)
        self.assertEqual(
            module.weights[0].name,
            "model.layers.92.mlp.experts.{expert_id}.down_proj.weight",
        )

    def test_recurses_into_sub_weights(self):
        # MoE weights nest their real tensors under sub_weights; a walk that only
        # looked at .weights would leave the experts pointing at layer 0.
        inner = _FakeAtomic(["model.layers.{i}.mlp.gate.weight"])
        module = _FakeComposite({"gate": inner})
        _retarget_layer(module, 92)
        self.assertEqual(inner.weights[0].name, "model.layers.92.mlp.gate.weight")

    def test_leaves_unrelated_names_alone(self):
        module = _FakeAtomic(["lm_head.weight", "model.norm.weight"])
        _retarget_layer(module, 92)
        self.assertEqual(
            [c.name for c in module.weights], ["lm_head.weight", "model.norm.weight"]
        )


class Glm4MoeNextNConfigTest(unittest.TestCase):
    def _create_config(self, config_json):
        with tempfile.TemporaryDirectory() as ckpt:
            with open(os.path.join(ckpt, "config.json"), "w") as f:
                json.dump(config_json, f)
            return Glm4MoeNextN._create_config(ckpt)

    def test_collapses_to_a_single_layer(self):
        config = self._create_config(_glm47_config_json())
        self.assertEqual(config.num_layers, 1)

    def test_the_single_layer_is_moe(self):
        # first_k_dense_replace=3 makes layers 0-2 dense in the target, but the
        # NextN layer is a MoE layer, so the draft's layer 0 must be in the index.
        config = self._create_config(_glm47_config_json())
        self.assertEqual(list(config.moe_layer_index), [0])

    def test_marks_itself_as_mtp(self):
        config = self._create_config(_glm47_config_json())
        self.assertTrue(config.is_mtp)

    def test_keeps_glm_eh_proj_order(self):
        # GLM's eh_proj is trained on [embed; hidden]; DeepSeek's is [hidden; embed]
        # and sets reverse_e_h_norm. Flipping this silently lowers the accept rate
        # instead of failing, so it is pinned by a test.
        config = self._create_config(_glm47_config_json())
        self.assertFalse(config.reverse_e_h_norm)

    def test_structure_fields_survive_the_override(self):
        config = self._create_config(_glm47_config_json())
        self.assertEqual(config.attn_config.head_num, 96)
        self.assertEqual(config.attn_config.kv_head_num, 8)
        self.assertEqual(config.expert_num, 160)


class Glm4MoeNextNWeightGuardTest(unittest.TestCase):
    """The guards in _get_weight_info, without building the full loader object.

    Constructing Glm4MoeNextNWeight for real needs pybind config objects; these
    three failure modes are pure attribute checks, so drive them directly.
    """

    def _weight(self, nextn_layer_id, num_layers, moe_layer_index):
        w = object.__new__(Glm4MoeNextNWeight)
        w._nextn_layer_id = nextn_layer_id
        w._num_layers = num_layers
        w.moe_layer_index_ = moe_layer_index
        return w

    def test_refuses_when_process_meta_has_not_run(self):
        w = self._weight(-1, 1, [0])
        with self.assertRaises(RuntimeError):
            w._get_weight_info()

    def test_refuses_more_than_one_layer(self):
        w = self._weight(92, 92, [0])
        with self.assertRaises(ValueError):
            w._get_weight_info()

    def test_refuses_a_dense_draft_layer(self):
        w = self._weight(92, 1, [])
        with self.assertRaises(ValueError):
            w._get_weight_info()


class Glm4MoeNextNRegistrationTest(unittest.TestCase):
    """The draft type has to be resolvable by name, not just importable.

    register_model only runs once rtp_llm.models.glm4_moe is imported, and nothing
    imports it until the lazy registry says which module owns the name. A missing
    lazy entry therefore shows up as '--sp_model_type glm4_moe_nextn is unknown'
    at startup rather than as an import error, which is why it gets its own test.
    """

    def test_lazy_registry_knows_the_draft_module(self):
        from rtp_llm.model_factory_register import get_lazy_model_module_path

        self.assertEqual(
            get_lazy_model_module_path("glm4_moe_nextn"), "rtp_llm.models.glm4_moe"
        )

    def test_draft_shares_the_target_module(self):
        from rtp_llm.model_factory_register import get_lazy_model_module_path

        self.assertEqual(
            get_lazy_model_module_path("glm4_moe_nextn"),
            get_lazy_model_module_path("glm4_moe"),
        )

    def test_model_factory_resolves_the_draft_class(self):
        from rtp_llm.model_factory import ModelFactory

        self.assertIs(ModelFactory.get_model_cls("glm4_moe_nextn"), Glm4MoeNextN)

    def test_nextn_architecture_maps_to_the_draft_type(self):
        import rtp_llm.model_factory_register as reg

        reg.ensure_model_registered("glm4_moe_nextn")
        mapping = dict(reg._hf_repo_2_ft)
        mapping.update(reg._hf_architecture_2_ft)
        self.assertEqual(mapping.get("Glm4MoeForCausalLMNextN"), "glm4_moe_nextn")

    def test_sp_type_is_coerced_to_mtp_for_the_draft(self):
        # model_factory coerces sp_type to MTP for known MTP draft types. If
        # glm4_moe_nextn is missing from that list, a run that sets the draft but
        # leaves sp_type at vanilla/eagle silently uses the wrong sampler.
        import inspect

        from rtp_llm import model_factory

        src = inspect.getsource(model_factory)
        self.assertIn("glm4_moe_nextn", src)


class ResolveConfigDtypeTest(unittest.TestCase):
    """The checkpoint dtype must be found under either spelling.

    Getting this wrong does not fail at config time: data_type falls back to
    FP16 and the run dies much later, after loading every weight, with
    "no registered MOE compute backend can consume them" -- because the W8A8
    INT8 MoE executor requires bf16 activations. These cases pin the behaviour
    at the place where it is cheap to see.
    """

    class _Cfg:
        def __init__(self):
            self.config_dtype = None

    def _resolve(self, config_json, cfg=None):
        cfg = cfg if cfg is not None else self._Cfg()
        with tempfile.TemporaryDirectory() as ckpt:
            if config_json is not None:
                with open(os.path.join(ckpt, "config.json"), "w") as f:
                    json.dump(config_json, f)
            resolve_config_dtype(cfg, ckpt)
        return cfg

    def test_reads_the_new_dtype_key(self):
        self.assertEqual(self._resolve({"dtype": "bfloat16"}).config_dtype, "bfloat16")

    def test_reads_the_legacy_torch_dtype_key(self):
        self.assertEqual(
            self._resolve({"torch_dtype": "bfloat16"}).config_dtype, "bfloat16"
        )

    def test_legacy_key_wins_when_both_present(self):
        cfg = self._resolve({"torch_dtype": "float16", "dtype": "bfloat16"})
        self.assertEqual(cfg.config_dtype, "float16")

    def test_leaves_an_explicit_value_alone(self):
        cfg = self._Cfg()
        cfg.config_dtype = "float16"
        self._resolve({"dtype": "bfloat16"}, cfg)
        self.assertEqual(cfg.config_dtype, "float16")

    def test_rejects_a_checkpoint_with_neither_key(self):
        with self.assertRaises(ValueError):
            self._resolve({"hidden_size": 5120})

    def test_rejects_a_missing_config_json(self):
        with self.assertRaises(FileNotFoundError):
            self._resolve(None)


class Glm4MoeNextNDtypeTest(unittest.TestCase):
    def test_draft_config_inherits_the_dtype_resolution(self):
        # Glm4MoeNextN._create_config goes through Glm4Moe, so the draft must
        # pick up bfloat16 without its own copy of the logic.
        with tempfile.TemporaryDirectory() as ckpt:
            with open(os.path.join(ckpt, "config.json"), "w") as f:
                json.dump(_glm47_config_json(), f)
            config = Glm4MoeNextN._create_config(ckpt)
        self.assertEqual(config.config_dtype, "bfloat16")


class RouterLogitsFp32Test(unittest.TestCase):
    """GLM must route on fp32 logits.

    Measured on the real checkpoint: a bf16 router projection moves the logits by
    only 1.7e-3 relative, but reorders near-ties in the top-8-of-160 selection for
    3% of all (layer, token) pairs, starting at the first MoE layer. Because a
    different expert is a discrete change, the error then compounds -- 0.2
    relative by layer 69 against SGLang. Nothing fails loudly if this regresses,
    so it is pinned here.
    """

    def test_default_is_off_so_other_models_are_unaffected(self):
        from rtp_llm.config.model_config import ModelConfig

        self.assertFalse(ModelConfig().router_logits_fp32)

    def test_glm4_moe_turns_it_on(self):
        with tempfile.TemporaryDirectory() as ckpt:
            with open(os.path.join(ckpt, "config.json"), "w") as f:
                json.dump(_glm47_config_json(), f)
            config = Glm4Moe._create_config(ckpt)
        self.assertTrue(config.router_logits_fp32)

    def test_the_nextn_draft_inherits_it(self):
        # The draft runs the same MoE layer, so it has to route the same way or
        # its proposals will disagree with the target for a different reason.
        with tempfile.TemporaryDirectory() as ckpt:
            with open(os.path.join(ckpt, "config.json"), "w") as f:
                json.dump(_glm47_config_json(), f)
            config = Glm4MoeNextN._create_config(ckpt)
        self.assertTrue(config.router_logits_fp32)


if __name__ == "__main__":
    unittest.main()

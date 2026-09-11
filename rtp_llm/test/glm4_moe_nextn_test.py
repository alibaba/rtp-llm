import json
import os
import tempfile
import unittest

import torch

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.models.glm4_moe import (
    Glm4Moe,
    Glm4MoeNextN,
    Glm4MoeNextNWeight,
    _retarget_layer,
    find_nextn_layer_id,
    resolve_config_dtype,
)
from rtp_llm.utils.model_weight import W


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


class Glm4MoeNextNProcessMetaTest(unittest.TestCase):
    """_process_meta's successful path and its quantization guard.

    _process_meta is what turns checkpoint key names into the layer index the
    rest of the loader depends on, so it gets driven directly rather than only
    through the failure guards below.
    """

    def _meta(self, weight_keys):
        w = object.__new__(Glm4MoeNextNWeight)
        w._nextn_layer_id = -1
        w._process_meta(None, weight_keys)
        return w

    @staticmethod
    def _nextn_keys(layer_id=92, with_router_bias=True, extra=()):
        keys = {
            f"model.layers.{layer_id}.enorm.weight",
            f"model.layers.{layer_id}.hnorm.weight",
            f"model.layers.{layer_id}.eh_proj.weight",
            f"model.layers.{layer_id}.shared_head.head.weight",
            f"model.layers.{layer_id}.shared_head.norm.weight",
            f"model.layers.{layer_id}.embed_tokens.weight",
        }
        if with_router_bias:
            keys.add(f"model.layers.{layer_id}.mlp.gate.e_score_correction_bias")
        keys.update(extra)
        return keys

    def test_discovers_the_nextn_layer_id(self):
        self.assertEqual(self._meta(self._nextn_keys(92))._nextn_layer_id, 92)

    def test_router_bias_is_probed_on_the_nextn_layer(self):
        # The target path probes layer 0; a one-layer draft must probe the real
        # NextN index instead, or the correction bias is silently dropped.
        w = self._meta(self._nextn_keys(92, with_router_bias=True))
        self.assertTrue(w.has_e_score_correction_bias)

    def test_absent_router_bias_is_reported_as_absent(self):
        w = self._meta(self._nextn_keys(92, with_router_bias=False))
        self.assertFalse(w.has_e_score_correction_bias)

    def test_rejects_a_quantized_shared_head(self):
        keys = self._nextn_keys(
            92, extra=("model.layers.92.shared_head.head.weight_scale",)
        )
        with self.assertRaises(ValueError) as caught:
            self._meta(keys)
        self.assertIn("shared_head.head.weight_scale", str(caught.exception))

    def test_rejects_a_quantized_eh_proj(self):
        keys = self._nextn_keys(92, extra=("model.layers.92.eh_proj.weight_scale",))
        with self.assertRaises(ValueError) as caught:
            self._meta(keys)
        self.assertIn("eh_proj.weight_scale", str(caught.exception))

    def test_accepts_the_shipped_unquantized_layout(self):
        # GLM-4.7 INT8 W8A8 ships both tensors as BF16 with no scale, so the
        # guard must not fire on the checkpoint this model actually runs on.
        self._meta(self._nextn_keys(92))


def _collect_ckpt_names(module) -> set:
    """Every checkpoint tensor name a weight module tree reads."""
    names = set(getattr(module, "get_ckpt_tensor_names", list)() or [])
    for sub in getattr(module, "sub_weights", {}).values():
        names |= _collect_ckpt_names(sub)
    return names


class Glm4MoeNextNWeightInfoTest(unittest.TestCase):
    """The _process_meta -> _get_weight_info success path on a real weight object.

    The guards below only ever ran through object.__new__, so the redirection
    itself -- which layer the checkpoint names point at, and which tensors the
    draft takes from its own NextN layer rather than from the target -- had no
    execution coverage.
    """

    NEXTN_LAYER = 92

    def _weight_info(self):
        from rtp_llm.config.kv_cache_config import KVCacheConfig
        from rtp_llm.ops import HWKernelConfig, ParallelismConfig

        with tempfile.TemporaryDirectory() as ckpt:
            with open(os.path.join(ckpt, "config.json"), "w") as f:
                json.dump(_glm47_config_json(), f)
            config = Glm4MoeNextN._create_config(ckpt)
        weight = Glm4MoeNextNWeight(
            model_config=config,
            parallelism_config=ParallelismConfig(),
            hw_kernel_config=HWKernelConfig(),
            kv_cache_config=KVCacheConfig(),
        )
        weight._process_meta(
            None, Glm4MoeNextNProcessMetaTest._nextn_keys(self.NEXTN_LAYER)
        )
        return weight, weight._get_weight_info()

    def test_every_layer_tensor_is_retargeted_onto_the_nextn_layer(self):
        _weight, info = self._weight_info()
        prefix = f"model.layers.{self.NEXTN_LAYER}."
        layer_names = set()
        for module in info.layer_weights[0]:
            layer_names |= _collect_ckpt_names(module)
        self.assertTrue(layer_names, "the NextN layer read no checkpoint tensors")
        stray = {n for n in layer_names if not n.startswith(prefix)}
        self.assertEqual(stray, set(), f"tensors not retargeted onto {prefix}: {stray}")

    def test_no_layer_tensor_still_points_at_layer_zero(self):
        # _retarget_layer_list rewrites the target's layer-0 names. A missed one
        # silently loads the wrong layer's weights instead of failing.
        _weight, info = self._weight_info()
        layer_names = set()
        for module in info.layer_weights[0]:
            layer_names |= _collect_ckpt_names(module)
        self.assertEqual(
            {n for n in layer_names if n.startswith("model.layers.0.")}, set()
        )

    def test_embedding_and_lm_head_come_from_the_draft_layer(self):
        # The draft owns private copies of both; taking the target's would make
        # the draft score with the wrong head.
        _weight, info = self._weight_info()
        globals_by_name = {m.name: _collect_ckpt_names(m) for m in info.weights}
        prefix = f"model.layers.{self.NEXTN_LAYER}"
        self.assertEqual(
            globals_by_name[W.embedding], {f"{prefix}.embed_tokens.weight"}
        )
        self.assertEqual(
            globals_by_name[W.lm_head], {f"{prefix}.shared_head.head.weight"}
        )

    def test_mtp_scaffold_tensors_are_in_the_layer_list(self):
        _weight, info = self._weight_info()
        layer_module_names = {m.name for m in info.layer_weights[0]}
        for name in (
            W.multi_tokens_predict_enorm,
            W.multi_tokens_predict_hnorm,
            W.multi_tokens_predict_eh_proj,
            W.multi_tokens_predict_final_ln_gamma,
        ):
            self.assertIn(name, layer_module_names)

    def test_single_layer_weight_info_shape(self):
        _weight, info = self._weight_info()
        self.assertEqual(len(info.layer_weights), 1)


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
        # Go through the production lookup, not the raw maps: only the
        # architecture map is consulted here, so registering the name as an HF
        # repo would resolve to None while a merged-map assertion still passed.
        import rtp_llm.model_factory_register as reg
        from rtp_llm.model_factory_register import ModelDict

        reg.ensure_model_registered("glm4_moe_nextn")
        self.assertEqual(
            ModelDict.get_ft_model_type_by_config(
                {"architectures": ["Glm4MoeForCausalLMNextN"]}
            ),
            "glm4_moe_nextn",
        )

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

    def test_leaves_dtype_unset_when_neither_key_present(self):
        # Must not raise: resolve_config_dtype runs from _create_config, which
        # model_factory calls before build_model_config forwards act_type. A
        # raise here would reject the very --act_type that can satisfy the model.
        self.assertIsNone(self._resolve({"hidden_size": 5120}).config_dtype)

    def test_explicit_act_type_still_wins_without_a_ckpt_dtype(self):
        # The regression this pins: a checkpoint with no dtype field must still
        # be startable by passing --act_type, resolved by init_precision_config's
        # act_type > config_dtype > FP16 priority rather than dying at config time.
        config = ModelConfig()
        with tempfile.TemporaryDirectory() as ckpt:
            with open(os.path.join(ckpt, "config.json"), "w") as f:
                json.dump({"hidden_size": 5120}, f)
            resolve_config_dtype(config, ckpt)
            self.assertIsNone(config.config_dtype)
            config.ckpt_path = ckpt
            config.init_precision_config(kv_cache_config=None, act_type="bf16")
        self.assertEqual(config.compute_dtype, torch.bfloat16)

    def test_falls_back_to_fp16_when_neither_ckpt_dtype_nor_act_type(self):
        config = ModelConfig()
        with tempfile.TemporaryDirectory() as ckpt:
            with open(os.path.join(ckpt, "config.json"), "w") as f:
                json.dump({"hidden_size": 5120}, f)
            resolve_config_dtype(config, ckpt)
            config.ckpt_path = ckpt
            config.init_precision_config(kv_cache_config=None, act_type=None)
        self.assertEqual(config.compute_dtype, torch.float16)

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


class CompressedInt8RecognitionTest(unittest.TestCase):
    """W8A8 INT8 must be recognised from the checkpoint, and only from there.

    The scheme is deliberately absent from preset_quant_config, so it cannot be
    asked for with --quantization; it is inferred from the checkpoint's own
    quantization_config block. Both halves of that arrangement matter. If the
    checkpoint inference regresses, the run reaches the MoE factory unquantised
    and dies with "no registered MOE compute backend can consume them" only
    after every weight is loaded. If someone "helpfully" adds the scheme to the
    preset table, it becomes selectable by hand and can then disagree with what
    the checkpoint actually contains.
    """

    # The quantization_config block as GLM-4.7-W8A8-INT8 actually ships it.
    QUANT_BLOCK = {
        "quant_method": "compressed-tensors",
        "format": "int-quantized",
        "quantization_status": "compressed",
        "ignore": ["lm_head"],
        "config_groups": {
            "group_0": {
                "targets": ["Linear"],
                "weights": {
                    "num_bits": 8,
                    "strategy": "channel",
                    "symmetric": True,
                    "type": "int",
                    "dynamic": False,
                },
                "input_activations": {
                    "num_bits": 8,
                    "strategy": "token",
                    "symmetric": True,
                    "type": "int",
                    "dynamic": True,
                },
                "output_activations": None,
            }
        },
    }

    def _load(self, quant_block):
        from rtp_llm.config.quant_config import QuantizationConfig

        cfg = _glm47_config_json()
        if quant_block is not None:
            cfg["quantization_config"] = quant_block
        with tempfile.TemporaryDirectory() as ckpt:
            with open(os.path.join(ckpt, "config.json"), "w") as f:
                json.dump(cfg, f)
            return QuantizationConfig.load_from_ckpt(ckpt)

    def test_recognised_from_the_checkpoint(self):
        quant = self._load(self.QUANT_BLOCK)
        self.assertIsNotNone(quant, "the checkpoint's quantization_config was ignored")
        self.assertEqual(quant.get_method(), "W8A8_INT8_PER_CHANNEL_COMPRESSED")

    def test_the_ignore_list_survives(self):
        # lm_head is not quantised in this checkpoint; losing that would quantise
        # the output projection and change every logit.
        quant = self._load(self.QUANT_BLOCK)
        self.assertIn("lm_head", set(quant.exclude_modules))

    def test_a_plain_checkpoint_stays_unquantised(self):
        self.assertIsNone(self._load(None))

    def test_not_selectable_through_the_preset_table(self):
        from rtp_llm.config.quant_config import preset_quant_config

        self.assertNotIn(
            "W8A8_INT8_PER_CHANNEL_COMPRESSED",
            preset_quant_config,
            "the scheme is meant to be inferred from the checkpoint only; adding it "
            "here makes it hand-selectable and able to contradict the checkpoint",
        )


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

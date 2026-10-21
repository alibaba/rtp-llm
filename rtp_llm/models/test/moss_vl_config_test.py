import json
import tempfile
import unittest
from pathlib import Path

from rtp_llm.model_factory import ModelFactory
from rtp_llm.model_factory_register import ModelDict
from rtp_llm.server.server_args.server_args import setup_args
from rtp_llm.utils.model_weight import W

# Real OpenMOSS-Team/MOSS-VL-Instruct-0708 config.json. Values were
# cross-checked against the checkpoint tensor shapes (895 tensors, 11.336B
# params, all BF16; merger 18432, cross_attn k/v [1024, 4096]).
MOSS_VL_CONFIG = {
    "architectures": ["MossVLForConditionalGeneration"],
    "auto_map": {
        "AutoConfig": "configuration_moss_vl.MossVLConfig",
        "AutoModel": "modeling_moss_vl.MossVLForConditionalGeneration",
        "AutoModelForCausalLM": "modeling_moss_vl.MossVLForConditionalGeneration",
    },
    "dtype": "bfloat16",
    "image_token_id": 151655,
    "model_type": "moss_vl",
    "text_config": {
        "attention_bias": False,
        "attention_dropout": 0.0,
        "bos_token_id": 151643,
        "cross_attention_layers": [2, 6, 10, 14, 18, 22, 26, 30, 34, 38, 42, 46],
        "dtype": "bfloat16",
        "eos_token_id": 151645,
        "head_dim": 128,
        "hidden_act": "silu",
        "hidden_size": 4096,
        "initializer_range": 0.02,
        "intermediate_size": 12288,
        "max_position_embeddings": 262144,
        "model_type": "moss_vl_text",
        "num_attention_heads": 32,
        "num_hidden_layers": 48,
        "num_key_value_heads": 8,
        "rms_norm_eps": 1e-06,
        "rope_scaling": {
            "mrope_interleaved": True,
            "mrope_section": [24, 20, 20],
            "rope_type": "default",
        },
        "rope_theta": 5000000,
        "use_cache": True,
        "vocab_size": 151936,
    },
    "tie_word_embeddings": False,
    "transformers_version": "4.57.1",
    "video_token_id": 151656,
    "vision_config": {
        "deepstack_visual_indexes": [8, 16, 24],
        "depth": 27,
        "hidden_act": "gelu_pytorch_tanh",
        "hidden_size": 1152,
        "in_channels": 3,
        "initializer_range": 0.02,
        "intermediate_size": 4304,
        "model_type": "moss_vl_vision",
        "num_heads": 16,
        "num_position_embeddings": 2304,
        "out_hidden_size": 4096,
        "patch_size": 16,
        "spatial_merge_size": 2,
        "temporal_patch_size": 1,
    },
    "vision_end_token_id": 151653,
    "vision_seq_pad_multiple": 1,
    "vision_start_token_id": 151652,
}

EXPECTED_CROSS_LAYERS = [2, 6, 10, 14, 18, 22, 26, 30, 34, 38, 42, 46]
IMAGE_PLACEHOLDER = chr(60) + chr(124) + "image" + chr(124) + chr(62)


class MossVLConfigTest(unittest.TestCase):
    def _write_ckpt(self) -> str:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        ckpt = Path(tmp.name)
        (ckpt / "config.json").write_text(json.dumps(MOSS_VL_CONFIG))
        return str(ckpt)

    def _create_model_config(self, ckpt: str, extra_argv=()):
        argv = [
            "--model_type",
            "moss_vl",
            "--checkpoint_path",
            ckpt,
            "--act_type",
            "BF16",
            *extra_argv,
        ]
        configs = setup_args(argv)
        return ModelFactory.create_model_config(
            model_args=configs.model_args,
            lora_config=configs.lora_config,
            kv_cache_config=configs.kv_cache_config,
            profiling_debug_logging_config=configs.profiling_debug_logging_config,
        )

    def test_full_factory_config_lifecycle(self):
        ckpt = self._write_ckpt()
        config = self._create_model_config(ckpt)

        # text model geometry (non-default values throughout)
        self.assertEqual(config.num_layers, 48)
        self.assertEqual(config.hidden_size, 4096)
        self.assertEqual(config.inter_size, 12288)
        self.assertEqual(config.attn_config.head_num, 32)
        self.assertEqual(config.attn_config.kv_head_num, 8)
        self.assertEqual(config.attn_config.size_per_head, 128)
        self.assertEqual(config.vocab_size, 151936)
        self.assertEqual(config.layernorm_eps, 1e-06)
        self.assertFalse(config.tie_word_embeddings)
        self.assertTrue(config.qk_norm)
        self.assertEqual(config.max_seq_len, 262144)

        # interleaved MRoPE contract
        rope = config.attn_config.rope_config
        self.assertEqual(int(rope.style), 7)
        self.assertEqual(rope.dim, 128)
        self.assertEqual(rope.base, 5000000)
        self.assertEqual(rope.index_factor, 3)
        self.assertEqual((rope.mrope_dim1, rope.mrope_dim2, rope.mrope_dim3), (24, 20, 20))
        self.assertTrue(rope.mrope_interleaved)

        # multimodal wiring: MARKER_MROPE style + marker token activation
        self.assertTrue(config.mm_model_config.is_multimodal)
        self.assertEqual(config.mm_model_config.mm_sep_tokens, [[151652, 151653]])
        self.assertEqual(config.mm_model_config.mm_position_ids_style, 3)
        self.assertEqual(config.mm_model_config.mm_marker_token, 151655)
        self.assertEqual(
            config.mm_related_params.special_tokens["default_mm_token"],
            IMAGE_PLACEHOLDER,
        )

        # marker-mode metadata carried for the mixin / engine
        mm_config = config.mm_related_params.config
        self.assertEqual(mm_config["cross_attention_layers"], EXPECTED_CROSS_LAYERS)
        self.assertEqual(mm_config["image_token_id"], 151655)
        self.assertEqual(mm_config["video_token_id"], 151656)
        self.assertEqual(mm_config["vision_start_token_id"], 151652)
        self.assertEqual(mm_config["vision_end_token_id"], 151653)
        self.assertEqual(mm_config["vision_seq_pad_multiple"], 1)
        self.assertEqual(mm_config["vision_config"]["depth"], 27)
        self.assertEqual(mm_config["vision_config"]["out_hidden_size"], 4096)
        self.assertEqual(mm_config["vision_config"]["deepstack_visual_indexes"], [8, 16, 24])
        self.assertEqual(mm_config["ckpt_path"], ckpt)

    def test_lazy_class_resolution(self):
        # factory path resolves the class without any test-side import of
        # rtp_llm.models.moss_vl (guards the lazy registration contract).
        model_cls = ModelFactory.get_model_cls("moss_vl")
        self.assertEqual(model_cls.__name__, "MossVL")
        self.assertEqual(model_cls.get_weight_cls().__name__, "MossVLWeight")
        self.assertEqual(
            ModelDict.get_ft_model_type_by_hf_architectures(
                "MossVLForConditionalGeneration"
            ),
            "moss_vl",
        )

    def test_architecture_inference_from_config(self):
        from rtp_llm.config.server_config_setup import _infer_model_type

        ckpt = self._write_ckpt()
        self.assertEqual(_infer_model_type(ckpt), "moss_vl")

    def test_cli_max_seq_len_override(self):
        ckpt = self._write_ckpt()
        config = self._create_model_config(ckpt, ["--max_seq_len", "8192"])
        self.assertEqual(config.max_seq_len, 8192)

    def test_cross_gate_weight_names_registered(self):
        # the two tanh-gate slots must exist in W and the TP split table so
        # the loader does not KeyError on _get_split_func (TP=1 bring-up).
        self.assertEqual(W.cross_attn_attn_gate, "cross_attention_weights.attn_gate")
        self.assertEqual(W.cross_attn_mlp_gate, "cross_attention_weights.mlp_gate")


if __name__ == "__main__":
    unittest.main()

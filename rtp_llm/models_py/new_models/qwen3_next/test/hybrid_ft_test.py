"""Producer descriptors → rank-local files → real text owners, no golden tensors."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from safetensors.torch import save_file

from rtp_llm.models.kimi_k25.kimi_k25 import KimiK25
from rtp_llm.models.kimi_linear.kimi_linear import KimiLinear
from rtp_llm.models.qwen3_next.qwen3_next import Qwen3Next, Qwen35Dense, Qwen35Moe
from rtp_llm.models.qwen3_next.qwen3_next_mtp import (
    Qwen3NextMTP,
    Qwen35DenseMTP,
    Qwen35MoeMTP,
)
from rtp_llm.models_py.ft_checkpoint import CheckpointTopology, FTCheckpoint
from rtp_llm.models_py.model_loader import NewModelLoader
from rtp_llm.models_py.new_models.kimi_k25.test.test_language import (
    config as k25_config,
)
from rtp_llm.models_py.new_models.kimi_k25.test.test_language import language_weights
from rtp_llm.models_py.new_models.kimi_linear.test.test_kimi_owner import (
    _checkpoint_weights,
    _raw_config,
)
from rtp_llm.models_py.new_models.qwen3_next.language import Qwen3NextForCausalLM
from rtp_llm.models_py.new_models.qwen3_next.test.main_checkpoint_test import (
    _config,
    _dense_weights,
    _load_config,
)
from rtp_llm.ops import (
    DataType,
    HWKernelConfig,
    HybridAttentionType,
    KVCacheConfig,
    ParallelismConfig,
)
from rtp_llm.tools.convert.ft_export_job import create_ft_export_job
from rtp_llm.tools.convert.ft_export_metadata import generate_ft_index


class HybridFTRoundTrip(unittest.TestCase):
    def _check(self, source, ft, config, wrapper, weights, *, stage_only=False):
        config.ckpt_path = str(source)
        config.data_type = "fp32"
        config.lora_infos = {}
        save_file(weights, str(source / "model.safetensors"))
        pc = ParallelismConfig()
        pc.ep_size = 1
        descriptor = SimpleNamespace(
            model_config=config,
            parallelism_config=pc,
            get_weight_cls=wrapper.get_weight_cls,
            hw_kernel_config=HWKernelConfig(),
            kv_cache_config=KVCacheConfig(),
            merge_lora=False,
            load_method="scratch",
            force_cpu_load_weights=True,
            moe_pure_tp_preshard=False,
            loader_recycle_handles=False,
            _init_custom_module=lambda: None,
        )
        create_ft_export_job(descriptor).dump_weight_as_ft_style("cpu", str(ft))
        generate_ft_index(str(ft), {})
        if stage_only:
            # GDN explicitly has no CPU execution backend. Exercise the two real
            # parameter importers and completeness checks, not a fake CPU service.
            with torch.no_grad():
                reference = Qwen3NextForCausalLM(config, _load_config())
                reference.load_weights(weights.items())
                restored = Qwen3NextForCausalLM(config, _load_config())
                restored.load_ft_checkpoint(
                    FTCheckpoint.open(str(ft), CheckpointTopology())
                )
                for owner in (reference, restored):
                    NewModelLoader._validate_loaded_weights(owner)
        else:
            reference = NewModelLoader(config, _load_config(), str(source)).load()
            config.ckpt_path = str(ft)
            restored = NewModelLoader(config, _load_config(), str(ft)).load()

        def runtime_state(owner):
            # KC/VC are derived (nonpersistent) in HF but independently serialized
            # in FT. Compare the live buffers, including HF's derived values.
            return {
                **owner.state_dict(),
                **{
                    k: v
                    for k, v in owner.named_buffers()
                    if k.endswith(("._kc_w", "._vc_w"))
                },
            }

        expected, actual = runtime_state(reference), runtime_state(restored)
        self.assertEqual(actual.keys(), expected.keys())
        for name in expected:
            with self.subTest(name=name):
                torch.testing.assert_close(actual[name], expected[name], rtol=0, atol=0)

    def test_qwen35_dense_mtp(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ft = root / "ft"
            cfg = _config(tie=False)
            cfg.model_type = "qwen35_dense_mtp"
            cfg.is_mtp = True
            cfg.expert_num = cfg.moe_k = cfg.moe_inter_size = cfg.moe_style = 0
            weights = {}
            for key, value in _dense_weights(True).items():
                if key == "model.embed_tokens.weight":
                    key = "model.language_model.embed_tokens.weight"
                elif key.startswith("model."):
                    key = "mtp." + key.removeprefix("model.")
                weights[key] = value
            weights.update(
                {
                    "mtp.fc.weight": torch.arange(32).reshape(4, 8).float(),
                    "mtp.pre_fc_norm_embedding.weight": torch.arange(4).float(),
                    "mtp.pre_fc_norm_hidden.weight": torch.arange(4).float() + 5,
                }
            )
            self._check(root, ft, cfg, Qwen35DenseMTP, weights)

    def test_qwen35_dense_gdn(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cfg = _config(HybridAttentionType.LINEAR, tie=False)
            cfg.model_type = "qwen35_dense"
            cfg.expert_num = cfg.moe_k = cfg.moe_inter_size = cfg.moe_style = 0
            prefix = "model.language_model.layers.0.linear_attn."
            weights = {
                key.replace("model.", "model.language_model.", 1): value
                for key, value in _dense_weights(True).items()
                if ".self_attn." not in key
            }
            for name, shape in (
                ("in_proj_qkv.weight", (12, 4)),
                ("in_proj_z.weight", (4, 4)),
                ("in_proj_b.weight", (2, 4)),
                ("in_proj_a.weight", (2, 4)),
                ("conv1d.weight", (12, 1, 2)),
                ("out_proj.weight", (4, 4)),
                ("A_log", (2,)),
                ("dt_bias", (2,)),
                ("norm.weight", (2,)),
            ):
                weights[prefix + name] = (
                    torch.arange(torch.Size(shape).numel()).float().reshape(shape)
                )
            self._check(root, root / "ft", cfg, Qwen35Dense, weights, stage_only=True)

    def test_qwen_moe_targets_and_drafts(self):
        for tag, wrapper in (
            ("qwen3_next", Qwen3Next),
            ("qwen35_moe", Qwen35Moe),
            ("qwen3_next_mtp", Qwen3NextMTP),
            ("qwen35_moe_mtp", Qwen35MoeMTP),
        ):
            with self.subTest(model=tag), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                cfg = _config(tie=False)
                cfg.model_type, cfg.moe_layer_index, cfg.n_shared_experts = tag, [0], 1
                cfg.is_mtp = tag.endswith("_mtp")
                prefix = (
                    "model.language_model." if tag.startswith("qwen35") else "model."
                )
                layer_prefix = "mtp." if cfg.is_mtp else prefix
                weights = {}
                for key, value in _dense_weights(True).items():
                    if key == "model.embed_tokens.weight":
                        weights[prefix + "embed_tokens.weight"] = value
                    elif key == "lm_head.weight":
                        weights[key] = value
                    elif ".mlp." not in key:
                        weights[layer_prefix + key.removeprefix("model.")] = value
                mlp = layer_prefix + "layers.0.mlp."
                weights[mlp + "gate.weight"] = torch.arange(8).float().reshape(2, 4)
                weights[mlp + "shared_expert_gate.weight"] = (
                    torch.arange(4).float().reshape(1, 4)
                )
                for slot, name in enumerate(
                    ("shared_expert", "experts.0", "experts.1")
                ):
                    for proj, suffix in enumerate(
                        ("gate_proj", "up_proj", "down_proj")
                    ):
                        weights[mlp + name + "." + suffix + ".weight"] = torch.arange(
                            16
                        ).reshape(4, 4).float() + 16 * (slot * 3 + proj)
                if cfg.is_mtp:
                    weights.update(
                        {
                            "mtp.fc.weight": torch.arange(32).reshape(4, 8).float(),
                            "mtp.pre_fc_norm_embedding.weight": torch.arange(4).float(),
                            "mtp.pre_fc_norm_hidden.weight": torch.arange(4).float()
                            + 5,
                        }
                    )
                self._check(root, root / "ft", cfg, wrapper, weights)

    def test_kimi_linear_kda_mla_moe(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ft = root / "ft"
            ft.mkdir()
            for path in (root, ft):
                (path / "config.json").write_text(json.dumps(_raw_config()))
            cfg = KimiLinear._create_config(str(root))
            cfg.model_type, cfg.max_seq_len, cfg.enable_fp32_lm_head = (
                "kimi_linear",
                128,
                False,
            )
            cfg.linear_attention_config.ssm_state_dtype = DataType.TYPE_FP32
            cfg.linear_attention_config.conv_state_dtype = DataType.TYPE_FP32
            self._check(root, ft, cfg, KimiLinear, _checkpoint_weights())

    def test_kimi_k25_text(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ft = root / "ft"
            ft.mkdir()
            raw = k25_config()
            raw["text_config"]["tie_word_embeddings"] = False
            for path in (root, ft):
                (path / "config.json").write_text(json.dumps(raw))
            cfg = KimiK25._create_config(str(root))
            cfg.model_type, cfg.enable_fp32_lm_head = "kimi_k25", False
            cfg.attn_config.kernel_tokens_per_block = 64
            weights = {"language_model." + k: v for k, v in language_weights().items()}
            self._check(root, ft, cfg, KimiK25, weights)


if __name__ == "__main__":
    unittest.main()

"""Producer descriptors → rank-local files → real text owners, no golden tensors."""

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from safetensors.torch import save_file

from rtp_llm.models.downstream_modules.classifier.classifier import ClassifierHandler
from rtp_llm.models.downstream_modules.custom_module import CustomModule
from rtp_llm.models.kimi_k25.kimi_k25 import KimiK25
from rtp_llm.models.kimi_linear.kimi_linear import KimiLinear
from rtp_llm.models.qwen3_next.qwen3_next import Qwen3Next, Qwen35Dense, Qwen35Moe
from rtp_llm.models.qwen3_next.qwen3_next_mtp import (
    Qwen3NextMTP,
    Qwen35DenseMTP,
    Qwen35MoeMTP,
)
from rtp_llm.models_py.checkpoint_reader_test_support import checkpoint_readers
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
    TaskType,
)
from rtp_llm.tools.convert.ft_export_job import create_ft_export_job
from rtp_llm.tools.convert.ft_export_metadata import generate_ft_index


def _export_ft(source, output, config, wrapper, tp=1, *, handler=None):
    config.ckpt_path = str(source)
    config.lora_infos = {}
    for rank in range(tp):
        pc = ParallelismConfig()
        pc.tp_size = pc.ffn_tp_size = pc.world_size = pc.local_world_size = tp
        pc.tp_rank = pc.ffn_tp_rank = pc.world_rank = pc.local_rank = rank
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
            _init_custom_module=lambda: handler,
        )
        with patch("sys.argv", ["weights_convert"]):
            create_ft_export_job(descriptor).dump_weight_as_ft_style("cpu", str(output))
    generate_ft_index(str(output), {"TP_SIZE": tp, "EP_SIZE": 1, "DP_SIZE": 1})


class HybridFTRoundTrip(unittest.TestCase):
    def _check_decoder_handler(self, source, ft, cfg, wrapper, weights):
        hidden = torch.tensor([[1.0, 2.0, -1.0, 0.0], [0.0, 1.0, 2.0, -1.0]])
        weight = torch.tensor([[1.0, -3.0, 2.0, 5.0], [-2.0, 1.0, 4.0, 3.0]])
        bias = torch.tensor([3.0, -2.0])
        scores = hidden @ weight.T + bias
        (source / "config.json").write_text(json.dumps({"num_labels": 2}))
        save_file(
            {**weights, "classifier.weight": weight, "classifier.bias": bias},
            str(source / "model.safetensors"),
        )
        cfg.ckpt_path = str(source)
        # Both of these services use ClassifierHandler. Exercise its actual
        # declarations and forward without starting a tokenizer/HTTP renderer.
        for task in (TaskType.SEQ_CLASSIFICATION, TaskType.RERANKER):
            cfg.task_type = task
            module = CustomModule(cfg, None)
            module.handler = ClassifierHandler(cfg)
            module.handler.device = torch.device("cpu")
            options = replace(
                _load_config(),
                custom_weight_mappings=module.checkpoint_weight_mappings(),
            )
            archive = source / task.name
            _export_ft(source, archive, cfg, wrapper, handler=module)
            for path in (source, archive):
                for reader in checkpoint_readers("scratch", "fastsafetensors"):
                    with self.subTest(
                        model=cfg.model_type, task=task, path=path.name, reader=reader
                    ):
                        load = replace(
                            options, load_method=reader, force_cpu_load_weights=True
                        )
                        owner = NewModelLoader(cfg, load, str(path)).load()
                        table = owner.runtime_weight_view()
                        self.assertIsNone(owner.lm_head)
                        for key, value in (
                            ("classifier.weight", weight), ("classifier.bias", bias)
                        ):
                            torch.testing.assert_close(
                                table["__custom__." + key], value, rtol=0, atol=0
                            )
                        module.init(
                            SimpleNamespace(get_global_weight=table.__getitem__)
                        )
                        actual = module.handler.forward(
                            torch.tensor([0, 1]), hidden, torch.tensor([1, 1])
                        )
                        torch.testing.assert_close(actual, scores, rtol=0, atol=0)
        # A generation-only archive is not a source for a declared score head.
        with self.assertRaisesRegex(ValueError, "Missing FT handler tensor"):
            NewModelLoader(cfg, options, str(ft)).load()
        save_file(
            {
                **weights,
                "classifier.weight": weight[:, :2].contiguous(),
                "classifier.bias": bias,
            },
            str(source / "model.safetensors"),
        )
        malformed = source / "malformed"
        _export_ft(source, malformed, cfg, wrapper, handler=module)
        with self.assertRaisesRegex(
            ValueError, "Invalid downstream tensor classifier.weight"
        ):
            NewModelLoader(cfg, options, str(malformed)).load()

    def test_qwen35_gated_q_peft_hf_and_ft(self):
        # The reference merges PEFT into the original HF tensor. No
        # expected tensor is constructed using the loader's split helper.
        cfg = _config(tie=False)
        cfg.model_type = "qwen35_dense"
        cfg.expert_num = cfg.moe_k = cfg.moe_inter_size = cfg.moe_style = 0
        weights = {
            key.replace("model.", "model.language_model.", 1): value
            for key, value in _dense_weights(True).items()
        }
        q = "model.language_model.layers.0.self_attn.q_proj"
        a = torch.tensor([[1., 2., 3., 4.], [4., -2., 1., 0.]])
        b = torch.arange(16, dtype=torch.float32).view(8, 2) - 7
        with tempfile.TemporaryDirectory() as tmp:
            src, ref, adapter = (Path(tmp) / name for name in ("hf", "ref", "adapter"))
            for path in (src, ref, adapter):
                path.mkdir()
            save_file(weights, str(src / "model.safetensors"))
            save_file(
                {**weights, q + ".weight": weights[q + ".weight"] + (b @ a) * 1.5},
                str(ref / "model.safetensors"),
            )
            (adapter / "adapter_config.json").write_text(json.dumps({"r": 2, "lora_alpha": 3}))

            def save_adapter(output_factor):
                save_file({
                    f"base_model.model.{q}.lora_A.weight": a,
                    f"base_model.model.{q}.lora_B.weight": output_factor,
                }, str(adapter / "adapter_model.safetensors"))

            save_adapter(b)
            for tp in (1, 2):
                ft = Path(tmp) / f"ft{tp}"
                _export_ft(src, ft, cfg, Qwen35Dense, tp=tp)
                for rank in range(tp):
                    config = _load_config(tp_size=tp, tp_rank=rank)
                    reference = NewModelLoader(cfg, config, str(ref)).load().state_dict()
                    for folder in (src, ft):
                        for backend in checkpoint_readers("scratch", "fastsafetensors"):
                            load = replace(config, load_method=backend, force_cpu_load_weights=True)
                            with self.subTest(tp=tp, rank=rank, path=folder.name, backend=backend):
                                owner = NewModelLoader(cfg, load, str(folder)).load(
                                    startup_lora=("gated", str(adapter))
                                )
                                actual = owner.state_dict()
                                self.assertEqual(actual.keys(), reference.keys())
                                for key in actual:
                                    torch.testing.assert_close(actual[key], reference[key], rtol=0, atol=0)

            # A Q-only factor is not a valid PEFT factor for this HF
            # checkpoint. Never publish half of what the adapter requested.
            save_adapter(b[:4])
            with self.assertRaisesRegex(ValueError, "must have 8 rows"):
                NewModelLoader(cfg, _load_config(), str(src)).load(
                    startup_lora=("truncated", str(adapter))
                )

    def _check(self, source, ft, config, wrapper, weights, *, stage_only=False):
        config.data_type = "fp32"
        save_file(weights, str(source / "model.safetensors"))
        _export_ft(source, ft, config, wrapper)
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
            if config.model_type in ("qwen3_next_mtp", "qwen35_dense_mtp", "qwen35_moe_mtp"):
                embed = ("model.embed_tokens.weight" if config.model_type == "qwen3_next_mtp"
                         else "model.language_model.embed_tokens.weight")
                torch.testing.assert_close(reference.embed_tokens.weight, weights[embed])
                torch.testing.assert_close(reference.lm_head.weight, weights["lm_head.weight"])
                torch.testing.assert_close(reference.norm.weight, weights["mtp.norm.weight"] + 1)
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
        if not stage_only and config.model_type in (
            "qwen3_next", "qwen35_dense", "qwen35_moe"
        ):
            self._check_decoder_handler(source, ft, config, wrapper, weights)

    def test_qwen35_dense_target_and_handler(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cfg = _config(tie=False)
            cfg.model_type = "qwen35_dense"
            cfg.expert_num = cfg.moe_k = cfg.moe_inter_size = cfg.moe_style = 0
            weights = {
                key.replace("model.", "model.language_model.", 1): value
                for key, value in _dense_weights(True).items()
            }
            self._check(root, root / "ft", cfg, Qwen35Dense, weights)

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

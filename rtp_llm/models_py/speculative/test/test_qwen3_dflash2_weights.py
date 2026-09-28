"""Production DFlash2 checkpoint mapping and replicated TP weight contracts."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from safetensors.torch import save_file

from rtp_llm.config.kv_cache_config import KVCacheConfig
from rtp_llm.model_factory import ModelFactory
from rtp_llm.model_factory_register import ModelDict, ensure_model_registered
from rtp_llm.model_loader.load_config import LoadMethod
from rtp_llm.model_loader.loader import ModelLoader
from rtp_llm.models.qwen_3_dflash import Qwen3DFlash
from rtp_llm.models.qwen_3_dflash2 import Qwen3DFlash2, Qwen3DFlash2Weight
from rtp_llm.ops import HWKernelConfig, ParallelismConfig, SpeculativeType
from rtp_llm.utils.database import CkptDatabase
from rtp_llm.utils.model_weight import W


def _raw_config():
    return {
        "architectures": ["DFlash2DraftModel"],
        "model_type": "qwen3",
        "hidden_size": 80,
        "intermediate_size": 128,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "head_dim": 2,
        "num_hidden_layers": 2,
        "vocab_size": 128,
        "dtype": "bfloat16",
        "rms_norm_eps": 1e-6,
        "rope_parameters": {"rope_theta": 10_000_000, "rope_type": "default"},
        "layer_types": ["sliding_attention"] * 2,
        "sliding_window": 2048,
        "dflash_config": {
            "block_size": 8,
            "mask_token_id": 123,
            "target_layer_ids": [5, 19, 33, 47, 61],
            "conv_group_size": 16,
            "conv_kernel_size": 2,
            "selector_rank": 16,
            "selector_top_k": 8,
        },
    }


def _checkpoint():
    tensors = {}

    def add(name, shape):
        tensors[name] = torch.randn(*shape).to(torch.bfloat16)

    add("fc.weight", (80, 5 * 80))
    add("hidden_norm.weight", (80,))
    add("norm.weight", (80,))
    add("candidate_selector.predecessor_codebook", (128, 16))
    add("candidate_selector.successor_codebook", (128, 16))
    add("candidate_selector.hidden_projection.weight", (16, 80))
    for index in range(2):
        prefix = f"layers.{index}."
        for name in ("input_layernorm", "post_attention_layernorm"):
            add(prefix + name + ".weight", (80,))
        for name in ("q_norm", "k_norm"):
            add(prefix + "self_attn." + name + ".weight", (2,))
        for name, rows in (("q", 64), ("k", 16), ("v", 16)):
            add(prefix + "self_attn." + name + "_proj.weight", (rows, 80))
        add(prefix + "self_attn.o_proj.weight", (80, 64))
        for name in ("gate", "up"):
            add(prefix + "mlp." + name + "_proj.weight", (128, 80))
        add(prefix + "mlp.down_proj.weight", (80, 128))
        for name in ("attention_conv", "mlp_conv"):
            add(prefix + name + ".base_kernel", (2, 2, 80))
            add(prefix + name + ".kernel_projection.weight", (20, 80))
    return tensors


class Qwen3DFlash2WeightTest(unittest.TestCase):
    def _setup_configs(self):
        with tempfile.TemporaryDirectory() as path:
            Path(path, "config.json").write_text(json.dumps(_raw_config()))
            draft = Qwen3DFlash2._create_config(path)
        draft.init_precision_config(
            SimpleNamespace(fp8_kv_cache=False), act_type="bf16"
        )
        target = SimpleNamespace(
            num_layers=64,
            hidden_size=80,
            vocab_size=128,
            input_vocab_size=128,
            capture_aux_hidden_layer_ids=None,
        )
        sp = SimpleNamespace(type=SpeculativeType.DFLASH2, gen_num_per_cycle=7)
        return sp, target, draft

    def test_lazy_registry_is_distinct_from_v1(self):
        self.assertTrue(ensure_model_registered("qwen_3_dflash2"))
        self.assertEqual(
            ModelDict.get_ft_model_type_by_config(_raw_config()), "qwen_3_dflash2"
        )

    def test_setup_native_eight_and_all_runtime_widths(self):
        for gamma in range(1, 8):
            sp, target, draft = self._setup_configs()
            sp.gen_num_per_cycle = gamma
            ModelFactory._setup_dflash_configs(sp, target, draft)
            self.assertEqual(target.capture_aux_hidden_layer_ids, [5, 19, 33, 47, 61])
            self.assertFalse(sp.sp_dspark_sample_from_anchor)
            self.assertEqual(sp.sp_dspark_mask_token_id, 123)

    def test_setup_rejects_incompatible_metadata(self):
        for field, value in (
            ("dflash_native_block_size", 16),
            ("dflash2_selector_rank", None),
            ("dflash2_selector_top_k", None),
            ("dflash2_conv_kernel_size", None),
            ("dflash2_conv_group_size", 3),
        ):
            sp, target, draft = self._setup_configs()
            setattr(draft, field, value)
            with self.subTest(field=field), self.assertRaises(ValueError):
                ModelFactory._setup_dflash_configs(sp, target, draft)
        for gamma in (0, 8, 15):
            sp, target, draft = self._setup_configs()
            sp.gen_num_per_cycle = gamma
            with self.subTest(gamma=gamma), self.assertRaisesRegex(ValueError, "gamma"):
                ModelFactory._setup_dflash_configs(sp, target, draft)
        for field in ("hidden_size", "vocab_size", "input_vocab_size"):
            sp, target, draft = self._setup_configs()
            setattr(target, field, getattr(target, field) + 1)
            with self.subTest(field=field), self.assertRaises(ValueError):
                ModelFactory._setup_dflash_configs(sp, target, draft)

    def test_metadata_and_architecture_isolation(self):
        with tempfile.TemporaryDirectory() as path:
            Path(path, "config.json").write_text(json.dumps(_raw_config()))
            config = Qwen3DFlash2._create_config(path)
            self.assertEqual(config.dflash_native_block_size, 8)
            self.assertEqual(config.dflash2_conv_group_size, 16)
            self.assertEqual(config.dflash2_conv_kernel_size, 2)
            self.assertEqual(config.dflash2_selector_rank, 16)
            self.assertEqual(config.dflash2_selector_top_k, 8)
            self.assertEqual(config.dflash2_input_embedding_scale, 1)
            with self.assertRaisesRegex(ValueError, "DFlashDraftModel"):
                Qwen3DFlash._create_config(path)
            for field, value in (
                ("conv_group_size", 3),
                ("conv_kernel_size", 0),
                ("selector_rank", True),
                ("selector_top_k", 129),
            ):
                raw = _raw_config()
                raw["dflash_config"][field] = value
                Path(path, "config.json").write_text(json.dumps(raw))
                with self.subTest(field=field), self.assertRaises(ValueError):
                    Qwen3DFlash2._create_config(path)

    def test_actual_checkpoint_loader_replicates_new_weights_on_every_tp_rank(self):
        tensors = _checkpoint()
        with tempfile.TemporaryDirectory() as path:
            Path(path, "config.json").write_text(json.dumps(_raw_config()))
            save_file(tensors, str(Path(path, "model.safetensors")))
            config = Qwen3DFlash2._create_config(path)
            config.ckpt_path, config.phy2log_path, config.data_type = path, "", "bf16"
            database = CkptDatabase(path)
            try:
                for tp in (1, 2, 8):
                    for rank in range(tp):
                        with self.subTest(tp=tp, rank=rank):
                            self._check_rank(config, database, tensors, tp, rank)
            finally:
                for file_info in database.pretrain_file_list:
                    file_info.close_safetensor_handle()

    def _check_rank(self, config, database, tensors, tp, rank):
        parallel = ParallelismConfig()
        parallel.tp_size = parallel.world_size = parallel.local_world_size = tp
        parallel.tp_rank = parallel.world_rank = parallel.local_rank = rank
        parallel.ffn_tp_size, parallel.ffn_tp_rank = tp, rank
        builder = Qwen3DFlash2Weight(
            model_config=config,
            parallelism_config=parallel,
            hw_kernel_config=HWKernelConfig(),
            kv_cache_config=KVCacheConfig(),
        )
        builder.process_meta_from_ckpt(database.pretrain_file_list)
        aliases = {
            name: torch.empty(
                config.vocab_size // tp, config.hidden_size, dtype=torch.bfloat16
            )
            for name in (W.embedding, W.lm_head)
        }
        device = SimpleNamespace(
            maybe_rewrite_weight_by_key=lambda _name, tensor, **_kwargs: tensor
        )
        with patch(
            "rtp_llm.device.get_current_device", return_value=device
        ), patch.object(ModelLoader, "force_clean_cuda_memory"):
            loader = ModelLoader(
                config,
                builder,
                None,
                database,
                load_method=LoadMethod.SCRATCH,
                force_cpu_load_weights=True,
            )
            weights = loader.load_weights("cpu", global_weight_aliases=aliases)
        for name in aliases:
            self.assertIs(weights.global_weights[name], aliases[name])
        for runtime, ckpt in (
            (W.dflash2_selector_predecessor, "predecessor_codebook"),
            (W.dflash2_selector_successor, "successor_codebook"),
            (W.dflash2_selector_projection, "hidden_projection.weight"),
        ):
            torch.testing.assert_close(
                weights.global_weights[runtime],
                tensors["candidate_selector." + ckpt],
                atol=0,
                rtol=0,
            )
        for index, layer in enumerate(weights.weights):
            for name, base, kernel in (
                (
                    "attention_conv",
                    W.dflash2_attention_conv_base,
                    W.dflash2_attention_conv_kernel,
                ),
                ("mlp_conv", W.dflash2_mlp_conv_base, W.dflash2_mlp_conv_kernel),
            ):
                prefix = f"layers.{index}.{name}."
                torch.testing.assert_close(
                    layer[base], tensors[prefix + "base_kernel"], atol=0, rtol=0
                )
                torch.testing.assert_close(
                    layer[kernel],
                    tensors[prefix + "kernel_projection.weight"],
                    atol=0,
                    rtol=0,
                )


if __name__ == "__main__":
    unittest.main()

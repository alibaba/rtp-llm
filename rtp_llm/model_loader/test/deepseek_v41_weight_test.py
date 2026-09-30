"""Descriptor and named-reader integration; requires the native config library."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from rtp_llm.config.pp_layout import resolve_pp_partition
from rtp_llm.config.quant_config import Fp8BlockWiseQuantConfig
from rtp_llm.model_loader.load_config import LoadConfig, LoadMethod
from rtp_llm.model_loader.loader import ModelLoader
from rtp_llm.model_loader.per_block_fp8_quant_weight import V4PerBlockFp8Weight
from rtp_llm.models.deepseek_v4 import DeepSeekV4, DeepSeekV4Weight
from rtp_llm.models.deepseek_v41 import DeepSeekV41, DeepSeekV41Weight
from rtp_llm.ops import TaskType
from rtp_llm.utils.database import CkptDatabase
from rtp_llm.utils.model_weight import W


class V41WeightTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        fixture = (
            Path(__file__).parents[2] / "config/test/testdata/deepseek_v41_flash.json"
        )
        self.source = json.loads(fixture.read_text())
        (Path(self.directory.name) / "config.json").write_text(json.dumps(self.source))
        self.config = DeepSeekV41._create_config(self.directory.name)
        self.config.model_type = "deepseek_v41"

    def descriptor(self, ep_size=8, ep_rank=0):
        descriptor = object.__new__(DeepSeekV41Weight)
        descriptor.model_config = self.config
        descriptor._num_layers = self.config.num_layers
        descriptor._hidden_size = self.config.hidden_size
        descriptor._head_num = self.config.attn_config.head_num
        descriptor._head_num_kv = self.config.attn_config.kv_head_num
        descriptor._size_per_head = self.config.attn_config.size_per_head
        descriptor.expert_num_ = self.config.expert_num
        descriptor._moe_align_size = 64
        descriptor.enable_fp32_lm_head = False
        descriptor.ep_size, descriptor.ep_rank = ep_size, ep_rank
        descriptor._process_meta({}, [])
        return descriptor

    def test_nested_runtime_config_and_v4_defaults(self):
        config = self.config
        self.assertTrue(config.is_deepseek_v41)
        self.assertFalse(config.mm_model_config.is_multimodal)
        self.assertEqual(
            (config.num_layers, config.hidden_size, config.moe_inter_size),
            (40, 5120, 2304),
        )
        self.assertEqual(
            (config.expert_num, config.moe_k, config.num_hash_layers), (384, 6, 0)
        )
        self.assertEqual(config.attn_config.softmax_extra_scale, 1.0)
        self.assertEqual(len(config.attn_config.layer_compress_ratios), 40)
        self.assertEqual(resolve_pp_partition(40, 2, config), [21, 19])
        self.assertEqual(config.config_dtype, "bfloat16")
        config.attn_config.tokens_per_block = 64
        config.attn_config.kernel_tokens_per_block = 64
        with self.assertRaisesRegex(NotImplementedError, "serving requires"):
            DeepSeekV41._post_build_model_config(config)
        self.assertEqual(config.attn_config.tokens_per_block, 256)
        self.assertEqual(len(config.kv_cache_spec_descs), 40)
        # The optional nested-config entry point must not alter legacy file parsing.
        v4 = dict(self.source["text_config"])
        v4.update(
            torch_dtype="bfloat16", num_hash_layers=2, compress_ratios=[0, 0] + [4] * 38
        )
        (Path(self.directory.name) / "config.json").write_text(json.dumps(v4))
        legacy = DeepSeekV4._create_config(self.directory.name)
        self.assertFalse(legacy.is_deepseek_v41)
        self.assertEqual(legacy.num_hash_layers, 2)
        self.assertGreater(legacy.attn_config.softmax_extra_scale, 1.0)

    def test_source_only_descriptors_and_precision(self):
        descriptor = self.descriptor()
        info = descriptor._get_weight_info()
        self.assertEqual(
            {w.name for w in info.weights},
            {W.embedding, W.lm_head, W.final_ln_gamma, W.final_ln_beta},
        )
        for layer, weights in enumerate(info.layer_weights):
            by_name = {w.name: w for w in weights}
            self.assertEqual(W.v4_compressor_wkv in by_name, layer in (2, 8, 14, 20))
            self.assertEqual(W.v41_indexer_wk in by_name, layer in (2, 8, 14, 20))
            self.assertEqual(W.v4_compressor_wgate in by_name, layer in (2, 8, 14))
            self.assertEqual(
                W.v4_indexer_wq_b_w in by_name, layer in (2, 8, 14, 20, 24, 28, 32, 36)
            )
            self.assertNotIn(W.v4_router_tid2eid, by_name)
            self.assertNotIn(W.v4_compressor_ape, by_name)
            self.assertEqual(by_name[W.v4_hc_attn_fn].data_type, torch.float32)
            self.assertEqual(by_name[W.v4_router_bias].data_type, torch.float32)
            self.assertEqual(by_name[W.v4_routed_w1_w].data_type, torch.int8)
            self.assertEqual(by_name[W.v4_routed_w1_s].data_type, torch.float8_e8m0fnu)
            for weight in weights:
                self.assertFalse(
                    any(
                        "engram" in ckpt.name or "vision" in ckpt.name
                        for ckpt in weight.weights
                    )
                )
        legacy = object.__new__(DeepSeekV4Weight)
        legacy.__dict__.update(descriptor.__dict__)
        legacy._num_hash_layers = 2
        self.assertIn(
            W.v4_router_tid2eid, {w.name for w in legacy._get_hf_layer_weight_info(0)}
        )

    def test_fp8_group32_keeps_scale_descriptor(self):
        descriptor = self.descriptor()
        quant = Fp8BlockWiseQuantConfig(group_size=32, is_quanted=True)
        info = descriptor._get_weight_info().to_quant_weight_info(quant)
        weight = next(w for w in info.layer_weights[2] if w.name == W.v4_attn_wq_a_w)
        self.assertIsInstance(weight, V4PerBlockFp8Weight)
        self.assertEqual(weight.group_size, 32)
        self.assertEqual(weight.scale.data_type, torch.float8_e8m0fnu)
        self.assertEqual(weight.scale.weights[0].name, "layers.{i}.attn.wq_a.scale")

    def make_loader(self, stage, ep_size, ep_rank):
        pp = 2 if ep_size == 4 else 1
        config = LoadConfig(
            database=MagicMock(spec=CkptDatabase),
            num_layers=40,
            hidden_size=5120,
            head_num=64,
            head_num_kv=1,
            size_per_head=512,
            moe_pure_tp_mode=False,
            align_size=0,
            moe_align_size=64,
            moe_layer_index=list(range(40)),
            moe_n_group=0,
            expert_num=384,
            enable_eplb=False,
            phy_exp_num=384,
            tp_size=1,
            tp_rank=0,
            ep_size=ep_size,
            ep_rank=ep_rank,
            dp_size=ep_size if pp == 1 else 1,
            dp_rank=ep_rank if pp == 1 else 0,
            lm_head_tp_size=1,
            lm_head_tp_rank=0,
            ffn_tp_size=1,
            ffn_tp_rank=0,
            num_nodes=1,
            compute_dtype=torch.bfloat16,
            pp_size=pp,
            pp_rank=stage,
            pp_stage_layer_counts=[21, 19] if pp == 2 else [40],
        )
        loader = object.__new__(ModelLoader)
        loader._load_config = config
        loader._layer_ids = config.pp_layer_range()
        loader._model_weights_info = (
            self.descriptor(ep_size, ep_rank)
            ._get_weight_info()
            .to_quant_weight_info(
                Fp8BlockWiseQuantConfig(group_size=32, is_quanted=True)
            )
        )
        loader._misc_weights_info = []
        loader._global_weight_aliases = {}
        loader._task_type = TaskType.LANGUAGE_MODEL
        loader._load_embedding = config.has_pp_embedding
        loader._load_lm_head = config.has_pp_lm_head
        return loader

    def test_real_loader_name_map_respects_stage_and_expert_ownership(self):
        for stage, ep_size, ep_rank in ((0, 4, 3), (1, 4, 0), (0, 8, 7)):
            with self.subTest(stage=stage, ep_size=ep_size, ep_rank=ep_rank):
                loader = self.make_loader(stage, ep_size, ep_rank)
                names, _ = loader._generate_weight_info()
                layer_names = [n for n in names if n.startswith("layers.")]
                self.assertEqual(
                    {int(n.split(".")[1]) for n in layer_names}, set(loader._layer_ids)
                )
                expert_names = [n for n in layer_names if ".experts." in n]
                local_count = 384 // ep_size
                self.assertEqual(
                    {int(n.split(".")[4]) for n in expert_names},
                    set(range(ep_rank * local_count, (ep_rank + 1) * local_count)),
                )
                self.assertEqual("embed.weight" in names, stage == 0)
                self.assertEqual("head.weight" in names, ep_size == 8 or stage == 1)
                self.assertFalse(
                    any(
                        "engram" in n or "vision" in n or n.startswith("mtp.")
                        for n in names
                    )
                )
                if stage == 1:
                    self.assertNotIn("layers.20.attn.compressor.wkv.weight", names)

    def test_invalid_expert_parallelism_rejected(self):
        for size, rank in ((7, 0), (8, 8), (8, -1)):
            with self.assertRaises(ValueError):
                self.descriptor(size, rank)

    def test_v41_reader_policy_before_gpu_io(self):
        loader = object.__new__(ModelLoader)
        loader._weights_info = SimpleNamespace(model_config=self.config)
        for method in (LoadMethod.AUTO, LoadMethod.SCRATCH):
            loader._load_method = method
            with patch.object(
                ModelLoader, "_load_from_scratch", return_value="named"
            ) as named:
                self.assertEqual(loader._load_weight("cuda:0"), "named")
                named.assert_called_once_with("cuda:0")
        loader._load_method = LoadMethod.FASTSAFETENSORS
        with self.assertRaisesRegex(ValueError, "host Engram"):
            loader._load_weight("cuda:0")
        loader._weights_info.model_config = SimpleNamespace(is_deepseek_v41=False)
        with patch.object(
            ModelLoader, "_load_from_fastsafetensor", return_value="legacy"
        ):
            self.assertEqual(loader._load_weight("cuda:0"), "legacy")


if __name__ == "__main__":
    unittest.main()

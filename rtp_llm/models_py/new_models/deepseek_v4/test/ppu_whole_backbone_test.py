"""Real two-rank checkpoint layout for a SWA/CSA/HCA PPU backbone, on CPU."""

import itertools
import json
import os
import struct
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from safetensors.torch import save_file

from rtp_llm.models_py.checkpoint_reader_test_support import checkpoint_readers
from rtp_llm.models_py.fst_range_reader import iter_fst_range_tensors
from rtp_llm.models_py.model_loader import NewLoaderConfig, NewModelLoader
from rtp_llm.models_py.modules.dsv4.ppu.ppu_module_provider import PpuModuleProvider
from rtp_llm.models_py.modules.dsv4.ppu.transformer import PpuV4Transformer
from rtp_llm.models_py.new_models.deepseek_v4.ppu_backbone import PpuOwnedV4Transformer
from rtp_llm.models_py.weight_mapper import get_all_weights


class PpuWholeBackboneTest(unittest.TestCase):
    def test_release_tp4_full_flash_checkpoint_read_plan(self):
        from rtp_llm.models.deepseek_v4 import DeepSeekV4
        from rtp_llm.models_py.new_models.deepseek_v4.ppu_language import (
            PpuDeepSeekV4ForCausalLM,
        )
        from rtp_llm.ops import KvCacheDataType, MoeConfig, ParallelismConfig, RoleType

        path = os.getenv("PPU_FLASH_RELEASE_PATH")
        if not path:
            self.skipTest("requires the actual Flash-0731 release")
        root = Path(path)
        cfg = DeepSeekV4._create_config(path)
        cfg.model_type = "deepseek_v4"
        cfg.ckpt_path = path
        cfg.gen_num_per_cycle = 1
        cfg.attn_config.kv_cache_dtype = KvCacheDataType.FP8
        cfg.attn_config.kernel_tokens_per_block = cfg.attn_config.tokens_per_block = 256
        parallel = ParallelismConfig()
        parallel.world_size = parallel.local_world_size = parallel.tp_size = 4
        parallel.ffn_tp_size = 4
        parallel.local_rank = parallel.world_rank = parallel.tp_rank = (
            parallel.ffn_tp_rank
        ) = 3
        parallel.role_type = RoleType.PREFILL
        load = NewLoaderConfig.from_runtime(
            parallel,
            device="cpu",
            compute_dtype=torch.bfloat16,
            max_generate_batch_size=1,
            moe_config=MoeConfig(),
        )
        with torch.device("meta"):
            model = PpuDeepSeekV4ForCausalLM(cfg, load)
        self.assertEqual(len(model.v4.layers), 43)
        self.assertTrue(all(param.is_meta for param in model.parameters()))
        self.assertEqual(tuple(model.v4.embed.weight.shape), (cfg.vocab_size, 1024))
        loader = NewModelLoader(cfg, load, model_path=path)
        owns = model.checkpoint_weight_name_filter()
        resolve = loader._checkpoint_read_spec_resolver(model, owns, None)
        index = json.loads((root / "model.safetensors.index.json").read_bytes())[
            "weight_map"
        ]
        visited = set()
        for shard in set(index.values()):
            with (root / shard).open("rb") as source:
                (size,) = struct.unpack("<Q", source.read(8))
                header = json.loads(source.read(size))
            for name, info in header.items():
                if name == "__metadata__" or not owns(name):
                    continue
                self.assertEqual(shard, index[name])
                self.assertTrue(resolve(name, tuple(info["shape"])).load, name)
                visited.add(name)
        self.assertEqual(visited, {name for name in index if owns(name)})
        self.assertGreater(len(visited), 50000)

    def test_all_checkpoint_keys_swa_csa_hca_with_ep_and_tp(self):
        from rtp_llm.models_py.new_models.deepseek_v4.backbone_test import (
            Dsv4FullBackboneCheckpointTest,
        )

        fixture = Dsv4FullBackboneCheckpointTest()
        data = fixture.weights()
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "model.safetensors")
            save_file(data, path)
            for rank, reader in itertools.product(
                range(2), checkpoint_readers(get_all_weights, iter_fst_range_tensors)
            ):
                with self.subTest(rank=rank, backend=reader.__name__):
                    args = fixture.args(rank)
                    args.lm_head_tp_size = 2
                    args.lm_head_tp_rank = rank
                    model = PpuOwnedV4Transformer(
                        args, platform_provider=PpuModuleProvider({})
                    )
                    self.assertIs(
                        model.forward_decode.__func__, PpuV4Transformer.forward_decode
                    )
                    loader = NewModelLoader(
                        SimpleNamespace(model_type="deepseek_v4"),
                        NewLoaderConfig(
                            device="cpu",
                            tp_size=2,
                            tp_rank=rank,
                            ep_size=2,
                            ep_rank=rank,
                        ),
                        model_path=tmp,
                    )
                    plan = {}
                    accept = model.checkpoint_weight_name_filter()
                    resolve = loader._checkpoint_read_spec_resolver(
                        model, accept, None, plan
                    )
                    model._rtp_checkpoint_load_plan = plan
                    model.load_standalone_checkpoint(
                        reader([path], name_filter=accept, read_spec_resolver=resolve)
                    )
                    NewModelLoader._validate_loaded_weights(model)
                    torch.testing.assert_close(
                        model.embed.weight, data["embed.weight"].chunk(2, 1)[rank]
                    )
                    for i, layer in enumerate(model.layers):
                        root = f"layers.{i}."
                        torch.testing.assert_close(
                            layer.attn_hc.fn, data[root + "hc_attn_fn"]
                        )
                        torch.testing.assert_close(
                            layer.ffn_norm.weight, data[root + "ffn_norm.weight"]
                        )
                        torch.testing.assert_close(
                            layer.attn.wkv.weight.view(torch.uint8),
                            data[root + "attn.wkv.weight"].view(torch.uint8),
                        )
                        torch.testing.assert_close(
                            layer.attn.wq_b.scale.view(torch.uint8),
                            data[root + "attn.wq_b.scale"]
                            .chunk(2)[rank]
                            .view(torch.uint8),
                        )
                        self.assertEqual(layer.ffn.expert_start, rank * 2)
                    self.assertEqual(
                        model.layers[1].attn.indexer.compressor.norm.weight.dtype,
                        torch.float32,
                    )
                    with self.assertRaisesRegex(RuntimeError, "PPU|M890P"):
                        model.bind_runtime()


if __name__ == "__main__":
    unittest.main()

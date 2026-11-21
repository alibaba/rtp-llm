"""Opt-in release config/header planning audit; never reads tensor payloads."""

import json
import os
import struct
import unittest
from pathlib import Path

import torch

from rtp_llm.models.deepseek_v41 import DeepSeekV41DSpark
from rtp_llm.models_py.model_loader import NewLoaderConfig, NewModelLoader
from rtp_llm.models_py.new_models.deepseek_v41_dspark.language import (
    DeepSeekV41DSparkForCausalLM,
)
from rtp_llm.ops import KvCacheDataType, ParallelismConfig, RoleType


class ReleaseDraftMetadataTest(unittest.TestCase):
    @unittest.skipUnless(
        os.environ.get("V41_RELEASE_CHECKPOINT"), "requires opt-in release metadata"
    )
    def test_actual_release_tp4_ep4_proposal_and_commit_readspec_shapes(self):
        root = Path(os.environ["V41_RELEASE_CHECKPOINT"])
        index = json.loads((root / "model.safetensors.index.json").read_text())[
            "weight_map"
        ]
        headers = {}
        for commit in (False, True):
            config = DeepSeekV41DSpark._create_config(str(root))
            config.model_type, config.ckpt_path, config.data_type = (
                "deepseek_v41_dspark",
                str(root),
                "bf16",
            )
            config.gen_num_per_cycle = 3
            config.attn_config.kv_cache_dtype = KvCacheDataType.FP8
            parallel = ParallelismConfig()
            parallel.tp_size = parallel.ep_size = parallel.world_size = 4
            parallel.role_type = RoleType.PREFILL if commit else RoleType.DECODE
            load = NewLoaderConfig.from_runtime(
                parallel,
                device="cpu",
                compute_dtype=torch.bfloat16,
                load_method="scratch",
                max_generate_batch_size=2,
            )
            with torch.device("meta"):
                model = DeepSeekV41DSparkForCausalLM(config, load)
            model_filter = model.checkpoint_weight_name_filter()
            loader = NewModelLoader(config, load, model_path=str(root))
            expert_filter = loader._expert_filter()
            predicate = loader._checkpoint_name_filter(model_filter, expert_filter)
            plan = {}
            resolver = loader._checkpoint_read_spec_resolver(
                model, model_filter, expert_filter, plan
            )
            selected = {name: shard for name, shard in index.items() if predicate(name)}
            self.assertNotIn("mtp.2.confidence_head.proj.weight", selected)
            self.assertEqual(len(model.v4.layers), 3)
            if commit:
                self.assertEqual(len(selected), 14)
            else:
                self.assertGreater(sum(model_filter(name) for name in index), 1000)
            loaded = 0
            for name, shard in selected.items():
                if shard not in headers:
                    with (root / shard).open("rb") as stream:
                        size = struct.unpack("<Q", stream.read(8))[0]
                        if size > 64 * 1024 * 1024:
                            raise ValueError("Unexpected metadata size")
                        headers[shard] = json.loads(stream.read(size))
                spec = resolver(name, tuple(headers[shard][name]["shape"]))
                self.assertIn(loader._mapped_checkpoint_name(model, name), plan, name)
                self.assertTrue(spec.load, name)
                loaded += 1
            self.assertGreater(loaded, 0)
            self.assertEqual(tuple(model.main_proj.scale.shape), (160, 480))
            print(
                "RELEASE_DRAFT_HEADER_ONLY",
                "commit",
                commit,
                "selected",
                len(selected),
                "rank0_load",
                loaded,
                "headers",
                len(headers),
                flush=True,
            )


if __name__ == "__main__":
    unittest.main()

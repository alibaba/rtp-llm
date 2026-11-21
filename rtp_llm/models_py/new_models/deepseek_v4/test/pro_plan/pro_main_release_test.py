"""Plan the Pro release on the real typed main runtime without allocating its weights.

Requires the native main ABI. The standard SM103 golden remains the execution
test; this CPU test covers the all-gather CP, EP4, checkpoint dispatch contract.
"""

import json
import os
import struct
import unittest
from pathlib import Path

import torch


def _release_headers(root):
    from rtp_llm.models_py.weight_mapper import safetensor_catalog

    catalog = safetensor_catalog(str(root))
    if catalog is None:
        raise FileNotFoundError(f"No indexed Pro checkpoint at {root}")
    names = dict(catalog.entries)
    result = {}
    for shard in set(names.values()):
        with Path(shard).open("rb") as stream:
            nbytes = struct.unpack("<Q", stream.read(8))[0]
            if not 0 < nbytes <= 100_000_000:
                raise ValueError(f"Invalid header length {nbytes} for {shard}")
            header = json.loads(stream.read(nbytes))
        for key, info in header.items():
            if key == "__metadata__":
                continue
            if key in result or names.get(key) != shard:
                raise ValueError(f"Inconsistent Pro checkpoint index entry for {key}")
            result[key] = tuple(info["shape"])
    if names.keys() != result.keys():
        raise ValueError("Pro checkpoint index and headers disagree")
    return result


class V4ProMainReleasePlanTest(unittest.TestCase):
    def test_all_headers_contain_pro_widths_and_expert_geometry(self):
        root = Path(os.environ["DSV4_PRO_SOURCE"])
        headers = _release_headers(root)
        for i in range(61):
            p = f"layers.{i}."
            self.assertEqual(headers[p + "attn.wq_b.weight"], (65536, 1536))
            self.assertEqual(headers[p + "attn.wo_a.weight"], (16384, 4096))
            for j in range(384):
                for proj in ("w1", "w3", "w2"):
                    rows, cols = (7168, 3072) if proj == "w2" else (3072, 7168)
                    prefix = p + f"ffn.experts.{j}.{proj}."
                    self.assertEqual(headers[prefix + "weight"], (rows, cols // 2))
                    self.assertEqual(headers[prefix + "scale"], (rows, cols // 32))

    def test_typed_pro_cp4_ep4_resolves_the_complete_target_plan(self):
        from rtp_llm.models.deepseek_v4 import DeepSeekV4
        from rtp_llm.models_py.model_loader import NewLoaderConfig, NewModelLoader
        from rtp_llm.models_py.new_models.deepseek_v4.language import (
            DeepSeekV4ForCausalLM,
        )
        from rtp_llm.ops import (
            CPRotateMethod,
            KvCacheDataType,
            MoeConfig,
            ParallelismConfig,
            PrefillCPConfig,
        )

        root = Path(os.environ["DSV4_PRO_SOURCE"])
        config = DeepSeekV4._create_config(str(root))
        config.model_type = "deepseek_v4"
        config.ckpt_path = str(root)
        config.max_seq_len = 8192
        config.gen_num_per_cycle = 1
        config.attn_config.kv_cache_dtype = KvCacheDataType.FP8
        self.assertEqual((config.num_layers, config.expert_num), (61, 384))
        self.assertEqual(config.attn_config.head_num, 128)
        headers = _release_headers(root)
        parallel = ParallelismConfig()
        parallel.world_size = parallel.local_world_size = parallel.tp_size = (
            parallel.ep_size
        ) = 4
        parallel.ffn_tp_size = 4
        parallel.prefill_cp_config = PrefillCPConfig()
        parallel.prefill_cp_config.method = CPRotateMethod.ALL_GATHER
        moe = MoeConfig()
        moe.moe_strategy = "mega_moe_se"
        for rank in range(4):
            parallel.world_rank = parallel.local_rank = parallel.tp_rank = (
                parallel.ep_rank
            ) = rank
            parallel.ffn_tp_rank = rank
            load = NewLoaderConfig(
                device="cpu",
                compute_dtype=torch.bfloat16,
                tp_size=4,
                tp_rank=rank,
                ep_size=4,
                ep_rank=rank,
                max_generate_batch_size=1,
                attn_tp_size=parallel.get_attn_tp_size(),
                attn_tp_rank=parallel.get_attn_tp_rank(),
                ffn_tp_size=parallel.get_ffn_tp_size(),
                ffn_tp_rank=parallel.get_ffn_tp_rank(),
                lm_head_tp_size=4,
                lm_head_tp_rank=rank,
                parallelism_config=parallel,
                moe_config=moe,
            )
            with torch.device("meta"):
                model = DeepSeekV4ForCausalLM(config, load)
            self.assertTrue(all(p.is_meta for p in model.parameters()))
            self.assertEqual((model.v4.args.tp_size, model.v4.args.tp_rank), (1, 0))
            self.assertEqual(model.v4.head.weight.shape, (config.vocab_size // 4, 7168))
            self.assertEqual(
                tuple(model.v4.embed.weight.shape), (config.vocab_size, 7168)
            )
            self.assertEqual(model.v4.args.q_lora_rank, 1536)
            self.assertEqual(model.v4.args.o_groups, 16)
            self.assertEqual(model.v4.args.moe_inter_dim, 3072)
            self.assertEqual(model.v4.args.moe_strategy, "mega_moe_se")
            self.assertEqual(len(model.v4.layers), 61)
            for block in model.v4.layers:
                self.assertEqual(
                    (block.ffn.expert_start, block.ffn.n_local_experts), (rank * 96, 96)
                )
            loader = NewModelLoader(config, load, model_path=str(root))
            owns = model.checkpoint_weight_name_filter()
            resolved = loader._checkpoint_read_spec_resolver(
                model, owns, loader._expert_filter()
            )
            routed = 0
            for name, shape in headers.items():
                spec = resolved(name, shape)
                if name.startswith("mtp."):
                    self.assertFalse(spec.load, name)
                    continue
                self.assertTrue(owns(name), name)
                target, _ = model.resolve_checkpoint_load_plan(
                    model.WEIGHTS_MAPPER.map_name(name), shape
                )
                self.assertIsNotNone(target, name)
                if ".ffn.experts." in name:
                    expert = int(name.split(".ffn.experts.")[1].split(".", 1)[0])
                    self.assertEqual(spec.load, expert // 96 == rank, name)
                    routed += int(spec.load)
                else:
                    self.assertTrue(spec.load, name)
            self.assertEqual(routed, 61 * 96 * 6)


if __name__ == "__main__":
    unittest.main()

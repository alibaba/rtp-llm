import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from safetensors.torch import save_file

from rtp_llm.config.qwen4_exp_config import (
    resolve_qwen4_parallelism,
    uses_fp8_ple_table,
)
from rtp_llm.model_loader.ffn_weight import MoeConfig
from rtp_llm.models.qwen4_exp.qwen4_exp import Qwen4Exp
from rtp_llm.models.qwen4_exp.qwen4_exp_weight import Qwen4ExpWeight, _shared_ple_table
from rtp_llm.models.qwen4_exp.test.config_fixture import qwen4_config
from rtp_llm.models_py.modules.factory.fused_moe.defs.config_adapter import (
    MoEConfigAdapter,
)
from rtp_llm.ops import MoeConfig as RuntimeMoeConfig
from rtp_llm.ops import ParallelismConfig
from rtp_llm.utils.database import CkptDatabase
from rtp_llm.utils.model_weight import W
from rtp_llm.utils.qwen4_ple import (
    resolve_shared_ple_table,
    stage_qwen4_ple_for_server,
)


class Qwen4Mxfp4Test(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.quant = {
            "quant_method": "mxfp4",
            "ple_quant_scheme": "fp8_pertensor",
        }

    def test_precision_initialization_keeps_dense_bf16_and_fp8_ple(self):
        config = qwen4_config()
        config["quantization_config"] = self.quant
        (self.root / "config.json").write_text(json.dumps(config))
        runtime = Qwen4Exp.create_config(str(self.root))
        runtime.init_precision_config(None, "BF16")
        Qwen4Exp._post_build_model_config(runtime)
        self.assertTrue(runtime._qwen4_mxfp4_source)
        self.assertFalse(runtime._qwen4_fp8_source)
        self.assertIsNone(runtime.quant_config)
        self.assertEqual(runtime.dense_weight_dtype, "bf16")
        self.assertEqual(runtime.expert_weight_dtype, "fp4")
        self.assertTrue(uses_fp8_ple_table(runtime))

        parallelism = ParallelismConfig()
        parallelism.world_size = parallelism.tp_size = parallelism.ep_size = 2
        adapter = MoEConfigAdapter(
            runtime, parallelism, RuntimeMoeConfig(), runtime.quant_config
        )
        self.assertEqual(adapter.moe_quant_method, "FP8_FP4")

    def test_cross_node_dp_ep_preserves_local_rank(self):
        config = qwen4_config()
        config["quantization_config"] = self.quant
        (self.root / "config.json").write_text(json.dumps(config))
        runtime = Qwen4Exp.create_config(str(self.root))
        parallelism = ParallelismConfig()
        parallelism.tp_size = 1
        parallelism.world_size = parallelism.dp_size = parallelism.ep_size = 16
        parallelism.local_world_size = 4
        parallelism.local_rank = 3
        resolved = resolve_qwen4_parallelism(runtime, parallelism)
        self.assertEqual(
            (resolved.dp_size, resolved.ep_size, resolved.local_rank), (16, 16, 3)
        )
        self.assertEqual((resolved.ffn_tp_size, resolved.ffn_tp_rank), (1, 0))

    @unittest.skipUnless(sys.platform == "linux", "requires Linux memfd")
    def test_shared_ple_derives_shard_offsets_from_checkpoint(self):
        config_path = self.root / "config.json"
        config_path.write_text(json.dumps({"quantization_config": self.quant}))
        weight_map = {}
        for part in range(2):
            tensors = {}
            filename = f"ple-{part}.safetensors"
            for shard in reversed(range(part * 64, (part + 1) * 64)):
                name = (
                    "model.language_model.layers.0.ple.ple_embedding.ngram_embedding."
                    f"shard_{shard}.weight"
                )
                tensors[name] = torch.full((2, 2), shard, dtype=torch.uint8).view(
                    torch.float8_e4m3fn
                )
                weight_map[name] = filename
            save_file(tensors, self.root / filename)
        (self.root / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": weight_map})
        )
        checkpoint_files = set(self.root.iterdir())
        model_args = SimpleNamespace(model_type="qwen4_exp", ckpt_path=str(self.root))
        self.addCleanup(_shared_ple_table.cache_clear)
        with (
            patch("rtp_llm.utils.qwen4_ple._check_staging_memory") as check_memory,
            stage_qwen4_ple_for_server(
                model_args, shutdown_requested=lambda: False
            ) as path,
        ):
            check_memory.assert_called_once_with(128 * 4)
            assert path is not None
            table, entries = _shared_ple_table(path, str(self.root))
            self.assertTrue(table.untyped_storage()._qwen4_ple_shared)
            self.assertEqual(set(entries), set(weight_map))
            for name, (offset, length, shape) in entries.items():
                shard = int(name.rsplit("_", 1)[-1].split(".")[0])
                self.assertEqual(shape, [2, 2])
                self.assertEqual(table[offset : offset + length].tolist(), [shard] * 4)
            self.assertEqual(set(self.root.iterdir()), checkpoint_files)
            config_path.write_text("{}")
            with self.assertRaisesRegex(ValueError, "table does not match checkpoint"):
                resolve_shared_ple_table(path, self.root)

    def test_layer_loading_releases_safetensors_handle_and_keeps_tensor_view(self):
        expected = torch.arange(4, dtype=torch.bfloat16)
        save_file({"weight": expected}, self.root / "model.safetensors")
        database = CkptDatabase(
            str(self.root), release_safetensor_handles_per_layer=True
        )
        shard = database.pretrain_file_list[0]
        self.addCleanup(shard.close_safetensor_handle)
        view = database.load_tensor("weight", expected.dtype)[0]
        self.assertIsNotNone(shard._st_handle)

        database.finish_loading_layer()

        self.assertIsNone(shard._st_handle)
        torch.testing.assert_close(view, expected)
        torch.testing.assert_close(
            database.load_tensor("weight", expected.dtype)[0], expected
        )

    def test_weight_mapping_preserves_bytes_and_gate_up_order(self):
        weight_info = Qwen4ExpWeight.__new__(Qwen4ExpWeight)
        weight_info.prefix = "model.language_model."
        weight_info.model_config = SimpleNamespace(_qwen4_mxfp4_source=True)
        weights = {
            weight.name: weight
            for weight in weight_info._create_moe_expert_weights(
                MoeConfig(expert_num=1)
            )
        }
        self.assertEqual(
            weights[W.moe_w1].get_ckpt_tensor_names(),
            [
                f"model.language_model.layers.{{i}}.mlp.experts.{{expert_id}}.{proj}_proj.weight"
                for proj in ("gate", "up")
            ],
        )
        packed = torch.tensor([0x12, 0xAB], dtype=torch.uint8)
        for key, dtype in (
            (W.moe_w1, torch.int8),
            (W.moe_s1, torch.float8_e8m0fnu),
        ):
            view = weights[key]._postprocess(packed, "cpu", None)[key]
            self.assertEqual(view.dtype, dtype)
            torch.testing.assert_close(view.view(torch.uint8), packed)


if __name__ == "__main__":
    unittest.main()

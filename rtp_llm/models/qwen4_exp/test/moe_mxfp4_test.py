import hashlib
import json
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
        with patch.dict("os.environ", {"QWEN4_PLE_SHARED_TABLE": "/ple.json"}):
            resolved = resolve_qwen4_parallelism(runtime, parallelism)
        self.assertEqual(
            (resolved.dp_size, resolved.ep_size, resolved.local_rank), (16, 16, 3)
        )
        self.assertEqual((resolved.ffn_tp_size, resolved.ffn_tp_rank), (1, 0))

    def test_shared_ple_manifest_matches_the_checkpoint(self):
        config = b"config"
        index = b"index"
        (self.root / "config.json").write_bytes(config)
        (self.root / "model.safetensors.index.json").write_bytes(index)
        table_path = self.root / "table"
        table_path.write_bytes(b"\x12")
        manifest = self.root / "ple.json"
        data = {
            "path": str(table_path),
            "entries": {},
            "checkpoint_config_sha256": hashlib.sha256(config).hexdigest(),
            "checkpoint_index_sha256": hashlib.sha256(index).hexdigest(),
        }
        self.addCleanup(_shared_ple_table.cache_clear)
        for key in ("checkpoint_config_sha256", "checkpoint_index_sha256"):
            manifest.write_text(json.dumps({**data, key: "stale"}))
            with self.assertRaisesRegex(
                ValueError, "shared PLE manifest does not match"
            ):
                _shared_ple_table(str(manifest), str(self.root))

        manifest.write_text(json.dumps(data))
        table, _ = _shared_ple_table(str(manifest), str(self.root))
        self.assertEqual(table.tolist(), [0x12])

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

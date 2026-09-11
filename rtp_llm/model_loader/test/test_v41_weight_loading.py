"""Exercise V4.1 descriptors through the actual loader and real checkpoint.

These are component checks, not distributed-model or numerical acceptance.
"""

import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from safetensors.torch import save_file

from rtp_llm.config.dsv41_weights import V41TensorSpec
from rtp_llm.config.kv_cache_config import KVCacheConfig
from rtp_llm.device.device_base import DeviceBase
from rtp_llm.model_loader.load_config import LoadMethod
from rtp_llm.model_loader.loader import ModelLoader
from rtp_llm.model_loader.model_weight_info import ModelDeployWeightInfo
from rtp_llm.model_loader.tensor_source import DatabaseTensorSource, TensorCollector
from rtp_llm.models.deepseek_v41 import DeepSeekV41, DeepSeekV41Weight, V41AtomicWeight
from rtp_llm.models_py.modules.dsv41.compressor import OwnerCompressor
from rtp_llm.ops import CPRotateMethod, HWKernelConfig, ParallelismConfig
from rtp_llm.utils.database import CkptDatabase
from rtp_llm.utils.model_weight import W, identity


class V41WeightLoadingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.checkpoint = Path(os.environ["DSV41_MODEL_PATH"])
        if not torch.cuda.is_available():
            raise RuntimeError("V4.1 loader checks require a real development GPU")
        cls.config = DeepSeekV41._create_config(str(cls.checkpoint))
        cls.database = CkptDatabase(str(cls.checkpoint))
        cls.records = []

    @classmethod
    def tearDownClass(cls):
        output = os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR")
        if output:
            Path(output, "v41_weight_loading.json").write_text(
                json.dumps({"tensors": cls.records}, indent=2) + "\n",
                encoding="utf-8",
            )

    def deployment(self, ep_size=8, rank=0, cp=False):
        parallel = ParallelismConfig()
        parallel.world_size = parallel.ep_size = ep_size
        parallel.local_world_size = 4 if ep_size == 16 else 8
        parallel.world_rank = parallel.ep_rank = rank
        parallel.local_rank = rank % parallel.local_world_size
        if cp:
            parallel.tp_size = 8
            parallel.tp_rank = rank
            parallel.prefill_cp_config.method = CPRotateMethod.ALL_GATHER
            parallel.prefill_cp_config.prefill_cp_size = 8
        else:
            parallel.dp_size = ep_size
            parallel.dp_rank = rank
        deploy = DeepSeekV41Weight(
            self.config, parallel, HWKernelConfig(), KVCacheConfig()
        )
        load = deploy.create_load_config(
            torch.bfloat16, self.database, exported_device=DeviceBase()
        )
        self.assertEqual(load.tp_size, 1)
        return deploy, deploy.get_weight_info(), load

    def load_real(self, descriptor, layer, load):
        result = descriptor.load(
            DatabaseTensorSource(self.database), layer, "cuda", load
        )[descriptor.name]
        raw = result.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
        self.records.append(
            {
                "name": descriptor.name,
                "layer": layer,
                "ep_size": load.ep_size,
                "ep_rank": load.ep_rank,
                "dtype": str(result.dtype),
                "shape": list(result.shape),
                "sha256": hashlib.sha256(raw).hexdigest(),
            }
        )
        return result

    def test_all_profile_expert_descriptors_and_host_exclusion(self):
        for ep, cp in ((8, True), (8, False), (16, False)):
            for rank in (0, ep - 1):
                with self.subTest(ep=ep, cp=cp, rank=rank):
                    deploy, info, load = self.deployment(ep, rank, cp)
                    local = 384 // ep
                    for layer in (0, 14, 39):
                        names = {
                            ckpt.name
                            for weight in info.layer_weights[layer]
                            for ckpt in weight.weights
                        }
                        experts = {
                            int(name.split(".")[4])
                            for name in names
                            if ".ffn.experts." in name
                        }
                        self.assertEqual(
                            experts, set(range(rank * local, (rank + 1) * local))
                        )
                    ordinary = [
                        *info.weights,
                        *(w for layer in info.layer_weights for w in layer),
                    ]
                    loaded_keys = {
                        ckpt.name for weight in ordinary for ckpt in weight.weights
                    }
                    self.assertFalse(loaded_keys.intersection(deploy.host_weight_specs))
                    descriptor = info.get_layer_weight_info(14, "v41.hc_attn_scale")
                    actual = self.load_real(descriptor, 14, load)
                    expected = self.database.load_tensor(
                        "layers.14.hc_attn_scale", torch.float32
                    )[0]
                    self.assertEqual(actual.shape, (3,))
                    torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)

    def test_real_dense_and_expert_payloads_keep_checkpoint_bytes(self):
        _, info, load = self.deployment(16, 15)
        for key, dtype in (
            ("attn.wq_a.weight", torch.float8_e4m3fn),
            ("attn.wq_a.scale", torch.float8_e8m0fnu),
            ("ffn.experts.383.w1.weight", torch.int8),
            ("ffn.experts.383.w1.scale", torch.float8_e8m0fnu),
        ):
            with self.subTest(key=key):
                actual = self.load_real(
                    info.get_layer_weight_info(0, "v41." + key), 0, load
                )
                expected = self.database.load_tensor("layers.0." + key, dtype)[0]
                self.assertEqual(actual.dtype, dtype)
                torch.testing.assert_close(
                    actual.cpu().view(torch.uint8),
                    expected.view(torch.uint8),
                    rtol=0,
                    atol=0,
                )

    def test_real_ratio_specific_compressor_installation(self):
        _, info, load = self.deployment()
        for layer in (2, 8, 14, 20):
            with self.subTest(layer=layer):

                def loaded(key):
                    return self.load_real(
                        info.get_layer_weight_info(layer, "v41.attn.compressor." + key),
                        layer,
                        load,
                    )

                wkv = loaded("wkv.weight")
                wgate = loaded("wgate.weight") if layer != 20 else None
                module = OwnerCompressor(layer, wkv, loaded("norm.weight"), wgate)
                expected = self.database.load_tensor(
                    f"layers.{layer}.attn.compressor.wkv.weight", torch.bfloat16
                )[0]
                self.assertEqual(
                    module.wkv.dtype, torch.bfloat16 if layer == 20 else torch.float32
                )
                torch.testing.assert_close(
                    module.wkv.cpu().float(), expected.float(), rtol=0, atol=0
                )

    def test_real_wo_a_decodes_both_checkpoint_dtypes_before_bf16(self):
        _, info, load = self.deployment()
        descriptor = info.get_layer_weight_info(0, "v41.attn.wo_a.weight")
        actual = self.load_real(descriptor, 0, load)
        raw = self.database.load_tensor(
            "layers.0.attn.wo_a.weight", torch.float8_e4m3fn
        )[0]
        scale = self.database.load_tensor(
            "layers.0.attn.wo_a.scale", torch.float8_e8m0fnu
        )[0].float()
        expected = (
            raw.float() * scale.repeat_interleave(32, 0).repeat_interleave(32, 1)
        ).bfloat16()
        self.assertEqual(actual.dtype, torch.bfloat16)
        torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)

    def test_cp_lm_head_keeps_vocab_partition(self):
        _, info, load = self.deployment(rank=7, cp=True)
        head = next(weight for weight in info.weights if weight.name == W.lm_head)
        raw = torch.arange(64 * 4, dtype=torch.float32).view(64, 4)
        actual = head._split(raw, load)[W.lm_head]
        torch.testing.assert_close(actual, raw[56:64], rtol=0, atol=0)
        embedding = next(
            weight for weight in info.weights if weight.name == W.embedding
        )
        torch.testing.assert_close(
            embedding._split(raw, load)[W.embedding], raw, rtol=0, atol=0
        )

    def test_database_and_collector_preserve_vector_scale_shape(self):
        _, _, load = self.deployment()
        with tempfile.TemporaryDirectory() as directory:
            original = torch.tensor([0.25, 1.0, 2.0], dtype=torch.float32)
            save_file(
                {"layers.0.hc_ffn_scale": original},
                str(Path(directory, "model.safetensors")),
            )
            database = CkptDatabase(directory)
            collector = TensorCollector({"layers.0.hc_ffn_scale"}, database)
            collector.store_tensor("layers.0.hc_ffn_scale", original)
            descriptor = V41AtomicWeight(
                "v41.hc_ffn_scale",
                [V41TensorSpec("layers.0.hc_ffn_scale", (3,), "F32")],
                identity,
                torch.float32,
            )
            for source in (DatabaseTensorSource(database), collector):
                actual = descriptor.load(source, 0, "cuda", load)[descriptor.name]
                self.assertEqual(actual.shape, (3,))
                torch.testing.assert_close(actual.cpu(), original, rtol=0, atol=0)

    def test_host_table_and_invalid_shape_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "host-shared"):
            V41AtomicWeight(
                "v41.engram.embed.weight",
                [
                    V41TensorSpec(
                        "layers.1.engram.embed.weight",
                        (1, 256),
                        "F8_E4M3",
                        "host_shared",
                    )
                ],
                identity,
                torch.float8_e4m3fn,
            )
        _, _, load = self.deployment()
        collector = TensorCollector({"bad"}, self.database)
        collector.store_tensor("bad", torch.ones(3, 1))
        descriptor = V41AtomicWeight(
            "v41.bad", [V41TensorSpec("bad", (3,), "F32")], identity, torch.float32
        )
        with self.assertRaisesRegex(ValueError, "shape or dtype"):
            descriptor.load(collector, 0, "cuda", load)
        collector.clear()
        collector.store_tensor("bad", torch.ones(3, dtype=torch.bfloat16))
        with self.assertRaisesRegex(ValueError, "shape or dtype"):
            descriptor.load(collector, 0, "cuda", load)

    def test_host_weight_policy_blocks_whole_shard_staging(self):
        deploy, _, _ = self.deployment()
        loader = ModelLoader.__new__(ModelLoader)
        loader._weights_info = deploy
        loader._load_method = LoadMethod.AUTO
        with patch.object(
            loader, "_load_from_scratch", return_value="selective"
        ) as scratch, patch.object(loader, "_load_from_fastsafetensor") as fast:
            self.assertEqual(loader._load_weight("cuda"), "selective")
            scratch.assert_called_once_with("cuda")
            fast.assert_not_called()
        for mode in (LoadMethod.FASTSAFETENSORS, "FASTSAFETENSORS", "FastSafeTensors"):
            loader._load_method = mode
            with self.subTest(mode=mode), patch.object(
                loader, "_load_from_fastsafetensor"
            ) as fast:
                with self.assertRaisesRegex(ValueError, "selective tensor loading"):
                    loader._load_weight("cuda")
                fast.assert_not_called()
        self.assertTrue(ModelDeployWeightInfo.supports_fastsafetensors)
        loader._weights_info = ModelDeployWeightInfo.__new__(ModelDeployWeightInfo)
        with patch.object(loader, "_load_from_fastsafetensor", return_value="fast"):
            self.assertEqual(loader._load_weight("cuda"), "fast")


if __name__ == "__main__":
    unittest.main()

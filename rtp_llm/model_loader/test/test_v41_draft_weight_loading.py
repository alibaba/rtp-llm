"""Actual checkpoint descriptor loading for the three P commit-only stages."""

import hashlib
import json
import os
import unittest
from pathlib import Path

import torch

from rtp_llm.config.kv_cache_config import KVCacheConfig
from rtp_llm.device.device_base import DeviceBase
from rtp_llm.model_loader.model_weight_info import ModelWeights
from rtp_llm.model_loader.tensor_source import DatabaseTensorSource
from rtp_llm.models.deepseek_v41 import DeepSeekV41, DeepSeekV41PrefillDraftWeight
from rtp_llm.models_py.modules.dsv41.cache_layout import CacheLayout, CacheRegion
from rtp_llm.models_py.modules.dsv41.ced import AuxRowMap, ReplayConfig, ReplayMode
from rtp_llm.models_py.modules.dsv41.compact_reader import CompactPages, SwaBinding
from rtp_llm.models_py.modules.dsv41.draft import V41PrefillDraftCommit
from rtp_llm.models_py.modules.dsv41.linear import is_supported
from rtp_llm.ops import CPRotateMethod, HWKernelConfig, ParallelismConfig
from rtp_llm.utils.database import CkptDatabase
from rtp_llm.utils.model_weight import W


class PrefillDraftWeightLoadingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available() or not is_supported(
            torch.empty(0, device="cuda")
        ):
            raise RuntimeError("actual V4.1 draft loading requires CUDA13/SM100")
        cls.checkpoint = Path(os.environ["DSV41_MODEL_PATH"])
        cls.config = DeepSeekV41._create_config(str(cls.checkpoint))
        cls.database = CkptDatabase(str(cls.checkpoint))
        cls.records = []

    @classmethod
    def tearDownClass(cls):
        output = os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR")
        if output:
            Path(output, "v41_prefill_draft_weights.json").write_text(
                json.dumps({"tensors": cls.records}, indent=2) + "\n",
                encoding="utf-8",
            )

    def deployment(self, rank=0):
        parallel = ParallelismConfig()
        parallel.world_size = parallel.ep_size = parallel.tp_size = 8
        parallel.local_world_size = 4
        parallel.world_rank = parallel.ep_rank = parallel.tp_rank = rank
        parallel.local_rank = rank % 4
        parallel.prefill_cp_config.method = CPRotateMethod.ALL_GATHER
        parallel.prefill_cp_config.prefill_cp_size = 8
        deploy = DeepSeekV41PrefillDraftWeight(
            self.config, parallel, HWKernelConfig(), KVCacheConfig()
        )
        load = deploy.create_load_config(
            torch.bfloat16, self.database, exported_device=DeviceBase()
        )
        return deploy, deploy.get_weight_info(), load

    def test_descriptor_contains_only_shared_aliases_and_commit_projections(self):
        for rank in (0, 7):
            with self.subTest(rank=rank):
                deploy, info, load = self.deployment(rank)
                self.assertEqual(load.num_layers, 3)
                self.assertEqual(load.tp_size, 1)
                self.assertEqual(load.moe_layer_index, [])
                self.assertFalse(load.enable_eplb)
                self.assertEqual(deploy.host_weight_specs, {})
                self.assertEqual(
                    {weight.name for weight in info.weights},
                    {
                        W.embedding,
                        W.lm_head,
                        "v41.mtp.0.main_proj.weight",
                        "v41.mtp.0.main_proj.scale",
                        "v41.mtp.0.main_norm.weight",
                    },
                )
                expected = {
                    "v41.attn.wkv.weight",
                    "v41.attn.wkv.scale",
                    "v41.attn.kv_norm.weight",
                }
                self.assertEqual(len(info.layer_weights), 3)
                for stage, layer in enumerate(info.layer_weights):
                    self.assertEqual({weight.name for weight in layer}, expected)
                    self.assertEqual(
                        {ckpt.name for weight in layer for ckpt in weight.weights},
                        {
                            f"mtp.{stage}." + name.removeprefix("v41.")
                            for name in expected
                        },
                    )

    def test_real_payload_installation_and_selected_checked_commit(self):
        _, info, load = self.deployment()
        installed = ModelWeights(3, "cuda", torch.bfloat16)
        source = DatabaseTensorSource(self.database)
        descriptors = [
            (None, descriptor)
            for descriptor in info.weights
            if descriptor.name not in (W.embedding, W.lm_head)
        ] + [
            (stage, descriptor)
            for stage, layer in enumerate(info.layer_weights)
            for descriptor in layer
        ]
        self.assertEqual(len(descriptors), 12)
        for stage, descriptor in descriptors:
            tensor = descriptor.load(source, stage or 0, "cuda", load)[descriptor.name]
            expected = self.database.load_tensor(
                descriptor.weights[0].name, tensor.dtype
            )[0]
            actual_bytes = tensor.detach().cpu().contiguous().view(torch.uint8)
            expected_bytes = expected.contiguous().view(torch.uint8)
            torch.testing.assert_close(actual_bytes, expected_bytes, rtol=0, atol=0)
            self.records.append(
                {
                    "checkpoint_name": descriptor.weights[0].name,
                    "installed_name": descriptor.name,
                    "stage": stage,
                    "shape": list(tensor.shape),
                    "dtype": str(tensor.dtype),
                    "sha256": hashlib.sha256(
                        actual_bytes.numpy().tobytes()
                    ).hexdigest(),
                }
            )
            if stage is None:
                installed.set_global_weight(descriptor.name, tensor)
            else:
                installed.set_layer_weight(stage, descriptor.name, tensor)
        module = V41PrefillDraftCommit.from_model_weights(
            self.config.dsv41_config, installed
        )
        self.assertIs(
            module.main_projection.weight,
            installed.global_weights["v41.mtp.0.main_proj.weight"],
        )
        for stage in range(3):
            self.assertIs(
                module.stage_projections[stage].weight,
                installed.weights[stage]["v41.attn.wkv.weight"],
            )
        generator = torch.Generator().manual_seed(4171)
        aux = torch.randn((7, 15360), generator=generator).bfloat16().cuda()
        row_map = AuxRowMap(
            "real-weight-component",
            1,
            ReplayConfig(ReplayMode.BOUNDED).fingerprint,
            tuple(range(1048448, 1048455)),
            tuple(range(7)),
            (False, True, False, True, False, False, True),
        )
        layout = CacheLayout()
        bindings = {}
        for page in layout.pages:
            if page.slot.region != CacheRegion.SWA or page.slot.owner_layer < 40:
                continue
            data = torch.full(
                (2, page.page_stride_bytes), 0x5A, dtype=torch.uint8, device="cuda"
            )
            meta = torch.tensor([0], dtype=torch.int32, device="cuda")
            bindings[page.slot.owner_layer] = SwaBinding(
                CompactPages(data, CacheRegion.SWA, page.entries),
                meta + 1,
                meta.clone(),
                meta.clone(),
            )
        required = (1048448, 1048451, 1048454)
        result = module.commit(
            aux,
            row_map,
            required_positions=required,
            swa_bindings=bindings,
            request_id=row_map.request_id,
            forward_epoch=1,
            replay_fingerprint=row_map.replay_fingerprint,
            replay_floor=1048448,
        )
        self.assertEqual(result.main_projection_rows, 3)
        self.assertEqual(result.stage_projection_rows, (3, 3, 3))
        self.assertEqual(result.positions, required)
        self.assertTrue(result.write_completed)
        for stage, write in zip(range(40, 43), result.writes):
            self.assertTrue((write.status == 0).all().item())
            self.assertTrue((bindings[stage].pages.data[0] == 0x5A).all().item())
            rows = bindings[stage].pages.data[1, : 136 * 528].view(136, 528)
            selected = torch.tensor(required, device="cuda") % 136
            self.assertTrue((rows[selected] != 0x5A).any().item())
            untouched = torch.ones(136, dtype=torch.bool, device="cuda")
            untouched[selected] = False
            self.assertTrue((rows[untouched] == 0x5A).all().item())


if __name__ == "__main__":
    unittest.main()

"""Real GPU checks for P-only draft projection and compact history writes."""

import os
import unittest
from unittest.mock import patch

import torch
from torch import nn

from rtp_llm.config.dsv41_config import V41Config
from rtp_llm.models_py.modules.dsv41.cache_layout import CacheRegion
from rtp_llm.models_py.modules.dsv41.ced import AuxRowMap, ReplayConfig, ReplayMode
from rtp_llm.models_py.modules.dsv41.compact_reader import SwaBinding
from rtp_llm.models_py.modules.dsv41.draft import V41PrefillDraftCommit
from rtp_llm.models_py.modules.dsv41.linear import is_supported
from rtp_llm.models_py.modules.dsv41.test.fixture import flash_config
from rtp_llm.models_py.modules.dsv41.test.official_compressor import (
    load_official_compressor,
)
from rtp_llm.models_py.modules.dsv41.test.test_compact_writer import (
    _load_official_kernel,
    _official_gpu,
    _pages,
)


class PrefillDraftGpuTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.device = torch.device(os.environ.get("DSV41_TEST_DEVICE", "cuda"))
        if not torch.cuda.is_available() or not is_supported(
            torch.empty(0, device=cls.device)
        ):
            raise RuntimeError("V4.1 draft component checks require CUDA13/SM100")
        torch.backends.cuda.matmul.allow_tf32 = False
        cls.config = V41Config.from_dict(flash_config())
        cls.official = load_official_compressor()
        cls.kernel = _load_official_kernel()
        cls.official.__dict__.update(
            fp8_block_size=32,
            scale_fmt="ue8m0",
            scale_dtype=torch.float8_e8m0fnu,
            act_quant=cls.kernel.act_quant,
            fp8_gemm=cls.kernel.fp8_gemm,
        )
        from rtp_llm.model_loader.model_weight_info import ModelWeights

        weights = ModelWeights(3, str(cls.device), torch.bfloat16)
        cls.raw_projections = []
        for stage, (outputs, inputs) in enumerate(
            ((5120, 15360), (512, 5120), (512, 5120), (512, 5120))
        ):
            row_ids = torch.arange(outputs, device=cls.device)
            columns = (row_ids * 7 + stage * 31) % inputs
            weight = torch.zeros((outputs, inputs), device=cls.device)
            weight[row_ids, columns] = (1 - 2 * (row_ids % 2)).float()
            weight = weight.to(torch.float8_e4m3fn)
            scale = (
                2.0
                ** (
                    torch.arange(
                        outputs // 32 * (inputs // 32), device=cls.device
                    ).reshape(outputs // 32, inputs // 32)
                    % 5
                    - 2
                )
            ).to(torch.float8_e8m0fnu)
            cls.raw_projections.append((weight, scale))
            if stage == 0:
                for suffix, value in (("weight", weight), ("scale", scale)):
                    weights.set_global_weight("v41.mtp.0.main_proj." + suffix, value)
                weights.set_global_weight(
                    "v41.mtp.0.main_norm.weight",
                    torch.ones(5120, dtype=torch.bfloat16, device=cls.device),
                )
            else:
                for suffix, value in (("weight", weight), ("scale", scale)):
                    weights.set_layer_weight(stage - 1, "v41.attn.wkv." + suffix, value)
                weights.set_layer_weight(
                    stage - 1,
                    "v41.attn.kv_norm.weight",
                    torch.linspace(0.5, 1.5, 512, device=cls.device).bfloat16(),
                )
        cls.weights = weights
        cls.module = V41PrefillDraftCommit.from_model_weights(cls.config, weights)
        with cls.official.set_dtype(torch.bfloat16), torch.device(cls.device):
            cls.frequencies = cls.official.precompute_freqs_cis(
                64, 1048576, 0, 10000, 16, 32, 1
            )

    def aux(self, positions, mode=ReplayMode.BOUNDED):
        count = len(positions)
        values = torch.tensor(
            [448, -448, 1.0625, 1.1875, 0, 32, -32, 0.5] * 4,
            dtype=torch.bfloat16,
            device=self.device,
        )
        scales = 2.0 ** (
            torch.arange(count * 480, device=self.device).reshape(count, 480) % 6 - 3
        )
        hidden = (values[None, None, :].float() * scales[:, :, None]).flatten(1)
        row_map = AuxRowMap(
            "request-a",
            7,
            ReplayConfig(mode).fingerprint,
            tuple(positions),
            tuple(index * 3 + 11 for index in range(count)),
            tuple(index % 3 == 0 for index in range(count)),
        )
        return hidden.bfloat16(), row_map

    def bindings(self):
        bindings, backing = {}, {}
        for stage in range(40, 43):
            pages, storage = _pages(CacheRegion.SWA, 136, entries=136)
            bindings[stage] = SwaBinding(
                pages,
                torch.tensor([2], device=self.device, dtype=torch.int32),
                torch.tensor([61], device=self.device, dtype=torch.int32),
                torch.tensor([134], device=self.device, dtype=torch.int32),
            )
            backing[stage] = storage
        return bindings, backing

    def run_commit(self, hidden, row_map, positions, bindings, **overrides):
        arguments = dict(
            required_positions=positions,
            swa_bindings=bindings,
            request_id=row_map.request_id,
            forward_epoch=row_map.forward_epoch,
            replay_fingerprint=row_map.replay_fingerprint,
        )
        arguments.update(overrides)
        return self.module.commit(hidden, row_map, **arguments)

    def reference_linear(self, values, index):
        weight, scale = self.raw_projections[index]
        encoded, scales = self.kernel.act_quant(
            values, 32, "ue8m0", torch.float8_e8m0fnu
        )
        with self.official.set_dtype(torch.bfloat16):
            return self.kernel.fp8_gemm(
                encoded,
                scales,
                weight,
                scale,
                torch.float8_e8m0fnu,
                block_size=32,
            )

    def reference_rows(self, selected, positions):
        norm = self.official.RMSNorm(5120, 1e-20).to(self.device).bfloat16()
        norm.weight = nn.Parameter(self.module.main_norm, requires_grad=False)
        main = norm(self.reference_linear(selected, 0))
        result = []
        for stage in range(3):
            stage_norm = self.official.RMSNorm(512, 1e-20).to(self.device).bfloat16()
            stage_norm.weight = nn.Parameter(
                getattr(self.module, f"stage_norm_{stage}"), requires_grad=False
            )
            values = stage_norm(self.reference_linear(main, stage + 1))
            self.official.apply_rotary_emb(
                values[None, :, -64:], self.frequencies[positions]
            )
            result.append(_official_gpu(self.kernel, values, CacheRegion.SWA))
        return main, result

    def assert_destination(self, bindings, backing, positions, encoded):
        for index, layer in enumerate(range(40, 43)):
            expected = torch.full_like(backing[layer], 0x5A)
            rows = expected[2, : 136 * 528].view(136, 528)
            rows[positions % 136] = encoded[index]
            torch.testing.assert_close(backing[layer], expected, rtol=0, atol=0)
            self.assertEqual(bindings[layer].valid_starts.item(), 61)
            self.assertEqual(bindings[layer].valid_ends.item(), 134)

    def test_valid_aux_is_selected_before_every_projection_and_absolute_rope(self):
        for mode in (ReplayMode.FULL, ReplayMode.BOUNDED):
            with self.subTest(mode=mode):
                hidden, rows = self.aux(range(120, 260), mode)
                required = tuple(range(132, 260))
                hidden[:12] = float("nan")
                original = hidden.view(torch.int16).clone()
                bindings, backing = self.bindings()
                seen = []
                handles = []
                for index, projection in enumerate(
                    (self.module.main_projection, *self.module.stage_projections)
                ):
                    handles.append(
                        projection.register_forward_pre_hook(
                            lambda _, args, index=index: seen.append(
                                (index, args[0].clone())
                            )
                        )
                    )
                try:
                    result = self.run_commit(hidden, rows, required, bindings)
                finally:
                    for handle in handles:
                        handle.remove()
                positions = torch.tensor(required, device=self.device)
                main, encoded = self.reference_rows(hidden[12:].contiguous(), positions)
                self.assertEqual([index for index, _ in seen], [0, 1, 2, 3])
                torch.testing.assert_close(seen[0][1], hidden[12:], rtol=0, atol=0)
                for _, stage_input in seen[1:]:
                    torch.testing.assert_close(stage_input, main, rtol=0, atol=0)
                self.assertEqual(result.main_projection_rows, 128)
                self.assertEqual(result.stage_projection_rows, (128, 128, 128))
                self.assertEqual(result.positions, required)
                self.assertEqual(result.logical_rows, rows.logical_rows[12:])
                self.assertEqual(result.image_mask, rows.image_mask[12:])
                self.assertTrue(result.write_completed)
                self.assertEqual(len(result.writes), 3)
                self.assert_destination(bindings, backing, positions, encoded)
                torch.testing.assert_close(
                    hidden.view(torch.int16), original, rtol=0, atol=0
                )

    def test_tail_127_128_129_and_model_context_boundary(self):
        for count in (1, 3, 127, 128, 129):
            with self.subTest(count=count):
                start = 1048576 - count
                hidden, rows = self.aux(range(start, 1048576))
                required = rows.positions[-128:]
                selected = hidden[-128:].contiguous()
                bindings, backing = self.bindings()
                result = self.run_commit(
                    hidden,
                    rows,
                    required,
                    bindings,
                    replay_floor=max(0, 1048576 - 128),
                )
                positions = torch.tensor(required, device=self.device)
                _, encoded = self.reference_rows(selected, positions)
                self.assertEqual(result.main_projection_rows, min(128, count))
                self.assert_destination(bindings, backing, positions, encoded)

    def test_empty_local_aux_executes_no_projection_or_writer(self):
        for valid_count in (0, 7):
            with self.subTest(valid_count=valid_count):
                hidden, rows = self.aux(range(136, 136 + valid_count))
                bindings, backing = self.bindings()
                with patch.object(
                    self.module.main_projection,
                    "forward",
                    side_effect=AssertionError("empty main projection"),
                ), patch(
                    "rtp_llm.models_py.modules.dsv41.draft.write_compact",
                    side_effect=AssertionError("empty compact write"),
                ):
                    result = self.run_commit(hidden, rows, (), bindings)
                self.assertEqual(result.main_projection_rows, 0)
                self.assertEqual(result.stage_projection_rows, (0, 0, 0))
                self.assertEqual(result.positions, ())
                for storage in backing.values():
                    self.assertTrue((storage == 0x5A).all().item())

    def test_rejects_stale_missing_aliased_or_colliding_rows_before_projection(self):
        hidden, rows = self.aux((0, 3, 136))
        cases = (
            ({"request_id": "another"}, (0,), "stale"),
            ({"forward_epoch": 8}, (0,), "stale"),
            ({"replay_fingerprint": ReplayConfig().fingerprint}, (0,), "stale"),
            ({}, (2,), "has not been computed"),
            ({}, (3, 0), "unique and ordered"),
            ({}, (0, 136), "collide"),
            ({"replay_floor": 4}, (3,), "replay floor"),
            ({"request_index": 1}, (0,), "outside"),
        )
        for overrides, required, message in cases:
            with self.subTest(message=message, overrides=overrides):
                bindings, backing = self.bindings()
                with patch.object(
                    self.module.main_projection,
                    "forward",
                    side_effect=AssertionError("invalid projection"),
                ), self.assertRaisesRegex(ValueError, message):
                    self.run_commit(hidden, rows, required, bindings, **overrides)
                for storage in backing.values():
                    self.assertTrue((storage == 0x5A).all().item())
        bindings, _ = self.bindings()
        bindings[41] = bindings[40]
        with self.assertRaisesRegex(ValueError, "must not alias"):
            self.run_commit(hidden, rows, (0,), bindings)
        bindings, _ = self.bindings()
        bindings[42].page_ids.zero_()
        with self.assertRaisesRegex(ValueError, "unmapped"):
            self.run_commit(hidden, rows, (0,), bindings)

    def test_component_binds_only_commit_weights_without_copy(self):
        self.assertIs(
            self.module.main_projection.weight,
            self.weights.global_weights["v41.mtp.0.main_proj.weight"],
        )
        self.assertIs(
            self.module.main_norm,
            self.weights.global_weights["v41.mtp.0.main_norm.weight"],
        )
        for stage in range(3):
            self.assertIs(
                self.module.stage_projections[stage].weight,
                self.weights.weights[stage]["v41.attn.wkv.weight"],
            )
        self.assertFalse(
            any(
                token in name
                for name in self.module.state_dict()
                for token in (
                    "wq",
                    "moe",
                    "ffn",
                    "markov",
                    "confidence",
                    "embed",
                    "head",
                )
            )
        )


if __name__ == "__main__":
    unittest.main()

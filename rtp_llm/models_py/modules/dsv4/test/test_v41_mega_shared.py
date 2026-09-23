"""Host checks for native MXFP8 shared-expert layout and launch contracts."""

import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from rtp_llm.models_py.modules.dsv4.moe.strategies.mega_se import MegaMoEStrategySE
from rtp_llm.utils.model_weight import W


class V41MegaSharedTest(unittest.TestCase):
    @staticmethod
    def _fake_deep_gemm():
        def pack(scale, mn, k, recipe, num_groups):
            assert recipe == (1, 32) and num_groups == 1
            assert tuple(scale.shape) == (1, mn, k // 32)
            raw = (scale.view(torch.int32) >> 23).to(torch.uint8)
            packed = raw.contiguous().view(torch.int32)
            out = torch.empty_strided(
                packed.shape, (mn * (k // 128), 1, mn), dtype=torch.int32
            )
            return out.copy_(packed)

        return SimpleNamespace(
            transform_sf_into_required_layout=mock.Mock(side_effect=pack),
            transform_weights_for_mega_moe=lambda l1, l2: (
                (l1[0].clone(), l1[1].clone()),
                (l2[0], l2[1]),
            ),
        )

    def test_group32_scale_values_and_mn_major_stride(self):
        dg = self._fake_deep_gemm()
        raw = torch.tensor(
            [[124, 125, 126, 127], [128, 129, 130, 131]], dtype=torch.uint8
        )
        result = MegaMoEStrategySE._shared_expert_sf_to_int(
            dg, raw.view(torch.float8_e8m0fnu), 37, 128, 32
        )
        expected = raw.repeat_interleave(32, dim=0)[:37]
        self.assertEqual(result.stride(0), 1)
        self.assertTrue(
            torch.equal(result.reshape(-1).view(torch.uint8), expected.reshape(-1))
        )
        self.assertIs(
            MegaMoEStrategySE._shared_expert_sf_to_int(dg, result, 37, 128, 32),
            result,
        )

    def test_native_scale_rejects_wrong_shape_dtype_and_packed_stride(self):
        dg = self._fake_deep_gemm()
        with self.assertRaisesRegex(ValueError, "scale shape"):
            MegaMoEStrategySE._shared_expert_sf_to_int(
                dg, torch.zeros((1, 1)).to(torch.float8_e8m0fnu), 128, 128, 32
            )
        with self.assertRaises(TypeError):
            MegaMoEStrategySE._shared_expert_sf_to_int(
                dg, torch.ones(4, 4), 128, 128, 32
            )
        with self.assertRaisesRegex(ValueError, "MN-major"):
            MegaMoEStrategySE._shared_expert_sf_to_int(
                dg, torch.zeros((128, 2), dtype=torch.int32), 128, 256, 32
            )

    def test_native_reload_preserves_addresses_without_aliasing_checkpoint(self):
        dg = self._fake_deep_gemm()
        obj = MegaMoEStrategySE.__new__(MegaMoEStrategySE)
        torch.nn.Module.__init__(obj)
        obj.cfg = SimpleNamespace(shared_fp8_block_size=32)
        weights = {
            W.v4_shared_w13_w: torch.randn(256, 128).to(torch.float8_e4m3fn),
            W.v4_shared_w13_s: torch.full((8, 4), 125, dtype=torch.uint8).view(
                torch.float8_e8m0fnu
            ),
            W.v4_shared_w2_w: torch.randn(128, 128).to(torch.float8_e4m3fn),
            W.v4_shared_w2_s: torch.full((4, 4), 126, dtype=torch.uint8).view(
                torch.float8_e8m0fnu
            ),
        }
        with mock.patch.object(torch.cuda, "empty_cache"):
            obj._setup_shared_expert_weights(dict(weights), dg, W, 128, 128)
            self.assertEqual(obj._shared_recipe, (1, 1, 32))
            self.assertNotEqual(
                obj._se_l2_w.data_ptr(), weights[W.v4_shared_w2_w].data_ptr()
            )
            names = ("_se_l1_w", "_se_l1_sf", "_se_l2_w", "_se_l2_sf")
            snapshots = {name: getattr(obj, name).clone() for name in names}
            addresses = {name: getattr(obj, name).data_ptr() for name in names}
            for _ in range(2):
                for name in names:
                    getattr(obj, name).zero_()
                obj._setup_shared_expert_weights(
                    dict(weights), dg, W, 128, 128, isolate_scratch=True
                )
                for name in names:
                    value = getattr(obj, name)
                    self.assertEqual(addresses[name], value.data_ptr())
                    self.assertTrue(
                        torch.equal(
                            value.reshape(-1).view(torch.uint8),
                            snapshots[name].reshape(-1).view(torch.uint8),
                        )
                    )

    def test_empty_rank_still_packs_and_launches_collective(self):
        obj = MegaMoEStrategySE.__new__(MegaMoEStrategySE)
        torch.nn.Module.__init__(obj)
        obj._validate_capacity = mock.Mock()
        obj._block_m = mock.Mock(return_value=64)
        obj._input_packer = mock.Mock()
        obj._mega_buf = object()
        obj._mega_y = torch.empty((8, 128), dtype=torch.bfloat16)
        obj._launch = mock.Mock()
        x = torch.empty((0, 128), dtype=torch.bfloat16)
        weights, indices = torch.empty((0, 6)), torch.empty((0, 6), dtype=torch.int64)
        result = obj(x, weights, indices)
        self.assertEqual(tuple(result.shape), (0, 128))
        obj._input_packer.pack.assert_called_once_with(
            x, weights, indices, obj._mega_buf, 0, 64
        )
        obj._launch.assert_called_once()
        self.assertEqual(obj._launch.call_args.args[1:], (0, x.device))

    def test_native_shared_rejects_routed_padding_fallback(self):
        obj = MegaMoEStrategySE.__new__(MegaMoEStrategySE)
        torch.nn.Module.__init__(obj)
        obj.cfg = SimpleNamespace(
            shared_fp8_block_size=32, moe_inter_dim=2304, dim=5120
        )
        with mock.patch(
            "rtp_llm.models_py.modules.dsv4.moe.strategies.mega_se."
            "_mega_intermediate_size",
            return_value=2560,
        ):
            with self.assertRaisesRegex(RuntimeError, "intermediate size"):
                obj.setup_weights({})


if __name__ == "__main__":
    unittest.main()

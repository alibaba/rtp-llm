"""Exercise the parameter owner with on-disk fused and split HF checkpoints."""

import os
import tempfile
import unittest
from types import SimpleNamespace

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from rtp_llm.models_py.module_base import RtpModule
from rtp_llm.models_py.new_models.qwen3_next.gdn_parameters import (
    Qwen3NextGdnParameters,
)
from rtp_llm.models_py.new_weight_manager import NewLoaderWeightManager
from rtp_llm.models_py.quant_methods.base import QuantizationConfig


class GdnOwnershipTest(unittest.TestCase):
    def test_rank_local_input_projection_matches_one_gemm_after_reload(self):
        # Same rank-local K,N layout as the CUDA mixed-attention reference.
        # Exercise checkpoint dispatch first, rather than constructing packed views
        # by hand. Include both replicated (DP) and sharded (TP) attention.
        devices = ["cpu"]
        if torch.cuda.is_available() and torch.version.hip is None:
            devices.append("cuda")
        config = SimpleNamespace(
            linear_key_head_dim=8,
            linear_value_head_dim=8,
            linear_num_key_heads=2,
            linear_num_value_heads=4,
            linear_conv_kernel_dim=3,
        )
        quant = QuantizationConfig(
            "modelopt_fp4", source_config=SimpleNamespace(mixed_attention=True)
        )
        gen = torch.Generator().manual_seed(43)

        def rand(n):
            return torch.randn(n, 256, generator=gen).bfloat16()

        parts = [rand(n) for n in (16, 16, 32, 32, 4, 4)]
        q, k, v, z, b, a = parts
        state = {
            "in_proj_qkv.weight": torch.cat((q, k, v)),
            "in_proj_z.weight": z,
            "in_proj_b.weight": b,
            "in_proj_a.weight": a,
        }
        for tp, rank in ((1, 0), (2, 0), (2, 1)):
            for device in devices:
                with self.subTest(tp=tp, rank=rank, device=device):
                    m = Qwen3NextGdnParameters(
                        config,
                        256,
                        1e-6,
                        tp,
                        rank,
                        torch.bfloat16,
                        quant,
                        "linear_attn",
                    ).to(device)
                    m.load_weights(state)
                    q_param, ba_param = m.in_proj_qkvz.weight, m.in_proj_ba_w
                    m._prepare_input_projection()
                    if device == "cpu":
                        self.assertIsNone(m._packed_input_weight())
                        m._pack_dense_input_projection()  # storage protocol on CPU
                    expected = (
                        torch.cat([t.chunk(tp)[rank] for t in parts])
                        .T.contiguous()
                        .T.to(device)
                    )
                    self.assertEqual(
                        expected.stride(), m._packed_input_weight().stride()
                    )
                    self.assertTrue(torch.equal(expected, m._packed_input_weight()))
                    self.assertIs(m.in_proj_qkvz.weight, q_param)
                    self.assertIs(m.in_proj_ba_w, ba_param)
                    ptr = ba_param.untyped_storage().data_ptr()
                    self.assertEqual(ptr, q_param.untyped_storage().data_ptr())
                    m._pack_dense_input_projection()
                    self.assertEqual(
                        ptr, m._packed_input_weight().untyped_storage().data_ptr()
                    )
                    for rows in (1, 3, 31):
                        x = rand(rows).to(device)
                        actual = torch.cat(m._input_project(x), dim=-1)
                        self.assertTrue(
                            torch.equal(actual, torch.nn.functional.linear(x, expected))
                        )
                    if device == "cuda":
                        graph = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(graph):
                            captured = torch.cat(m._input_project(x), dim=-1)
                        x.copy_(rand(x.size(0)).to(device))
                        graph.replay()
                        self.assertTrue(
                            torch.equal(
                                captured, torch.nn.functional.linear(x, expected)
                            )
                        )
                    # Existing registered views remain update destinations.
                    with torch.no_grad():
                        ba_param.zero_()
                    self.assertEqual(m._input_project(x)[1].count_nonzero().item(), 0)
                    m.in_proj_ba_w = torch.nn.Parameter(
                        torch.ones_like(ba_param), requires_grad=False
                    )
                    self.assertIsNone(m._packed_input_weight())

    def model(self, rank, tp=2):
        config = SimpleNamespace(
            linear_key_head_dim=2,
            linear_value_head_dim=2,
            linear_num_key_heads=2,
            linear_num_value_heads=4,
            linear_conv_kernel_dim=3,
        )
        return Qwen3NextGdnParameters(
            config, 8, 1e-6, tp, rank, torch.float32, None, "linear_attn"
        )

    def test_online_projection_update_matches_cold_load_and_is_atomic(self):
        from rtp_llm.models_py.model_loader import NewModelLoader

        class Model(RtpModule):
            def runtime_weight_view(self):
                return {}

        loader = SimpleNamespace(
            _mapped_checkpoint_name=NewModelLoader._mapped_checkpoint_name
        )
        inputs = {
            "in_proj_qkvz.weight": torch.arange(24 * 8).float().reshape(24, 8),
            "out_proj.weight": torch.arange(8 * 8).float().reshape(8, 8),
            "in_proj_ba.weight": torch.arange(64).float().reshape(8, 8),
            "conv1d.weight": torch.arange(48).float().reshape(16, 1, 3),
            "dt_bias": torch.arange(4).float() + 3,
            "A_log": torch.arange(4).float() + 5,
            "norm.weight": torch.arange(2).float() + 7,
        }
        request = {"linear_attn." + name: value for name, value in inputs.items()}
        for tp, rank in ((1, 0), (2, 0), (2, 1)):
            with self.subTest(tp=tp, rank=rank):
                model = Model()
                model.linear_attn = self.model(rank, tp)
                model.linear_attn.load_weights(
                    {k: torch.zeros_like(v) for k, v in inputs.items()}
                )
                expected = self.model(rank, tp)
                expected.load_weights(inputs)
                manager = NewLoaderWeightManager(model, loader)
                parameters = dict(model.linear_attn.named_parameters())
                names = [expected._state_parameters.get(k, k) for k in inputs]
                pointers = {k: parameters[k].data_ptr() for k in names}
                with self.assertRaises(ValueError):
                    manager.update_tensors(
                        {**request, "linear_attn.out_proj.weight": torch.ones(8, 1)}
                    )
                self.assertEqual(manager.version, 0)
                for name in names:
                    torch.testing.assert_close(
                        parameters[name], torch.zeros_like(parameters[name])
                    )
                manager.update_tensors(request)
                self.assertEqual(manager.version, 1)
                for name in names:
                    self.assertEqual(parameters[name].data_ptr(), pointers[name])
                    torch.testing.assert_close(
                        parameters[name], expected.get_parameter(name)
                    )

    def test_per_head_and_split_produce_same_projections(self):
        matrix = torch.arange(24 * 8, dtype=torch.float32).reshape(24, 8)
        q, k, v, z = matrix.split([4, 4, 8, 8])
        ba = torch.arange(64, dtype=torch.float32).reshape(8, 8)
        b, a = ba.chunk(2)
        fused = {
            "in_proj_qkvz.weight": torch.cat(
                [t.reshape(2, -1, 8) for t in (q, k, v, z)], dim=1
            ).reshape(-1, 8),
            "in_proj_ba.weight": torch.cat(
                [t.reshape(2, -1, 8) for t in (b, a)], dim=1
            ).reshape(-1, 8),
        }
        split = {
            "in_proj_qkv.weight": torch.cat([q, k, v]),
            "in_proj_z.weight": z,
            "in_proj_b.weight": b,
            "in_proj_a.weight": a,
        }
        conv = torch.arange(16 * 3, dtype=torch.float32).reshape(16, 1, 3)
        out = torch.arange(64, dtype=torch.float32).reshape(8, 8)
        common = {
            "A_log": torch.arange(4, dtype=torch.float32),
            "dt_bias": torch.arange(4, dtype=torch.float32) + 10,
            "conv1d.weight": conv,
            "norm.weight": torch.ones(2),
            "out_proj.weight": out,
        }
        for layout in (fused, split):
            with tempfile.TemporaryDirectory() as folder:
                path = os.path.join(folder, "model.safetensors")
                save_file({**layout, **common}, path)
                for rank in range(2):
                    owner = self.model(rank)
                    with safe_open(path, framework="pt", device="cpu") as weights:
                        for name in reversed(weights.keys()):
                            owner.load_weights({name: weights.get_tensor(name)})
                    owner.validate_weights_loaded()
                    local = lambda values: torch.cat(
                        [part.chunk(2)[rank] for part in values]
                    )
                    torch.testing.assert_close(
                        owner.in_proj_qkvz.weight, local((q, k, v, z))
                    )
                    torch.testing.assert_close(owner.in_proj_ba_w, local((b, a)).T)
                    torch.testing.assert_close(
                        owner.conv1d_w, local(conv.squeeze(1).split((4, 4, 8)))
                    )
                    torch.testing.assert_close(
                        owner.dt_bias, common["dt_bias"].chunk(2)[rank]
                    )
                    torch.testing.assert_close(
                        owner.a_log, common["A_log"].chunk(2)[rank]
                    )
                    torch.testing.assert_close(
                        owner.out_proj.weight, out.chunk(2, dim=1)[rank]
                    )

    def test_mixed_layout_fails(self):
        m = self.model(0)
        m.load_weights({"in_proj_b.weight": torch.ones(4, 8)})
        with self.assertRaisesRegex(RuntimeError, "mix"):
            m.load_weights({"in_proj_ba.weight": torch.ones(8, 8)})

    def test_fp8_block_scale_grid_uses_the_same_rank_as_its_weights(self):
        config = SimpleNamespace(
            linear_key_head_dim=128,
            linear_value_head_dim=128,
            linear_num_key_heads=2,
            linear_num_value_heads=2,
            linear_conv_kernel_dim=3,
        )
        quant = QuantizationConfig(
            "fp8_block", source_config=SimpleNamespace(weight_block_size=[128, 128])
        )
        q, k, v, z = torch.arange(16, dtype=torch.float32).reshape(8, 2).add(1).chunk(4)
        out = torch.arange(4, dtype=torch.float32).reshape(2, 2).add(1)
        for rank in range(2):
            for layout in (
                {
                    "in_proj_qkvz.weight_scale_inv": torch.stack(
                        (q, k, v, z), dim=1
                    ).reshape(8, 2)
                },
                {
                    "in_proj_qkv.weight_scale_inv": torch.cat((q, k, v)),
                    "in_proj_z.weight_scale_inv": z,
                },
            ):
                m = Qwen3NextGdnParameters(
                    config, 256, 1e-6, 2, rank, torch.float32, quant, "linear_attn"
                )
                m.load_weights({**layout, "out_proj.weight_scale_inv": out})
                torch.testing.assert_close(
                    m.in_proj_qkvz.weight_scale_inv,
                    torch.stack([t[rank] for t in (q, k, v, z)]),
                )
                torch.testing.assert_close(
                    m.out_proj.weight_scale_inv, out[:, rank : rank + 1]
                )

    @unittest.skipIf(
        os.environ.get("NEWLOADER_TEST_NATIVE_FST") == "0",
        "this platform lock does not ship the native FST reader",
    )
    def test_cpu_fst_stream_projects_ba_into_the_rank_local_owner(self):
        from rtp_llm.models_py.model_loader import NewLoaderConfig, NewModelLoader

        values = torch.arange(64, dtype=torch.float32).reshape(8, 8)
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "model.safetensors")
            save_file({"in_proj_ba.weight": values}, path)
            loader = NewModelLoader(
                {"model_type": "qwen3_next"},
                NewLoaderConfig(device="cpu", load_method="fastsafetensors"),
                model_path=folder,
            )
            for rank in range(2):
                owner = self.model(rank)
                # Explicitly call FST, not auto selection: no scratch fallback.
                for name, data in loader._fastsafetensors_weights_iterator(
                    [path], "cpu", None, None
                ):
                    owner.load_weights({name: data})
                torch.testing.assert_close(
                    owner.in_proj_ba_w, values[4 * rank : 4 * rank + 4].T
                )


if __name__ == "__main__":
    unittest.main()

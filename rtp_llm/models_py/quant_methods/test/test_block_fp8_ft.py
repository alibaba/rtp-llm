import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from safetensors.torch import save_file

from rtp_llm.models_py.checkpoint_reader_test_support import checkpoint_readers
from rtp_llm.models_py.ft_checkpoint import CheckpointTopology, FTCheckpoint
from rtp_llm.models_py.quant_methods.block_fp8_ft import (
    block_ft_bindings,
    block_ft_ffn_width,
    block_ft_scale,
    block_ft_weight,
    block_moe_ft_bindings,
)
from rtp_llm.models_py.quant_methods.fp8 import Fp8BlockLinearMethod


def write_ft(directory, data, *, producer=None):
    keys = {"rank_0_0_0." + k: v.contiguous() for k, v in data.items()}
    save_file(keys, str(Path(directory) / "ft.safetensors"))
    (Path(directory) / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "is_ft_style_weight": True,
                **({"ft_export_backend": producer} if producer is not None else {}),
                "weight_map": {k: "ft.safetensors" for k in keys},
            }
        )
    )


def linear(n, k):
    model = torch.nn.Module()
    model.proj = torch.nn.Module()
    model.proj.quant_config = SimpleNamespace(weight_block_size=[128, 128])
    Fp8BlockLinearMethod(model.proj.quant_config).create_weights(
        model.proj, input_size=k, output_size=n, params_dtype=torch.bfloat16
    )
    return model


def source_snapshot(w, s, packed):
    n, k = w.shape
    if not packed:
        # main PerBlockFp8Weight._postprocess: flat reshape, not transpose.
        return w.reshape(k, n), s.reshape(s.shape[1], -1)
    # Same persisted int32 little-endian representation as DeepGEMM's packer;
    # this CPU fixture only exercises decoding, not the SM100 CUDA pack kernel.
    e = torch.log2(s).int() + 127
    e = e.repeat_interleave(128, dim=0)[:n]
    e = torch.nn.functional.pad(e, (0, (-e.shape[1]) % 4))
    byte = e.view(n, -1, 4)
    raw = byte[..., 0] | byte[..., 1] << 8 | byte[..., 2] << 16 | byte[..., 3] << 24
    return w.clone(), raw


class BlockFTTest(unittest.TestCase):
    def test_lora_updates_only_affected_calibration_tiles_including_partial_tail(self):
        from rtp_llm.models_py.runtime_weight_update import LoRAUpdate

        n, k = 129, 130
        weight = torch.full((n, k), 3.0).to(torch.float8_e4m3fn)
        scale = torch.tensor([[1.0, 2.0], [4.0, 0.25]])
        a = torch.zeros(1, k)
        b = torch.zeros(n, 1)
        a[0, -2:] = torch.tensor([1.0, -0.5])
        b[-1] = 600
        for packed in (False, True):
            with self.subTest(packed=packed), tempfile.TemporaryDirectory() as root:
                kernel, steps = source_snapshot(weight, scale, packed)
                write_ft(root, {"layers.0.kernel": kernel, "layers.0.scale": steps}, producer="cuda")
                cp = FTCheckpoint.open(root, CheckpointTopology())
                m = linear(n, k)
                cp.load_parameters(m, block_ft_bindings(m.proj, "layers.0.kernel", "layers.0.scale", "proj", cp))
                Fp8BlockLinearMethod(m.proj.quant_config).merge_ft_lora(
                    m.proj, LoRAUpdate("proj.weight", a, b)
                )
                expected_step = scale.clone()
                expected_step[-1, -1] = 2.0 if packed else 600.75 / 448
                expected = weight.view(torch.uint8).clone()
                expected[-1, -2:] = (torch.tensor([600.75, -299.25]) / expected_step[-1, -1]).to(
                    torch.float8_e4m3fn
                ).view(torch.uint8)
                torch.testing.assert_close(m.proj.weight_scale_inv, expected_step, rtol=0, atol=0)
                torch.testing.assert_close(m.proj.weight.view(torch.uint8), expected)

    def test_padded_tp_ffn_keeps_producer_shards_and_zero_scale_blocks(self):
        # The producer pads 320 real channels to 512 BEFORE TP2 splitting.
        # Rank 0 therefore owns 256 real channels, not 160. A rank-local crop
        # loses channels 160..255 even if the target shapes look correct.
        hidden, logical, width = 128, 160, 256
        gate_key = "layers.0.ffn_weights.intermediate_weight13.kernel"
        down_key = "layers.0.ffn_weights.intermediate_weight2.kernel"

        def scale_key(key):
            return key.removesuffix(".kernel") + ".weight_only_quant_scale"

        for packed in (False, True):
            with self.subTest(packed=packed), tempfile.TemporaryDirectory() as root:
                data = {}
                ref = 0.0
                for rank in range(2):
                    live = min(width, 2 * logical - rank * width)
                    gate_up = torch.zeros(2 * width, hidden)
                    down = torch.zeros(hidden, width)
                    # Distinct checkpoint scales across projections and TP ranks.
                    gate_up[:live, 0] = 1
                    gate_up[width : width + live, 0] = 2
                    down[0, :live] = 3
                    s13 = torch.ones(4, 1) * (2**rank)
                    s2 = torch.ones(1, 2)
                    ref += (
                        live
                        * torch.nn.functional.silu(torch.tensor(2.0**rank)).item()
                        * (2.0 ** (rank + 1))
                        * 3
                    )
                    if not packed and rank == 1:
                        s13[[1, 3]] = 0
                        s2[:, 1] = 0
                    for key, w, s in ((gate_key, gate_up, s13), (down_key, down, s2)):
                        kernel, scale = source_snapshot(
                            w.to(torch.float8_e4m3fn), s, packed
                        )
                        prefix = f"rank_{rank}_0_0."
                        data[prefix + key] = kernel.contiguous()
                        data[prefix + scale_key(key)] = scale.contiguous()
                save_file(data, str(Path(root) / "weights.safetensors"))
                (Path(root) / "model.safetensors.index.json").write_text(
                    json.dumps(
                        {
                            "is_ft_style_weight": True,
                            "ft_export_backend": "cuda",
                            "weight_map": {key: "weights.safetensors" for key in data},
                        }
                    )
                )
                actual = 0.0
                for rank in range(2):
                    cp = FTCheckpoint.open(
                        root, CheckpointTopology(tp_size=2, tp_rank=rank)
                    )
                    m = torch.nn.Module()
                    m.gate = linear(2 * logical, hidden).proj
                    m.down = linear(hidden, logical).proj
                    found = block_ft_ffn_width(cp, down_key, logical, hidden)
                    self.assertEqual(found, width)
                    bindings = []
                    for name, key, projection in (
                        ("gate", gate_key, "gate_up"),
                        ("down", down_key, "down"),
                    ):
                        bindings.extend(
                            block_ft_bindings(
                                getattr(m, name),
                                key,
                                scale_key(key),
                                name,
                                cp,
                                projection=projection,
                                ffn_width=found,
                            )
                        )
                    if rank == 1:
                        invalid = torch.ones(2 * width, hidden).to(torch.float8_e4m3fn)
                        with self.assertRaisesRegex(ValueError, "nonzero weights"):
                            bindings[0].convert(
                                (invalid if packed else invalid.reshape(hidden, -1),)
                            )
                    cp.load_parameters(m, tuple(bindings))
                    self.assertEqual(
                        (m.gate.output_size, m.down.input_size), (2 * width, width)
                    )

                    def decoded(proj):
                        self.assertTrue(bool((proj.weight_scale_inv > 0).all()))
                        return (
                            proj.weight.float()
                            * proj.weight_scale_inv.repeat_interleave(
                                128, 0
                            ).repeat_interleave(128, 1)
                        )

                    gate, up = decoded(m.gate)[:, 0].chunk(2)
                    actual += (decoded(m.down) @ (torch.nn.functional.silu(gate) * up))[
                        0
                    ].item()
                self.assertAlmostEqual(actual, ref, delta=0.01)

    def test_real_file_into_block_linear_owner_and_output(self):
        n, k = 257, 521  # partial N, K blocks and packed int32 padding
        g = torch.Generator().manual_seed(57)
        w = torch.randint(-11, 12, (n, k), generator=g).to(torch.float8_e4m3fn)
        s = (2.0 ** torch.randint(-3, 4, (3, 5), generator=g)).float()
        input = torch.randn(3, k, generator=g)

        def forward(weight, scale):
            dequant = (
                weight.float()
                * scale.repeat_interleave(128, 0).repeat_interleave(128, 1)[:n, :k]
            )
            return input @ dequant.T

        ref = forward(w, s)
        for packed in (False, True):
            kernel, scale = source_snapshot(w, s, packed)
            with tempfile.TemporaryDirectory() as root:
                write_ft(
                    root, {"layers.0.kernel": kernel, "layers.0.scale": scale},
                    producer="cuda",
                )
                for backend in checkpoint_readers("scratch", "fastsafetensors"):
                    with self.subTest(packed=packed, backend=backend):
                        cp = FTCheckpoint.open(root, CheckpointTopology())
                        cp.read_backend = backend
                        m = linear(n, k)
                        ptr = m.proj.weight.data_ptr()
                        bs = block_ft_bindings(
                            m.proj, "layers.0.kernel", "layers.0.scale", "proj", cp
                        )
                        cp.load_parameters(m, bs)
                        self.assertEqual(ptr, m.proj.weight.data_ptr())
                        self.assertTrue(
                            torch.equal(
                                w.view(torch.uint8), m.proj.weight.view(torch.uint8)
                            )
                        )
                        torch.testing.assert_close(
                            m.proj.weight_scale_inv, s, rtol=0, atol=0
                        )
                        torch.testing.assert_close(
                            forward(m.proj.weight, m.proj.weight_scale_inv),
                            ref,
                            rtol=0,
                            atol=0,
                        )
                        self.assertEqual(bool(m.proj._packed_block_fp8_ft), packed)

    def test_reject_malformed_packed_and_inconsistent_shape(self):
        t = torch.full((259, 2), 127, dtype=torch.int32)
        for row in (2, 258):
            v = t.clone()
            v[row, 0] = 123
            with self.assertRaisesRegex(ValueError, "inconsistent"):
                block_ft_scale(v, (259, 600), source_dtype=torch.float8_e4m3fn)
        with self.assertRaisesRegex(ValueError, "NaN exponent"):
            block_ft_scale(t.fill_(-1), (259, 600), source_dtype=torch.float8_e4m3fn)
        with self.assertRaisesRegex(ValueError, "weight shape"):
            block_ft_weight(torch.ones(7, 2).to(torch.float8_e4m3fn), (7, 2), False)

    def test_rocm_fnuz_canonicalization(self):
        w = torch.tensor([[1.0, 2.0, 4.0], [-1.0, -2.0, -4.0]]).to(torch.float8_e4m3fn)
        raw = w.view(torch.float8_e4m3fnuz).reshape(3, 2)
        decoded = block_ft_weight(raw, (2, 3), False)
        s = block_ft_scale(torch.tensor([[4.0]]), (2, 3), source_dtype=raw.dtype)
        torch.testing.assert_close(
            decoded.float() * s, raw.reshape(2, 3).float() * 4, rtol=0, atol=0
        )

    def test_rocm_dense_separate_up_and_gate_undo_only_producer_transforms(self):
        from rtp_llm.utils.swizzle_utils import swizzle_tensor

        n, k = 128, 256
        g = torch.Generator().manual_seed(34)
        ws = [
            torch.randint(-12, 13, (n, k), generator=g).to(torch.float8_e4m3fn)
            for _ in range(2)
        ]
        names = tuple(
            "layers.0.ffn_weights." + x + ".kernel"
            for x in ("intermediate_weight", "intermediate_weight3")
        )
        keys = tuple(
            x.removesuffix(".kernel") + ".weight_only_quant_scale" for x in names
        )
        for swizzled in (False, True):
            # ROCm DeviceImpl does NOT rewrite separate w1; separate w3
            # uses swizzle_tensor or its shuffle_gemm_weight tile permutation.
            up = ws[1]
            if swizzled:
                up = swizzle_tensor(up, False)
            else:
                up = (
                    up.view(n // 16, 16, k // 32, 2, 16)
                    .permute(0, 2, 3, 1, 4)
                    .contiguous()
                    .view(n, k)
                )
            ft = {key: w.reshape(k, n) for key, w in zip(names, (ws[0], up))}
            scales = torch.tensor([[1.0, 2.0], [0.25, 4.0]])
            ft.update(
                {key: row.reshape(2, 1).clone() for key, row in zip(keys, scales)}
            )
            with tempfile.TemporaryDirectory() as root:
                write_ft(root, ft, producer="rocm")
                cp = FTCheckpoint.open(root, CheckpointTopology())
                # Explicit cross-runtime source ABI; same-device untagged
                # legacy ROCm follows torch.version.hip in production.
                m = linear(2 * n, k)
                cp.load_parameters(
                    m,
                    block_ft_bindings(
                        m.proj, names, keys, "proj", cp, swizzled=swizzled
                    ),
                )
                expected = torch.cat([w.view(torch.uint8) for w in ws])
                torch.testing.assert_close(m.proj.weight.view(torch.uint8), expected)
                torch.testing.assert_close(
                    m.proj.weight_scale_inv, scales, rtol=0, atol=0
                )

    def test_expert_ft_vs_hf_routed_output(self):
        from rtp_llm.config.quant_config import Fp8BlockWiseQuantConfig
        from rtp_llm.models_py.layers.moe_experts import BaseMoEExperts
        from rtp_llm.models_py.quant_methods.base import QuantizationConfig

        cfg = Fp8BlockWiseQuantConfig(bits=8, group_size=128, is_quanted=True)
        h, m, e = 128, 256, 2

        def model():
            tree = torch.nn.Module()
            tree.experts = BaseMoEExperts(
                num_experts=e,
                top_k=1,
                hidden_size=h,
                moe_intermediate_size=m,
                tp_size=1,
                tp_rank=0,
                ep_size=1,
                ep_rank=0,
                params_dtype=torch.bfloat16,
                model_config=SimpleNamespace(
                    num_layers=1, activation_type="SiGLU", quant_config=cfg
                ),
                parallelism_config=SimpleNamespace(
                    dp_size=1, dp_rank=0, world_size=1, local_world_size=1
                ),
                moe_config=None,
                quant_config=QuantizationConfig("fp8_block", cfg),
                layer_idx=0,
            )
            return tree

        g = torch.Generator().manual_seed(121)
        hf = {}
        for idx in range(e):
            for proj in ("up", "gate", "down"):
                shape = (h, m) if proj == "down" else (m, h)
                key = f"{idx}.{proj}_proj."
                hf[key + "weight"] = torch.randint(-5, 6, shape, generator=g).to(
                    torch.float8_e4m3fn
                )
                hf[key + "weight_scale_inv"] = 2.0 ** torch.randint(
                    -4, 0, tuple(n // 128 for n in shape), generator=g
                )
        ref = model().experts
        for key, t in hf.items():
            ref.load_weights({key: t})
        ref.validate_weights_loaded()

        def routed(owner):
            def dq(w, s):
                return w.float() * s.repeat_interleave(128, 1).repeat_interleave(128, 2)

            x = torch.arange(h).float().reshape(1, -1) / h
            up, gate = (x @ dq(owner.w13, owner.w13_scale).transpose(1, 2)).chunk(
                2, dim=-1
            )
            return (up * torch.nn.functional.silu(gate)) @ dq(
                owner.w2, owner.w2_scale
            ).transpose(1, 2)

        reference = routed(ref)
        for packed in (False, True):
            ft = {}
            for name, kind, ps in (
                ("intermediate_weight", "w13", ("up", "gate")),
                ("intermediate_weight2", "w2", ("down",)),
            ):
                stem = "layers.0.partial_moe_weights." + name
                ws, ss = [], []
                for idx in range(e):
                    w = torch.cat(
                        [hf[f"{idx}.{p}_proj.weight"].view(torch.uint8) for p in ps]
                    ).view(torch.float8_e4m3fn)
                    scale = torch.cat(
                        [hf[f"{idx}.{p}_proj.weight_scale_inv"] for p in ps]
                    )
                    ws.append(w)
                    ss.append(source_snapshot(w, scale, True)[1] if packed else scale)
                ft[stem + ".kernel"] = torch.stack(
                    [w.view(torch.uint8) for w in ws]
                ).view(torch.float8_e4m3fn)
                ft[stem + ".weight_only_quant_scale"] = torch.stack(ss)
            with tempfile.TemporaryDirectory() as root:
                write_ft(root, ft, producer="cuda")
                for backend in checkpoint_readers("scratch", "fastsafetensors"):
                    with self.subTest(packed=packed, backend=backend):
                        cp = FTCheckpoint.open(root, CheckpointTopology())
                        cp.read_backend = backend
                        tree = model()
                        bindings = block_moe_ft_bindings(
                            tree.experts, "layers.0.", "experts", cp, 0
                        )
                        cp.load_parameters(tree, bindings)
                        tree.experts.validate_weights_loaded()
                        for key in ("w13", "w2", "w13_scale", "w2_scale"):
                            torch.testing.assert_close(
                                getattr(tree.experts, key).float(),
                                getattr(ref, key).float(),
                                rtol=0,
                                atol=0,
                            )
                        torch.testing.assert_close(
                            routed(tree.experts), reference, rtol=0, atol=0
                        )
                        self.assertEqual(
                            getattr(tree.experts, "_packed_block_fp8_ft", set()),
                            {"w13_scale", "w2_scale"} if packed else set(),
                        )


if __name__ == "__main__":
    unittest.main()

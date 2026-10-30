"""CPU contracts for Prefill opt-outs, shape fallback and stream dependencies."""

import ast
import logging
import unittest
from contextlib import nullcontext
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

ROOT = Path(__file__).resolve().parents[1]


def definition(path, name, **namespace):
    # Load the production definition without importing native PPU dependencies.
    tree = ast.parse((ROOT / path).read_text())
    node = next(n for n in tree.body if getattr(n, "name", None) == name)
    module = ast.Module(body=[node], type_ignores=[])
    namespace.update(torch=torch, nn=torch.nn, logging=logging, partial=partial)
    exec(compile(module, str(path), "exec"), namespace)
    return namespace[name]


class PrefillDefaultsTest(unittest.TestCase):
    def test_hc_tiles_follow_device_mma_alignment(self):
        geometry = definition("models/dsv4/ppu_hc_prenorm.py", "launch_geometry")
        for is_ppu1v5, block_n in ((False, 32), (True, 24)):
            for rows in (1, 256, 257, 512, 513, 1024, 2049, 8192, 65536):
                with self.subTest(is_ppu1v5=is_ppu1v5, rows=rows):
                    config = geometry(rows, 16384, 24, is_ppu1v5=is_ppu1v5)
                    self.assertEqual(config["BLOCK_N"], block_n)
                    self.assertEqual(config["FAST_BF16_TO_TF32"], "true")
                    self.assertEqual(config["BLOCK_M"], 64)
                    self.assertEqual(config["BLOCK_K"], 64)

    def hc(self, options=None, **kwargs):
        kernel = SimpleNamespace(mhc_pre_big_fuse=Mock(), mhc_post_fwd=Mock())
        cls = definition(
            "models/dsv4/ppu_hc.py",
            "PpuHCUnit",
            TileLangHCUnit=object,
            importlib=SimpleNamespace(import_module=lambda _: kernel),
        )
        modules = {
            "rtp_llm.models_py.modules.dsv4": SimpleNamespace(
                tilelang_kernels=object()
            ),
            "rtp_llm.platforms.ppu.models.dsv4.ppu_hc_prenorm": SimpleNamespace(
                tf32_hc_prenorm_gemm=Mock()
            ),
        }
        with patch.dict("sys.modules", modules):
            return cls(
                options=options or {}, tp_size=kwargs.pop("tp_size", 4), **kwargs
            )

    def test_hc_defaults_and_explicit_opt_out(self):
        for options, expected in [
            ({}, True),
            (
                {
                    "DSV4_PPU_PREFILL_HC_QUANT": "0",
                    "DSV4_PPU_PREFILL_HC_POST": "0",
                },
                False,
            ),
        ]:
            hc = self.hc(options)
            self.assertEqual(hc._prefill_quant, expected)
            self.assertEqual(hc._prefill_post, expected)
        decode = self.hc(allow_graph=True)
        self.assertFalse(decode._prefill_quant)
        self.assertFalse(decode._prefill_post)
        self.assertFalse(self.hc(tp_size=1)._prefill_post)

    def test_hc_unsupported_inputs_return_to_original_norm(self):
        hc = self.hc(fuse_norm=True)
        norm = SimpleNamespace(
            weight=torch.ones(4096, dtype=torch.bfloat16), variance_epsilon=1e-6
        )
        for shape, tp in [
            ((8, 4, 4096), 4),
            ((1024, 4, 4096), 1),
            ((1024, 2, 4096), 4),
        ]:
            x = torch.empty(shape, dtype=torch.bfloat16)
            self.assertIsNone(
                hc.prefill_fast_pre_norm_quantized(x, norm, tp_size=tp, tp_rank=0)
            )
        hc._prenorm_gemm.assert_not_called()

    def moe(self, options=None, **overrides):
        options = options or {}
        strategy = Mock(routed_tp_size=overrides.get("tp_size", 4), name="grouped")
        shared = SimpleNamespace(
            w13=SimpleNamespace(weight=torch.empty(1)),
            w2=SimpleNamespace(weight=torch.empty(1)),
        )
        cls = definition(
            "models/dsv4/ppu_tp_moe.py",
            "PpuTPMoE",
            Gate=lambda *a, **k: Mock(),
            MoeCfg=lambda **k: k,
            PpuGroupedFP4Strategy=lambda *a, **k: strategy,
            PpuTPSharedExpert=lambda *a, **k: shared,
            all_reduce=lambda x, **k: x,
            Group=SimpleNamespace(TP="tp"),
            W=SimpleNamespace(
                **{
                    k: k
                    for k in (
                        "v4_shared_w13_w",
                        "v4_shared_w13_s",
                        "v4_shared_w2_w",
                        "v4_shared_w2_s",
                    )
                }
            ),
            record_function_range=lambda _: nullcontext(),
            combine_tp_partials=lambda routed, shared, scale: routed * scale + shared,
        )
        args = dict(
            layer_id=0,
            dim=4096,
            moe_inter_dim=2048,
            n_routed_experts=256,
            n_activated_experts=8,
            n_shared_experts=1,
            score_func="sigmoid",
            route_scale=2.0,
            swiglu_limit=0.0,
            n_hash_layers=3,
            vocab_size=100,
            layer_weights={
                k: object()
                for k in (
                    "v4_shared_w13_w",
                    "v4_shared_w13_s",
                    "v4_shared_w2_w",
                    "v4_shared_w2_s",
                )
            },
            tp_size=4,
            ep_size=1,
            ep_rank=0,
            max_tokens_per_rank=65536,
            is_decode_role=False,
            tp_rank=0,
            platform_provider=SimpleNamespace(
                _bool=lambda k, default: options.get(k, default)
            ),
        )
        args.update(overrides)
        return cls(**args)

    def test_moe_defaults_and_topology_fallback(self):
        self.assertTrue(self.moe()._shared_phases)
        for options in (
            {"DSV4_PPU_PREFILL_MOE_SHARED_PHASES": False},
            {"DSV4_MOE_GATHER_SHARED_FUSED": False},
        ):
            self.assertFalse(self.moe(options)._shared_phases)
        for kwargs in (
            {"tp_size": 1},
            {"ep_size": 4},
            {"dim": 8192},
            {"moe_inter_dim": 4096},
        ):
            self.assertFalse(self.moe(**kwargs)._shared_phases)
        with self.assertRaisesRegex(ValueError, "Prefill"):
            self.moe(is_decode_role=True)

    def test_serial_fallback_preserves_both_gather_modes(self):
        x = torch.ones(3, 8, dtype=torch.bfloat16)
        for fused in (False, True):
            moe = self.moe({"DSV4_MOE_GATHER_SHARED_FUSED": fused})
            moe.shared_experts = lambda x: x * 3
            calls = []

            def routed(x, weights, indices, **kwargs):
                calls.append(kwargs)
                return (
                    x * 2 * kwargs["route_scale"] + kwargs["shared"]
                    if kwargs
                    else x * 2
                )

            moe._strategy = routed
            actual = moe._forward_serial(x, None, None)
            torch.testing.assert_close(actual, x * 7, rtol=0, atol=0)
            self.assertEqual(len(calls), 1)
            self.assertNotIn("shared_schedule", calls[0])
            self.assertIsNone(moe._shared_stream)

    def test_small_chunks_use_serial_without_creating_streams(self):
        moe = self.moe()
        moe.gate = lambda *a, **k: (None, None)
        moe._forward_serial = Mock(return_value="result")
        moe._forward_phased = Mock()
        recorder = SimpleNamespace(should_record_layer=lambda _: False)
        with patch.dict(
            "sys.modules",
            {
                "rtp_llm.models_py.modules.dsv4": SimpleNamespace(
                    _record_tensor=recorder
                )
            },
        ):
            for rows in (1, 1023, 65537):
                x = SimpleNamespace(is_cuda=True, shape=(rows, 4096))
                self.assertEqual(moe._forward_local_chunk(x, None), "result")
        self.assertEqual(moe._forward_serial.call_count, 3)
        moe._forward_phased.assert_not_called()

    def test_phased_failure_joins_auxiliary_stream(self):
        moe = self.moe()
        stream, current = Mock(), Mock()
        moe._shared_stream = stream
        moe._strategy = Mock(side_effect=ValueError("routed failure"))
        moe._forward_phased.__func__.__globals__["_SharedPhases"] = (
            lambda *a: SimpleNamespace(output="shared")
        )
        with patch.dict(
            "sys.modules",
            {"rtp_llm.platforms.ppu.runtime": SimpleNamespace(PpuStreamPool=Mock())},
        ), patch.object(torch.cuda, "current_stream", return_value=current):
            with self.assertRaisesRegex(ValueError, "routed failure"):
                moe._forward_phased(SimpleNamespace(device="cuda:0"), None, None)
        current.wait_stream.assert_called_once_with(stream)

    def test_post_launch_opt_out_and_shape_fallback(self):
        factory = Mock(return_value=Mock())
        post = definition(
            "kernels/hc/post_kernel.py", "mhc_post_fwd", _mhc_post_fwd=factory
        )
        for rows, tune, optimized in (
            (1024, True, True),
            (1024, False, False),
            (1, True, False),
        ):
            residual = torch.empty((1, rows, 4, 4096), dtype=torch.bfloat16)
            post(
                torch.empty((1, rows, 4096), dtype=torch.bfloat16),
                residual,
                torch.empty((1, rows, 4, 1)),
                torch.empty((1, rows, 4, 4)),
                enable_pdl=False,
                prefill_tuned=tune,
            )
            self.assertEqual("n_thr" in factory.call_args.kwargs, optimized)

    def test_phase_dependencies_keep_gemms_apart(self):
        events = []

        class Stream:
            def __init__(self, name):
                self.name = name

            def wait_stream(self, other):
                events.append((self.name, "wait", other.name))

            def record_event(self):
                event = len(events)
                events.append((self.name, "record", event))
                return event

            def wait_event(self, event):
                events.append((self.name, "join", event))

        main, aux = Stream("main"), Stream("shared")
        tensor = SimpleNamespace(record_stream=lambda _: None, device="cuda:0")
        cls = definition(
            "models/dsv4/ppu_shared_phases.py",
            "SharedPhases",
            silu_mul_merged_bf16=lambda x, _: events.append("activation") or x,
        )
        shared = SimpleNamespace(
            swiglu_limit=0.0,
            w13=lambda x: events.append("shared_up") or x,
            w2=SimpleNamespace(
                quantize_input=lambda x: (x, x),
                forward_quantized=lambda *a, **k: events.append("shared_down"),
            ),
        )
        with patch.object(
            torch.cuda, "current_stream", return_value=main
        ), patch.object(torch.cuda, "stream", return_value=nullcontext()), patch.object(
            torch, "empty_like", return_value=tensor
        ):
            phases = cls(shared, tensor, aux)
            events.append("routed_pack")
            phases.before_up()
            events.append("routed_up")
            phases.after_up()
            events.append("routed_activation")
            phases.before_down()
            events.append("routed_down")
        self.assertEqual(
            events,
            [
                ("shared", "wait", "main"),
                "shared_up",
                ("shared", "record", 2),
                "routed_pack",
                ("main", "join", 2),
                "activation",
                "routed_up",
                ("shared", "wait", "main"),
                "shared_down",
                ("shared", "record", 9),
                "routed_activation",
                ("main", "join", 9),
                "routed_down",
            ],
        )


if __name__ == "__main__":
    unittest.main()

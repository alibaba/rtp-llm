"""Tiny split and stacked Qwen4 checkpoint loads, including pure FFN TP."""

import os
import tempfile
import unittest
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from rtp_llm.models_py.model_loader import NewLoaderConfig, NewModelLoader
from rtp_llm.models_py.module_base import RtpModule
from rtp_llm.models_py.new_models.qwen4_exp.expert_factory import make_qwen4_experts
from rtp_llm.models_py.registry import register_model
from rtp_llm.models_py.weight_mapper import WeightsMapper


@register_model("_qwen4_source_expert_fixture")
class _Owner(RtpModule):
    WEIGHTS_MAPPER = WeightsMapper(
        prefix_mapping={"model.language_model.layers.0.mlp.": ""}
    )

    def __init__(self, config, load):
        super().__init__()
        text = {
            "num_experts": 2,
            "moe_intermediate_size": 256,
            "num_hidden_layers": 1,
            "num_experts_per_tok": 1,
        }
        with safe_open(config["file"], framework="pt") as shard:
            shapes = {
                key: tuple(shard.get_slice(key).get_shape()) for key in shard.keys()
            }
        plan = SimpleNamespace(
            hidden_size=128,
            positive=text.__getitem__,
            text=text,
            model_config=config,
            prefix="model.language_model.",
            source_kind=config["source_kind"],
            shapes=shapes,
        )
        self.experts = make_qwen4_experts(plan, 0, load, prefix="experts")
        if "model.language_model.layers.0.mlp.shared_expert_gate.weight" in shapes:
            self.shared_expert_gate = torch.nn.Linear(
                128, 1, bias=False, dtype=torch.bfloat16
            )

    def load_weights(self, weights):
        return super().load_weights(self.WEIGHTS_MAPPER.apply(weights))


def _source(mixed, stacked=False):
    payload, ref = {}, {}
    prefix = "model.language_model.layers.0.mlp.experts."
    for e in range(2):
        for i, proj in enumerate(("up_proj", "gate_proj", "down_proj")):
            h, w = (128, 256) if proj == "down_proj" else (256, 128)
            r = torch.arange(h * w).reshape(h, w)
            # Different row/column ranges exercise the sliced-out shard's amax.
            raw = ((r % 11).float() - 5) * (1 + (r // 128) % 3) * (e + 1) * (i + 1)
            name = f"{prefix}{e}.{proj}.weight"
            if mixed:
                raw = raw.to(torch.float8_e4m3fn)
                scale = (
                    torch.arange((h // 128) * (w // 128))
                    .reshape(h // 128, w // 128)
                    .float()
                    + 1
                )
                payload[name + "_scale_inv"] = scale.bfloat16()
                ref[e, proj] = (
                    raw.float()
                    * scale.repeat_interleave(128, 0).repeat_interleave(128, 1)
                ).bfloat16()
                payload[name] = raw
            else:
                ref[e, proj] = payload[name] = raw.bfloat16()
    if stacked:
        payload = {
            prefix
            + "gate_up_proj": torch.stack(
                [torch.cat((ref[e, "gate_proj"], ref[e, "up_proj"])) for e in range(2)]
            ),
            prefix + "down_proj": torch.stack([ref[e, "down_proj"] for e in range(2)]),
        }
    return payload, ref


def _load(directory, kind, rank, method, ep=1):
    config = {
        "model_type": "_qwen4_source_expert_fixture",
        "source_kind": kind,
        "file": str(Path(directory) / "model.safetensors"),
        "num_experts": 2,
        "moe_k": 1,
    }
    load = NewLoaderConfig(
        device="cpu",
        compute_dtype=torch.bfloat16,
        load_method=method,
        tp_size=2,
        tp_rank=rank,
        ep_size=ep,
        ep_rank=rank if ep == 2 else 0,
        moe_config=SimpleNamespace(
            moe_strategy="nvfp4_cutlass" if kind == "fp8" else "auto", fp4_moe_op="auto"
        ),
    )
    loader = NewModelLoader(config, load, model_path=directory)
    owner = loader.load().experts
    assert loader._selected_load_method.value == method
    return owner


def _round_nvfp4(mat):
    # Independent nearest/even E2M1 grid, per full expert (before TP slicing).
    mat = mat.float()
    g = mat.abs().amax().clamp_min(1.0e-8) / (448 * 6)
    groups = mat.reshape(*mat.shape[:-1], -1, 16)
    scale = groups.abs().amax(-1) / (g * 6)
    scale = torch.where(scale == 0, 1.0, scale).to(torch.float8_e4m3fn)
    scaled = (
        (groups / (g * scale.float().unsqueeze(-1))).clamp(-6, 6).reshape(mat.shape)
    )
    levels = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
    delta = (scaled.abs().unsqueeze(-1) - levels).abs()
    a = delta.argmin(-1)
    b = (a + 1).clamp(max=7)
    tie = delta.gather(-1, a.unsqueeze(-1)) == delta.gather(-1, b.unsqueeze(-1))
    a = torch.where(tie.squeeze(-1) & (a % 2 == 1), b, a)
    codes = (a | ((scaled < 0).long() << 3)).byte()
    return codes[:, ::2] | (codes[:, 1::2] << 4), scale.view(torch.uint8), g


class Q4SourceExpertTest(unittest.TestCase):
    @staticmethod
    def _nvfp4_owner(rank, strategy, op, device="cpu"):
        from rtp_llm.config.quant_config import ModelOptFp4Config

        config = SimpleNamespace(
            model_type="qwen4_exp", num_layers=1, moe_k=1, activation_type="SiGLU",
            quant_config=ModelOptFp4Config(bits=4, group_size=16, is_quanted=True),
        )
        plan = SimpleNamespace(
            source_kind="nvfp4", model_config=config, hidden_size=128, prefix="", shapes={},
            positive={"moe_intermediate_size": 640, "num_experts": 1}.__getitem__,
        )
        load = NewLoaderConfig(
            device=device, tp_size=2, tp_rank=rank, ep_size=1, ep_rank=0,
            compute_dtype=torch.bfloat16,
            moe_config=SimpleNamespace(moe_strategy=strategy, fp4_moe_op=op),
        )
        return make_qwen4_experts(plan, 0, load)

    def test_prequantized_tp2_marlin_slices_weights_and_scales_together(self):
        owners = [self._nvfp4_owner(rank, "nvfp4_marlin", "auto") for rank in (0, 1)]
        # Positive E2M1 nibbles; weights and per-group scales both vary across
        # the TP boundary. Compare two local SwiGLUs to a full-checkpoint oracle.
        lut = torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6.])

        def decode(w, s):
            codes = torch.stack((w & 15, w >> 4), -1).flatten(-2)
            return lut[codes.long()] * s.float().repeat_interleave(16, -1)

        x = torch.linspace(0.01, 0.02, 128)
        dense = {}
        rng = torch.Generator().manual_seed(1024)
        for proj in ("up", "gate", "down"):
            rows, cols = (128, 640) if proj == "down" else (640, 128)
            w = torch.randint(0, 8, (rows, cols // 2), dtype=torch.uint8, generator=rng)
            w |= torch.randint(0, 8, w.shape, dtype=torch.uint8, generator=rng) << 4
            s = torch.randint(1, 5, (rows, cols // 16), generator=rng).to(torch.float8_e4m3fn)
            dense[proj] = decode(w, s)
            for suffix, value in (("weight", w), ("weight_scale", s),
                                  ("input_scale", torch.tensor(1.)), ("weight_scale_2", torch.tensor(1.))):
                key = f"0.{proj}_proj.{suffix}"
                for rank, leaf in enumerate(owners):
                    # Exercise direct full-source and range-reader dispatch.
                    data = leaf.checkpoint_read_spec(key, tuple(value.shape)).materialize(value) if rank else value
                    leaf.load_weights({key: data})
        outputs = []
        for leaf in owners:
            leaf.validate_weights_loaded()
            self.assertEqual((leaf.local_n, leaf.n), (320, 384))
            up, gate = decode(leaf.w13[0], leaf.s13[0]).chunk(2, dim=0)
            down = decode(leaf.w2[0], leaf.s2[0])
            outputs.append(down @ (torch.nn.functional.silu(gate @ x) * (up @ x)))
        reference = dense["down"] @ (torch.nn.functional.silu(dense["gate"] @ x) * (dense["up"] @ x))
        torch.testing.assert_close(outputs[0] + outputs[1], reference)
        if os.environ.get("RUN_Q4_NEWLOADER_MARLIN") == "1":
            # Execute both real TP shards on the allocated SM90/120 device,
            # then sum partials exactly as the direct TP router does.
            h = x.bfloat16().cuda().reshape(1, -1)
            ids = torch.zeros((1, 1), dtype=torch.int32, device=h.device)
            weights = torch.ones((1, 1), device=h.device)
            partials = []
            for leaf in owners:
                leaf.to(h.device)
                leaf.process_weights_after_loading()
                partials.append(leaf(h, ids, weights).clone())
            result = (partials[0] + partials[1]).cpu().float()[0]
            self.assertLess(((result - reference).norm() / reference.norm()).item(), 0.05)

    def test_prequantized_qwen4_uses_requested_or_architecture_backend(self):
        from rtp_llm.models_py.new_models.qwen4_exp.nvfp4_experts import Qwen4Nvfp4Experts

        for sm, expected in (((9, 0), "marlin"), ((12, 0), "marlin"), ((10, 0), "trtllm")):
            with patch("torch.cuda.get_device_capability", return_value=sm):
                layer = self._nvfp4_owner(0, "auto", "auto", "cuda:0")
                if expected == "marlin":
                    self.assertIsInstance(layer, Qwen4Nvfp4Experts)
                else:
                    self.assertEqual(layer.quant_method._backend, expected)
        layer = self._nvfp4_owner(0, "fp4_no_dp", "auto")
        self.assertEqual(layer.quant_method._backend, "trtllm")
        layer.load_weights({"0.gate_proj.weight_scale_2": torch.tensor(1.)})
        with self.assertRaisesRegex(ValueError, "equal up/gate outer scales"):
            layer.load_weights({"0.up_proj.weight_scale_2": torch.tensor(2.)})
        with self.assertRaisesRegex(ValueError, "requires FP4_MOE_OP"):
            self._nvfp4_owner(0, "fp4_no_dp", "cutedsl")

    def test_only_this_layers_shared_gate_reaches_factory_adapter(self):
        for present in (False, True):
            values, _ = _source(True)
            if present:
                values[
                    "model.language_model.layers.0.mlp.shared_expert_gate.weight"
                ] = torch.ones(1, 128, dtype=torch.bfloat16)
            with tempfile.TemporaryDirectory() as directory:
                save_file(values, str(Path(directory) / "model.safetensors"))
                leaf = _load(directory, "fp8", 0, "scratch")
                adapter = SimpleNamespace(physical_expert_num=2)
                leaf._configure_factory_adapter(adapter)
                self.assertIs(adapter.has_shared_expert_gate, present)
                self.assertEqual(adapter.expert_num, 2)

    def test_two_gloo_ranks_native_fst_matches_scratch_including_ep(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cases = []
            for name, kind, stacked, ep in (
                ("stacked_tp", "bf16", True, 1),
                ("stacked_ep", "bf16", True, 2),
                ("mixed_tp", "fp8", False, 1),
                ("mixed_ep", "fp8", False, 2),
            ):
                folder = root / name
                folder.mkdir()
                tensors, _ = _source(kind == "fp8", stacked)
                save_file(tensors, str(folder / "model.safetensors"))
                cases.append((str(folder), kind, ep))
            torch.multiprocessing.spawn(
                _run_two_ranks, args=((root / "rendezvous").as_uri(), cases), nprocs=2
            )

    def test_bf16_split_and_stacked_source(self):
        for stacked in (False, True):
            values, ref = _source(False, stacked=stacked)
            with tempfile.TemporaryDirectory() as directory:
                save_file(values, str(Path(directory) / "model.safetensors"))
                for rank in range(2):
                    leaf = _load(directory, "bf16", rank, "scratch")
                    self.assertIsNone(leaf.runtime)
                    for e in range(2):
                        up, gate = (
                            ref[e, p].chunk(2, dim=0)[rank]
                            for p in ("up_proj", "gate_proj")
                        )
                        torch.testing.assert_close(leaf.w13[e], torch.cat((up, gate)))
                        torch.testing.assert_close(
                            leaf.w2[e], ref[e, "down_proj"].chunk(2, dim=1)[rank]
                        )

    def test_mixed_fp8_cutlass_raw_packed_tp2_before_device_swizzle(self):
        values, ref = _source(True)
        with tempfile.TemporaryDirectory() as directory:
            save_file(values, str(Path(directory) / "model.safetensors"))
            for rank in range(2):
                leaf = _load(directory, "fp8", rank, "scratch")
                self.assertIsNone(leaf.runtime)
                for e in range(2):
                    w13, s13, g13 = _round_nvfp4(
                        torch.cat((ref[e, "up_proj"], ref[e, "gate_proj"]))
                    )
                    w2, s2, g2 = _round_nvfp4(ref[e, "down_proj"])
                    for actual, full in (
                        (leaf.w13[e], w13),
                        (leaf.w13_scale[e].view(torch.uint8), s13),
                    ):
                        up, gate = full.chunk(2, 0)
                        torch.testing.assert_close(
                            actual,
                            torch.cat((up.chunk(2, 0)[rank], gate.chunk(2, 0)[rank])),
                        )
                    torch.testing.assert_close(leaf.w2[e], w2.chunk(2, 1)[rank])
                    torch.testing.assert_close(
                        leaf.w2_scale[e].view(torch.uint8), s2.chunk(2, 1)[rank]
                    )
                    torch.testing.assert_close(leaf.w13_scale_2[e], g13)
                    torch.testing.assert_close(leaf.w2_scale_2[e], g2)
                    self.assertAlmostEqual(
                        leaf.w13_input_scale[e].item(), 32 / (6 * 448)
                    )

    def test_mixed_rejects_incomplete_or_wrong_source(self):
        for which in ("missing", "dtype", "shape"):
            values, _ = _source(True)
            k = next(k for k in values if k.endswith("_scale_inv"))
            if which == "missing":
                del values[k]
            elif which == "dtype":
                key = k[: -len("_scale_inv")]
                values[key] = values[key].bfloat16()
            else:
                values[k] = torch.ones(3, 8)
            with tempfile.TemporaryDirectory() as directory:
                save_file(values, str(Path(directory) / "model.safetensors"))
                with self.assertRaises((ValueError, RuntimeError, TypeError)):
                    _load(directory, "fp8", 0, "scratch")


def _run_two_ranks(rank, rendezvous, cases):
    torch.distributed.init_process_group(
        "gloo",
        init_method=rendezvous,
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=45),
    )
    try:
        for folder, kind, ep in cases:
            expected = _load(folder, kind, rank, "scratch", ep)
            actual = _load(folder, kind, rank, "fastsafetensors", ep)
            assert actual.local == 2 // ep
            assert actual.runtime is None
            for name, tensor in expected.state_dict().items():
                other = actual.state_dict()[name]
                assert tensor.dtype == other.dtype
                torch.testing.assert_close(
                    tensor.contiguous().view(torch.uint8),
                    other.contiguous().view(torch.uint8),
                )
    finally:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    unittest.main()

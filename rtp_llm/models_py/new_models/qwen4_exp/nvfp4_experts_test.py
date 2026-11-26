"""Real tiny NVFP4 split-source checkpoint tests; no GPU kernel is mocked."""

import gc
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from safetensors.torch import save_file

from rtp_llm.models_py.checkpoint_reader_test_support import requires_native_fst
from rtp_llm.models_py.model_loader import NewLoaderConfig, NewModelLoader
from rtp_llm.models_py.module_base import RtpModule
from rtp_llm.models_py.new_models.qwen4_exp.nvfp4_experts import \
    Qwen4Nvfp4Experts
from rtp_llm.models_py.registry import register_model
from rtp_llm.models_py.weight_mapper import WeightsMapper


@register_model("_qwen4_nvfp4_expert_cpu_test")
class _Layer(RtpModule):
    WEIGHTS_MAPPER = WeightsMapper(
        prefix_mapping={"model.language_model.layers.0.mlp.": ""}
    )

    def load_weights(self, weights):
        return super().load_weights(self.WEIGHTS_MAPPER.apply(weights))

    def __init__(self, config, load):
        super().__init__()
        self.experts = Qwen4Nvfp4Experts(128, 256, 4, load.ep_size, load.ep_rank)
        self.norm = torch.nn.Parameter(torch.empty(128, dtype=torch.bfloat16))


    def resident_constructor_state(self):
        self.experts.validate_weights_loaded()
        return {"inputs": sorted(self.experts._loaded)}

    def prepare_resident_checkpoint_restore(self, state, values):
        self.experts._loaded = set(state["inputs"])


def _payload():
    payload = {"norm": torch.arange(128).bfloat16()}
    for e in range(4):
        for p, projection in enumerate(("up", "gate", "down")):
            prefix = f"experts.{e}.{projection}_proj."
            rows, cols = (256, 128) if p < 2 else (128, 256)
            payload[prefix + "weight"] = torch.full(
                (rows, cols // 2), e * 16 + p, dtype=torch.uint8
            )
            payload[prefix + "weight_scale"] = torch.full(
                (rows, cols // 16), 1 + e + p
            ).to(torch.float8_e4m3fn)
            payload[prefix + "input_scale"] = torch.tensor([1.0 + e + p])
            payload[prefix + "weight_scale_2"] = torch.tensor(1.0 + e + (p == 2))
    return payload


def _load(directory, size=1, rank=0, method="scratch", device="cpu", **resident):
    model_config = SimpleNamespace(
        model_type="_qwen4_nvfp4_expert_cpu_test", num_experts=4, layer_num=1, moe_k=2
    )
    load = NewLoaderConfig(
        device=device,
        load_method=method,
        compute_dtype=torch.bfloat16,
        tp_size=size,
        tp_rank=rank,
        ep_size=size,
        ep_rank=rank,
    )
    loader = NewModelLoader(model_config, load, model_path=directory)
    model = loader.load(**resident)
    assert loader._selected_load_method.value == method
    return model


class Qwen4Nvfp4ExpertTest(unittest.TestCase):
    def _save(self, directory, values):
        save_file(
            {
                (
                    "model.language_model.layers.0.mlp." + key
                    if key.startswith("experts.")
                    else key
                ): value
                for key, value in values.items()
            },
            str(Path(directory) / "model.safetensors"),
        )

    def test_scratch_real_checkpoint_contiguous_ep(self):
        values = _payload()
        with tempfile.TemporaryDirectory() as directory:
            self._save(directory, values)
            for rank in range(2):
                model = _load(directory, 2, rank)
                self.assertEqual(model.norm.dtype, torch.bfloat16)
                torch.testing.assert_close(model.norm, values["norm"])
                leaf = model.experts
                for i in range(2):
                    e = 2 * rank + i
                    for p, proj in enumerate(("up", "gate", "down")):
                        prefix = f"experts.{e}.{proj}_proj."
                        w = (
                            leaf.w2[i]
                            if p == 2
                            else leaf.w13[i, p * 256 : (p + 1) * 256]
                        )
                        s = (
                            leaf.s2[i]
                            if p == 2
                            else leaf.s13[i, p * 256 : (p + 1) * 256]
                        )
                        torch.testing.assert_close(w, values[prefix + "weight"])
                        torch.testing.assert_close(
                            s.view(torch.uint8),
                            values[prefix + "weight_scale"].view(torch.uint8),
                        )
                        self.assertEqual(
                            leaf.input_scales[i, p],
                            values[prefix + "input_scale"].item(),
                        )
                        self.assertEqual(
                            leaf.outer_scales[i, p],
                            values[prefix + "weight_scale_2"].item(),
                        )
                with self.assertRaisesRegex(RuntimeError, "require initialized Marlin"):
                    leaf(None, None, None)

    def test_invalid_checkpoint_is_not_published(self):
        for fault in (
            "missing",
            "packed_dtype",
            "outer_mismatch",
            "bad_scale",
            "bad_shape",
            "bad_name",
        ):
            with self.subTest(fault=fault), tempfile.TemporaryDirectory() as directory:
                values = _payload()
                prefix = "experts.1.up_proj."
                if fault == "missing":
                    del values[prefix + "input_scale"]
                elif fault == "packed_dtype":
                    values[prefix + "weight"] = values[prefix + "weight"].to(torch.int8)
                elif fault == "outer_mismatch":
                    values[prefix + "weight_scale_2"] *= 2
                elif fault == "bad_scale":
                    values[prefix + "weight_scale"].view(torch.uint8)[0, 0] = 255
                elif fault == "bad_shape":
                    values[prefix + "weight"] = values[prefix + "weight"][
                        1:
                    ].contiguous()
                else:
                    values["experts.4.up_proj.input_scale"] = torch.tensor(1.0)
                self._save(directory, values)
                with self.assertRaises((ValueError, RuntimeError)):
                    _load(directory)

    def test_split_source_readspec_skips_remote_experts(self):
        leaf = Qwen4Nvfp4Experts(128, 256, 4, 2, 1)
        for e in range(4):
            self.assertEqual(
                leaf.checkpoint_read_spec(f"{e}.gate_proj.weight", (256, 64)).load,
                e >= 2,
            )
        with self.assertRaisesRegex(ValueError, "Unknown"):
            leaf.checkpoint_read_spec("8.up_proj.weight", (256, 64))

    @requires_native_fst
    def test_native_cpu_fst_closes_without_invalidating_owned_storage(self):
        with tempfile.TemporaryDirectory() as directory:
            self._save(directory, _payload())
            expected = _load(directory)
            actual = _load(directory, method="fastsafetensors")
        gc.collect()
        for name, param in expected.named_parameters():
            other = actual.get_parameter(name)
            self.assertEqual(param.dtype, other.dtype)
            torch.testing.assert_close(param.view(torch.uint8), other.view(torch.uint8))

    @unittest.skipUnless(
        os.environ.get("RUN_Q4_NEWLOADER_MARLIN") == "1",
        "requires the H20 CUDA 13 Marlin test target",
    )
    def test_restart_maps_packed_marlin_weights_without_repacking(self):
        from unittest.mock import patch
        from uuid import uuid4

        from rtp_llm.resident_weights import ResidentWeights, store

        name = "rtp_llm_q4_marlin." + uuid4().hex
        device = f"cuda:{torch.cuda.current_device()}"
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, RTP_LLM_RESIDENT_STORE=name):
            self._save(directory, _payload())
            store.ensure_running(name, owner_pid=os.getpid())
            try:
                x = torch.ones(1, 128, dtype=torch.bfloat16, device=device)
                ids = torch.zeros((1, 1), dtype=torch.int32, device=device)
                prob = torch.ones((1, 1), device=device)
                for hot in (False, True):
                    manager = ResidentWeights("target", 0, {"layout": "direct_marlin"}, False)
                    record = manager.acquire_for_load(device, synchronize_ranks=False)
                    self.assertEqual(record is not None, hot)
                    if hot:
                        with patch(
                            "rtp_llm.models_py.kernels.cuda.marlin_nvfp4.marlin_nvfp4.prepare_weight",
                            side_effect=AssertionError("hot reload repacked a resident weight"),
                        ):
                            model = _load(directory, device=device, resident=manager, restored=record)
                    else:
                        model = _load(directory, device=device, resident=manager)
                        Path(directory, "model.safetensors").unlink()
                    output = model.experts(x, ids, prob).cpu()
                    self.assertIsNotNone(manager.allocation_of(model.experts._marlin.w1))
                    if hot:
                        torch.testing.assert_close(output, expected, rtol=0, atol=0)
                    else:
                        expected = output
                    manager.finalize()
                    del model
                    manager.release_attach()
                    del manager
            finally:
                store.stop_store(name)

    @unittest.skipUnless(
        os.environ.get("RUN_Q4_NEWLOADER_MARLIN") == "1",
        "requires the H20 CUDA 13 Marlin test target",
    )
    def test_gpu_marlin_output_against_dequantized_reference(self):
        # Nonuniform raw E2M1 nibbles and distinct gate/up prevent accidentally
        # matching after a transpose or half swap.
        rng = torch.Generator().manual_seed(14)
        values = _payload()
        codebook = torch.tensor(
            [0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6]
        )
        for name, val in values.items():
            if name.endswith(".weight"):
                val.random_(0, 256, generator=rng)
            elif name.endswith("weight_scale_2"):
                val.fill_(0.0625)
        with tempfile.TemporaryDirectory() as directory:
            self._save(directory, values)
            leaf = _load(directory, device="cuda").experts
        x = torch.randn(4, 128, generator=rng).bfloat16()
        ids = torch.arange(4, dtype=torch.int32).reshape(4, 1)
        weights = torch.tensor([[0.25], [0.5], [0.75], [1.0]])
        output = leaf(x.cuda(), ids.cuda(), weights.cuda()).cpu().float()
        reference = []
        for e in range(4):

            def dense(proj):
                base = f"experts.{e}.{proj}_proj."
                q = values[base + "weight"]
                codes = torch.stack((q & 15, q >> 4), -1).flatten(-2).long()
                return (
                    codebook[codes]
                    * values[base + "weight_scale"].float().repeat_interleave(16, -1)
                    * values[base + "weight_scale_2"]
                )

            hidden = x[e].float()
            up, gate = dense("up") @ hidden, dense("gate") @ hidden
            reference.append(
                (dense("down") @ (torch.nn.functional.silu(gate) * up)) * weights[e]
            )
        ref = torch.stack(reference)
        self.assertLess(((output - ref).norm() / ref.norm()).item(), 0.05)


if __name__ == "__main__":
    unittest.main()

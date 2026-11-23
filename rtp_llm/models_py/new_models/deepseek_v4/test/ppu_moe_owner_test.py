"""CPU checkpoint to native PPU operand contract; never launch CUDA as PPU."""

import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

import torch
from safetensors.torch import save_file

from rtp_llm.models_py.model_loader import NewLoaderConfig, NewModelLoader
from rtp_llm.models_py.module_base import RtpModule
from rtp_llm.models_py.modules.dsv4.ppu.moe.packed_inputs import (
    PpuRoutedInputs,
    shared_expert_inputs,
)
from rtp_llm.models_py.new_models.deepseek_v4.ppu_moe import create_ppu_moe
from rtp_llm.models_py.registry import MODEL_REGISTRY


class Fixture(RtpModule):
    def __init__(self, config, load_config):
        super().__init__()
        self.ffn = create_ppu_moe(config, 0, getattr(config, "provider", object()))


def config(tp, rank, ep):
    return SimpleNamespace(
        model_type="v4_ppu_moe_fixture",
        dim=512,
        moe_inter_dim=256,
        n_routed_experts=4,
        n_activated_experts=2,
        n_shared_experts=1,
        score_func="sqrtsoftplus",
        route_scale=1.0,
        swiglu_limit=0.0,
        n_hash_layers=1,
        vocab_size=3,
        n_physical_experts=4,
        dp_size=1,
        tp_size=tp,
        tp_rank=rank,
        ep_size=ep,
        ep_rank=rank % ep,
    )


def bytes_tensor(rows, cols, dtype, offset):
    i, j = torch.arange(rows)[:, None], torch.arange(cols)[None, :]
    return ((i // 128 + i * 3 + j + offset) % 17 + 117).to(torch.uint8).view(dtype)


def checkpoint():
    state = {
        "ffn.gate.weight": torch.arange(4 * 512).reshape(4, 512).bfloat16(),
        "ffn.gate.tid2eid": torch.tensor([[1, 2], [3, 0], [1, 0]]),
    }
    for idx, p in enumerate(("w1", "w2", "w3")):
        n, k = (512, 256) if p == "w2" else (256, 512)
        for e in range(4):
            key = f"ffn.experts.{e}.{p}."
            state[key + "weight"] = bytes_tensor(n, k // 2, torch.int8, e + idx)
            state[key + "scale"] = bytes_tensor(
                n, k // 32, torch.float8_e8m0fnu, e + idx
            )
        key = f"ffn.shared_experts.{p}."
        state[key + "weight"] = bytes_tensor(n, k, torch.float8_e4m3fn, idx)
        state[key + "scale"] = bytes_tensor(
            n // 128, k // 128, torch.float8_e8m0fnu, idx
        )
    return state


class PpuMoEOwnerTest(unittest.TestCase):

    def test_ft_local_projection_is_not_sharded_twice(self):
        from rtp_llm.models_py.modules.dsv4.ppu.resources import split_shared_weight

        payload = checkpoint()
        for rank in (0, 1):
            owner = create_ppu_moe(config(2, rank, 1), 0, object())
            for attr in ("weight", "scale"):
                name = "shared_experts.w2." + attr
                full = payload["ffn." + name]
                local = split_shared_weight(
                    full, 2, rank, projection="w2", scale=attr == "scale"
                )
                owner.load_weights({name: owner.prepare_ft_projection(name, local)})
                self.assertTrue(
                    torch.equal(
                        owner._target(name).view(torch.uint8), local.view(torch.uint8)
                    )
                )
                name = "experts.1.w1." + attr
                local = payload["ffn." + name].chunk(2)[rank]
                owner.load_weights({name: owner.prepare_ft_projection(name, local)})
                self.assertTrue(
                    torch.equal(
                        owner._target(name).view(torch.uint8),
                        local.contiguous().view(torch.uint8),
                    )
                )

    def load(self, state, cfg, method):
        with tempfile.TemporaryDirectory() as directory, mock.patch.dict(
            MODEL_REGISTRY, {cfg.model_type: Fixture}
        ):
            save_file(state, os.path.join(directory, "model.safetensors"))
            loader = NewModelLoader(
                cfg,
                NewLoaderConfig(device="cpu", load_method=method),
                model_path=directory,
            )
            layer = loader.load().ffn
            self.assertEqual(loader._selected_load_method.value, method)
            return layer

    def assert_payload(self, owner, payload, cfg):
        tp = cfg.tp_size > 1 and cfg.ep_size == 1
        for name in owner._expected:
            src = payload["ffn." + name]
            if name == "gate.tid2eid":
                src = src.int()
            elif tp and not name.startswith("gate."):
                axis = 1 if ".w2." in name else 0
                src = src.chunk(cfg.tp_size, dim=axis)[cfg.tp_rank]
                if name.startswith("shared_experts.") and name.endswith(".scale"):
                    src = src.float()
            target = owner._target(name)
            self.assertEqual(src.dtype, target.dtype, name)
            torch.testing.assert_close(
                src.contiguous().view(torch.uint8),
                target.contiguous().view(torch.uint8),
                msg=name,
            )

    def test_real_tp_ep_and_single_rank_scratch_and_native_fst(self):
        state = checkpoint()
        methods = ("scratch", "fastsafetensors")
        if os.environ.get("NEWLOADER_TEST_NATIVE_FST") == "0":
            with self.assertRaisesRegex(
                RuntimeError, "fastsafetensors prerequisite failed"
            ):
                self.load(state, config(1, 0, 1), "fastsafetensors")
            methods = ("scratch",)
        for method in methods:
            for topology in ((1, 0, 1), (2, 0, 1), (2, 1, 1), (2, 1, 2)):
                with self.subTest(method=method, topology=topology):
                    cfg = config(*topology)
                    owner = self.load(state, cfg, method)
                    self.assert_payload(owner, state, cfg)
                    if cfg.ep_size == 2:
                        self.assertFalse(
                            owner.checkpoint_read_spec(
                                "experts.0.w1.weight", (256, 256)
                            ).load
                        )
                    self.assertIsNone(owner._runtime)
                    with self.assertRaisesRegex(RuntimeError, "PPU-resident"):
                        owner.bind_runtime(cfg)

    def test_shared_and_routed_executor_operands_reuse_original_storage(self):
        owner = self.load(checkpoint(), config(2, 1, 1), "scratch")
        inputs = PpuRoutedInputs(owner)
        original = tuple(getattr(owner, n) for n in inputs.keys)
        for t, before in zip(inputs.packed(), original):
            self.assertEqual(t.data_ptr(), before.data_ptr())
        inputs.consume()
        self.assertTrue(all(getattr(owner, n) is None for n in inputs.keys))
        ptr = owner.shared_s13.data_ptr()
        shared = shared_expert_inputs(owner)
        self.assertEqual(shared["w13_s"].data_ptr(), ptr)
        self.assertIsNone(owner.shared_s13)


if __name__ == "__main__":
    unittest.main()

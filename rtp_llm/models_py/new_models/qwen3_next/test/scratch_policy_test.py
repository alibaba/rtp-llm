"""Real scratch reads: optional tuning changes I/O, never rank-local values."""

import tempfile
import unittest
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from rtp_llm.models_py import weight_mapper
from rtp_llm.models_py.model_loader import NewModelLoader
from rtp_llm.models_py.new_models.qwen3_next.test.main_checkpoint_test import (
    _config,
    _dense_weights,
    _load_config,
)
from rtp_llm.models_py.read_spec import make_sharded_read_spec
from rtp_llm.utils import ckpt_file_info


class ScratchPolicyTest(unittest.TestCase):
    def test_file_sliced_offline_fp8_weights_and_scales(self):
        from rtp_llm.config.quant_config import Fp8BlockWiseQuantConfig
        from rtp_llm.models_py.layers.moe_experts import BaseMoEExperts
        from rtp_llm.models_py.quant_methods.base import QuantizationConfig
        from rtp_llm.models_py.scratch_moe_tp import read_parts

        cfg = Fp8BlockWiseQuantConfig(bits=8, group_size=128, is_quanted=True)

        def owner():
            return BaseMoEExperts(
                num_experts=1,
                top_k=1,
                hidden_size=128,
                moe_intermediate_size=256,
                tp_size=2,
                tp_rank=1,
                ep_size=1,
                ep_rank=0,
                params_dtype=torch.bfloat16,
                model_config=SimpleNamespace(
                    num_layers=1, activation_type="SiGLU", quant_config=cfg
                ),
                parallelism_config=_load_config(
                    tp_size=2, tp_rank=1
                ).parallelism_config,
                moe_config=None,
                quant_config=QuantizationConfig("fp8_block", cfg),
                layer_idx=0,
            )

        reference, local = owner(), owner()
        hf = {}
        for i, proj in enumerate(("gate_proj", "up_proj", "down_proj")):
            shape = (128, 256) if proj == "down_proj" else (256, 128)
            name = "0." + proj
            hf[name + ".weight"] = (
                (torch.arange(128 * 256).remainder(7) + i)
                .reshape(shape)
                .to(torch.float8_e4m3fn)
            )
            hf[name + ".weight_scale_inv"] = (
                torch.tensor([1 + i, 6 + i])
                .float()
                .reshape(tuple(n // 128 for n in shape))
            )
        with tempfile.TemporaryDirectory() as root:
            path = str(Path(root) / "model.safetensors")
            save_file(hf, path)
            with safe_open(path, framework="pt", device="cpu") as handle:
                for key in handle.keys():
                    reference.load_weights({key: handle.get_tensor(key)})
                    (part,) = read_parts(local, key, key, handle.get_slice(key))
                    self.assertEqual(part[1].tensor.numel() * 2, hf[key].numel())
                    local.load_weights(dict([part]))
        for owner_ in (local, reference):
            owner_.validate_weights_loaded()
        for name in ("w13", "w2", "w13_scale", "w2_scale"):
            self.assertTrue(
                torch.equal(
                    getattr(reference, name).view(torch.uint8),
                    getattr(local, name).view(torch.uint8),
                )
            )

    def test_pure_tp_real_qwen_experts(self):
        for stacked in (False, True):
            cfg = _config(tie=False)
            cfg.moe_layer_index, cfg.moe_style, cfg.moe_inter_size = [0], 1, 8
            cfg.model_type = "qwen35_moe" if stacked else "qwen3_next"
            prefix = "model.language_model." if stacked else "model."
            weights = {
                k.replace("model.", prefix, 1): v
                for k, v in _dense_weights(True).items()
                if ".mlp." not in k
            }
            mlp = prefix + "layers.0.mlp."
            weights[mlp + "gate.weight"] = torch.ones(2, 4)
            gate = torch.arange(64).float().reshape(2, 8, 4)
            up = gate + 100
            down = torch.arange(64).float().reshape(2, 4, 8) + 200
            if stacked:
                weights[mlp + "experts.gate_up_proj"] = torch.cat((gate, up), dim=1)
                weights[mlp + "experts.down_proj"] = down
            else:
                for e in range(2):
                    for p, v in (("gate", gate), ("up", up), ("down", down)):
                        weights[mlp + f"experts.{e}.{p}_proj.weight"] = v[e].clone()
            with tempfile.TemporaryDirectory() as root:
                save_file(weights, str(Path(root) / "model.safetensors"))
                for rank in (0, 1):
                    for enabled in (False, True):
                        with self.subTest(stacked=stacked, rank=rank, enabled=enabled):
                            full, slices = [], []

                            @contextmanager
                            def track(*args, **kwargs):
                                with safe_open(*args, **kwargs) as handle:

                                    class File:
                                        keys = handle.keys

                                        def get_tensor(self, key):
                                            if ".experts." in key:
                                                full.append(key)
                                            return handle.get_tensor(key)

                                        def get_slice(self, key):
                                            source = handle.get_slice(key)
                                            if ".experts." not in key:
                                                return source

                                            class Source:
                                                get_shape = source.get_shape

                                                def __getitem__(self, index):
                                                    value = source[index]
                                                    slices.append((key, value.numel()))
                                                    return value

                                            return Source()

                                    yield File()

                            load = replace(
                                _load_config(tp_size=2, tp_rank=rank),
                                moe_pure_tp_preshard=enabled,
                            )
                            with mock.patch("safetensors.safe_open", track):
                                model = NewModelLoader(cfg, load, root).load()
                            experts = next(
                                m for m in model.modules() if hasattr(m, "moe_inter_tp")
                            )
                            rows = slice(4 * rank, 4 * (rank + 1))
                            torch.testing.assert_close(
                                experts.w13,
                                torch.cat((up[:, rows], gate[:, rows]), dim=1),
                            )
                            torch.testing.assert_close(experts.w2, down[:, :, rows])
                            self.assertEqual(
                                len(full), 0 if enabled else (2 if stacked else 6)
                            )
                            self.assertEqual(
                                [count for _, count in slices],
                                [16] * 6 if enabled else [],
                            )

    def test_page_cache_counts_consumed_slices_and_finishes_on_close(self):
        with tempfile.TemporaryDirectory() as root:
            path = str(Path(root) / "model.safetensors")
            save_file({name: torch.ones(8) for name in ("a", "b", "c")}, path)
            for enabled in (False, True):
                with self.subTest(enabled=enabled), mock.patch.object(
                    ckpt_file_info, "_DROP_PAGE_CACHE", enabled
                ), mock.patch.object(
                    ckpt_file_info, "_DROP_PAGE_CACHE_THRESHOLD", 24
                ), mock.patch.object(
                    ckpt_file_info.ShardPageCache, "_drop_page_cache"
                ) as drop:
                    weights = weight_mapper.get_all_weights(
                        [path],
                        read_spec_resolver=lambda name, shape: make_sharded_read_spec(
                            name, shape, 0, 2, 1
                        ),
                        advise_scratch_cache=True,
                    )
                    next(weights)
                    next(weights)  # the first 16-byte slice has finished
                    self.assertEqual(drop.call_count, 0)
                    next(weights)  # the first two have finished: above 24 bytes
                    self.assertEqual(drop.call_count, int(enabled))
                    weights.close()
                    self.assertEqual(drop.call_count, 2 * int(enabled))


if __name__ == "__main__":
    unittest.main()

import os
import tempfile
import unittest
from contextlib import contextmanager
from math import prod
from types import SimpleNamespace
from unittest.mock import patch

import torch
from safetensors.torch import save_file

from rtp_llm.config.quant_config import Fp8BlockWiseQuantConfig
from rtp_llm.model_loader import per_block_fp8_quant_weight as pbq
from rtp_llm.model_loader.ffn_weight import MoeConfig
from rtp_llm.model_loader.load_config import LoadConfig
from rtp_llm.model_loader.tensor_source import DatabaseTensorSource
from rtp_llm.models.qwen3_next import qwen3_next_weight as qwen
from rtp_llm.utils.database import CkptDatabase
from rtp_llm.utils.model_weight import W

# Identity device stubs: parity vs legacy relies on no real post-processing here.
_DEVICE = SimpleNamespace(
    shuffle_moe_weight=lambda tensor, *_: tensor,
    maybe_rewrite_weight_by_key=lambda _, tensor: tensor,
)


def _config(rank=0, **overrides):
    values = dict(tp_size=2, tp_rank=rank, ep_size=1, ep_rank=0, dp_size=1, dp_rank=0)
    values.update(ffn_tp_size=1, ffn_tp_rank=0, hidden_size=8, head_num=1)
    values.update(head_num_kv=1, size_per_head=8, moe_pure_tp_mode=True)
    values.update(moe_pure_tp_preshard=True)
    values.update(compute_dtype=torch.float32, exported_device=_DEVICE, **overrides)
    return LoadConfig.model_construct(**values)


def _weights(stacked):
    factory = (
        qwen.Qwen35MoeWeight._create_moe_expert_weights_stacked
        if stacked
        else qwen.Qwen3NextBaseWeight._create_moe_expert_weights
    )
    return factory(SimpleNamespace(prefix="model."), MoeConfig(expert_num=2))


def _name(weight, expert, stacked):
    if stacked:
        return weight.tensor_name(0)
    return weight.name.format(i=0, i_1=1, expert_id=expert)


def _tensors(weight, scale_divisor=1):
    stacked = weight.stacked_ckpt_keys
    kind = {W.moe_s1: W.moe_w1, W.moe_s2: W.moe_w2}.get(weight.name, weight.name)
    shape = {W.moe_w1: (6, 8), W.moe_w2: (8, 6)}[kind]
    if stacked:
        # A stacked w1 ckpt carries gate+up in one tensor, doubling its split dim.
        shape = (2, shape[0] * (2 if kind == W.moe_w1 else 1), shape[1])
    if scale_divisor > 1:
        # Shrunk scales stop dividing by tp_size, forcing the legacy full read.
        head = int(stacked)
        shape = (*shape[:head], *(x // scale_divisor for x in shape[head:]))
    dtype = weight.data_type or torch.float32
    tensors = {}
    for index, ckpt in enumerate(weight.weights):
        for expert in (0,) if stacked else range(2):
            value = index * 100 + expert * 200 + torch.arange(prod(shape))
            tensors[_name(ckpt, expert, stacked)] = value.reshape(shape).to(dtype)
    return tensors


@contextmanager
def _database(tensors, safetensors=True):
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "model.safetensors" if safetensors else "model.bin")
        (save_file if safetensors else torch.save)(tensors, path)
        database = CkptDatabase(tmp)
        try:
            yield database
        finally:
            for ckpt in database.pretrain_file_list:
                ckpt.close_safetensor_handle()


@contextmanager
def _track_reads(database):
    with patch.object(
        database, "load_tensor_slice", wraps=database.load_tensor_slice
    ) as sliced, patch.object(
        database, "load_tensor", wraps=database.load_tensor
    ) as full:
        yield sliced, full


def _legacy(weight, tensors):
    stacked = weight.stacked_ckpt_keys
    raw = [
        tensors[_name(ckpt, expert, stacked)]
        for ckpt in weight.weights
        for expert in range(2)
    ]
    experts = list(range(2)) * len(weight.weights)
    return weight.process_fun([t[e] for t, e in zip(raw, experts)] if stacked else raw)


class PureTpPreshardTest(unittest.TestCase):
    def _assert_parity(self, weight, tensors, rank=0, preshard=True, **overrides):
        config = _config(rank, **overrides)
        with _database(tensors) as database, _track_reads(database) as (sliced, full):
            actual = weight.load(DatabaseTensorSource(database), 0, "cpu", config)
            expected = weight._split({weight.name: _legacy(weight, tensors)}, config)
            torch.testing.assert_close(actual[weight.name].cpu(), expected[weight.name])
            self.assertEqual(sliced.called, preshard)
            self.assertEqual(full.called, not preshard)
            return sliced.call_args_list

    def test_qwen_layouts_match_legacy_on_both_ranks(self):
        for stacked, weight in [(s, w) for s in (False, True) for w in _weights(s)]:
            tensors = _tensors(weight)
            calls = []
            for rank in (0, 1):
                with self.subTest(weight.name, stacked=stacked, rank=rank):
                    calls.extend(self._assert_parity(weight, tensors, rank))
            # Last-dim slicing is strided in safetensors: must stay whole.
            self.assertTrue(all(c.args[1][-1] == slice(None) for c in calls))

    def test_unsafe_scopes_skip_sliced_reads(self):
        weight = _weights(False)[0]
        tensors = _tensors(weight)
        with _database(tensors) as db, _track_reads(db) as (sliced, _):
            source = DatabaseTensorSource(db)
            for key, value in (
                ("moe_pure_tp_preshard", False),
                ("moe_pure_tp_mode", False),
                ("merge_lora", True),
            ):
                with self.subTest(scope=key):
                    self.assertIsNone(
                        weight._load_pure_tp(source, 0, "cpu", _config(**{key: value}))
                    )
            self.assertIsNone(weight._load_pure_tp(source, None, "cpu", _config()))
            with patch.object(weight, "_get_split_func", return_value=None):
                self.assertIsNone(weight._load_pure_tp(source, 0, "cpu", _config()))
            self.assertFalse(sliced.called)
        with _database(tensors, safetensors=False) as db:
            self.assertIsNone(
                weight._load_pure_tp(DatabaseTensorSource(db), 0, "cpu", _config())
            )

    def test_switch_off_rolls_back_to_legacy_full_reads(self):
        # The documented rollback lever must be flippable in-process.
        for weight in _weights(False):
            with self.subTest(weight.name):
                self._assert_parity(
                    weight,
                    _tensors(weight),
                    rank=1,
                    preshard=False,
                    moe_pure_tp_preshard=False,
                )

    def test_per_block_weights_and_scales_preshard_or_fall_back(self):
        for source in _weights(False):
            self.assertTrue(source.enable_pure_tp_preshard)
            offline = pbq.PerBlockFp8Weight(
                source, Fp8BlockWiseQuantConfig(is_quanted=True), name=source.name
            )
            online = pbq.LoadQuantPerBlockFp8Weight(
                source, Fp8BlockWiseQuantConfig(is_quanted=False), name=source.name
            )
            # Online quant clones must not inherit the preshard opt-in flag.
            self.assertTrue(offline.kernel.enable_pure_tp_preshard)
            self.assertFalse(online.kernel.enable_pure_tp_preshard)
            self.assertFalse(online.scale.enable_pure_tp_preshard)
            with self.subTest(source.name, divisible=True):
                tensors = _tensors(offline.kernel)
                self._assert_parity(offline.kernel, tensors, rank=1)
                self._assert_parity(offline.scale, _tensors(offline.scale), rank=1)
            with self.subTest(source.name, divisible=False):
                self._assert_parity(
                    offline.scale,
                    _tensors(offline.scale, scale_divisor=2),
                    rank=1,
                    preshard=False,
                )


if __name__ == "__main__":
    unittest.main()

import os
import tempfile
import unittest
from math import prod
from types import SimpleNamespace
from unittest.mock import patch

import torch
from safetensors.torch import save_file

from rtp_llm.config.quant_config import Fp8BlockWiseQuantConfig
from rtp_llm.model_loader import per_block_fp8_quant_weight as pbq
from rtp_llm.model_loader.ffn_weight import MoeConfig, PreShardedTensor
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


class _Database(CkptDatabase):
    """Fake for gating cases only; parity cases use a real CkptDatabase."""

    def __init__(self, tensors):
        self.tensors, self.safetensor = tensors, True
        self.slices, self.full_reads = [], []

    @property
    def is_safetensor(self):
        return self.safetensor

    def has_tensor(self, name):
        return name in self.tensors

    def get_tensor_shape(self, name):
        return self.tensors[name].shape

    def load_tensor_slice(self, name, tensor_slice, data_type):
        self.slices.append((name, tensor_slice))
        return self.tensors[name][tensor_slice].to(data_type)

    def load_tensor(self, name, data_type=torch.float16):
        self.full_reads.append(name)
        return [self.tensors[name].to(data_type)]


def _config(rank=0, **overrides):
    # model_construct keeps real defaults/methods but skips unread required fields.
    values = dict(tp_size=2, tp_rank=rank, ep_size=1, ep_rank=0, dp_size=1, dp_rank=0)
    values.update(ffn_tp_size=1, ffn_tp_rank=0, hidden_size=8, head_num=1)
    values.update(head_num_kv=1, size_per_head=8, moe_pure_tp_mode=True)
    # Fail-closed on LoadConfig, so parity cases must opt in explicitly.
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


def _database(weight, scale_divisor=1):
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
            value = torch.arange(prod(shape), dtype=torch.float32).reshape(shape)
            tensors[_name(ckpt, expert, stacked)] = (value + index * 100).to(dtype)
    return _Database(tensors)


def _legacy(weight, database):
    stacked = weight.stacked_ckpt_keys
    raw = [
        database.tensors[_name(ckpt, expert, stacked)]
        for ckpt in weight.weights
        for expert in range(2)
    ]
    experts = list(range(2)) * len(weight.weights)
    return weight.process_fun([t[e] for t, e in zip(raw, experts)] if stacked else raw)


class PureTpPreshardTest(unittest.TestCase):
    def _assert_parity(self, weight, database, rank=0, reference=None, **overrides):
        config = _config(rank, **overrides)
        actual = weight.load(DatabaseTensorSource(database), 0, "cpu", config)
        expected = weight._split(
            {weight.name: _legacy(weight, reference or database)}, config
        )
        torch.testing.assert_close(actual[weight.name].cpu(), expected[weight.name])

    def test_qwen_layouts_match_legacy_on_both_ranks(self):
        for stacked, weight in [(s, w) for s in (False, True) for w in _weights(s)]:
            reference = _database(weight)
            with tempfile.TemporaryDirectory() as tmp:
                save_file(reference.tensors, os.path.join(tmp, "m.safetensors"))
                db = CkptDatabase(tmp, recycle_handles=True)
                with patch.object(
                    db, "load_tensor_slice", wraps=db.load_tensor_slice
                ) as sliced, patch.object(
                    db, "load_tensor", wraps=db.load_tensor
                ) as full:
                    for rank in (0, 1):
                        with self.subTest(weight.name, stacked=stacked, rank=rank):
                            self._assert_parity(weight, db, rank, reference)
                    self.assertTrue(sliced.called)
                    self.assertFalse(full.called)
                    # Last-dim slicing is strided in safetensors: must stay whole.
                    last = [c.args[1][-1] for c in sliced.call_args_list]
                    self.assertEqual(last, [slice(None)] * len(last))
                db.pretrain_file_list[0].close_safetensor_handle()

    def test_fast_path_marks_output_so_split_is_skipped(self):
        # The fast path already returns rank-local data, so _split has to pass it
        # through: splitting twice would silently drop 1/tp_size of every expert.
        # Exclusivity with _load_raw_tensor_gpu_preallocate is not assertable on
        # CPU -- that path is gated on a cuda target device.
        weight = _weights(False)[0]
        config = _config(rank=1)
        reference = _database(weight)
        with tempfile.TemporaryDirectory() as tmp:
            save_file(reference.tensors, os.path.join(tmp, "m.safetensors"))
            db = CkptDatabase(tmp, recycle_handles=True)
            raw = weight._load_raw_tensor(DatabaseTensorSource(db), 0, "cpu", config)
            marker = raw[weight.name]
            self.assertIsInstance(marker, PreShardedTensor)
            self.assertIs(weight._split(raw, config)[weight.name], marker.tensor)
            db.pretrain_file_list[0].close_safetensor_handle()

    def test_unsafe_scopes_skip_sliced_reads(self):
        weight = _weights(False)[0]
        db = _database(weight)
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
        db.safetensor = False
        self.assertIsNone(weight._load_pure_tp(source, 0, "cpu", _config()))
        self.assertFalse(db.slices)

    def test_switch_off_rolls_back_to_legacy_full_reads(self):
        # Untestable while the gate was an import-time env constant: operators had
        # a documented rollback lever that no test could flip in-process.
        for weight in _weights(False):
            with self.subTest(weight.name):
                db = _database(weight)
                self._assert_parity(weight, db, rank=1, moe_pure_tp_preshard=False)
                self.assertFalse(db.slices)
                self.assertTrue(db.full_reads)

    def test_per_block_weights_and_scales_preshard_or_fall_back(self):
        for source in _weights(False):
            self.assertTrue(source.enable_pure_tp_preshard)
            offline = pbq.PerBlockFp8Weight(
                source, Fp8BlockWiseQuantConfig(is_quanted=True), name=source.name
            )
            online = pbq.LoadQuantPerBlockFp8Weight(
                source, Fp8BlockWiseQuantConfig(is_quanted=False), name=source.name
            )
            # Offline per-block quant opts in explicitly; online quant clones must
            # not inherit the flag, or pre-sharded tensors would reach online quant.
            self.assertTrue(offline.kernel.enable_pure_tp_preshard)
            self.assertFalse(online.kernel.enable_pure_tp_preshard)
            self.assertFalse(online.scale.enable_pure_tp_preshard)
            with self.subTest(source.name, divisible=True):
                # fp8 kernel parity runs the real safetensors boundary, not the fake.
                reference = _database(offline.kernel)
                with tempfile.TemporaryDirectory() as tmp:
                    save_file(reference.tensors, os.path.join(tmp, "m.safetensors"))
                    real = CkptDatabase(tmp, recycle_handles=True)
                    self._assert_parity(offline.kernel, real, 1, reference)
                    real.pretrain_file_list[0].close_safetensor_handle()
                db = _database(offline.scale)
                self._assert_parity(offline.scale, db, rank=1)
                self.assertTrue(db.slices)
                self.assertFalse(db.full_reads)
            with self.subTest(source.name, divisible=False):
                db = _database(offline.scale, scale_divisor=2)
                self._assert_parity(offline.scale, db, rank=1)
                self.assertFalse(db.slices)
                self.assertTrue(db.full_reads)


if __name__ == "__main__":
    unittest.main()

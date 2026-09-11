import json
import os
import tempfile
import unittest
from contextlib import contextmanager
from math import prod
from types import SimpleNamespace
from unittest.mock import mock_open, patch

import torch
from safetensors.torch import save_file

from rtp_llm.config.quant_config import (
    Fp8BlockWiseQuantConfig,
    ModelOptFp4Config,
    MXFp4QuarkQuantConfig,
)
from rtp_llm.model_loader import per_block_fp8_quant_weight as pbq
from rtp_llm.model_loader.ffn_weight import MoeConfig
from rtp_llm.model_loader.load_config import LoadConfig
from rtp_llm.model_loader.mixed_fp4_quant_weight import MixedFp4Weight
from rtp_llm.model_loader.per_group_fp4_quant_weight import PerGroupFp4Weight
from rtp_llm.model_loader.tensor_source import DatabaseTensorSource
from rtp_llm.models.qwen3_next import qwen3_next_weight as qwen
from rtp_llm.utils import model_weight as mw
from rtp_llm.utils.database import CkptDatabase
from rtp_llm.utils.model_weight import W

# Identity device stubs: parity vs legacy relies on no real post-processing here.
_DEVICE = SimpleNamespace(
    shuffle_moe_weight=lambda tensor, *_: tensor,
    maybe_rewrite_weight_by_key=lambda _, tensor, **kwargs: tensor,
    maybe_prepare_static_weights_for_fp4_moe=lambda _w, _s, kernel, scale: (
        kernel,
        scale,
    ),
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
    with (
        patch.object(
            database, "load_tensor_slice", wraps=database.load_tensor_slice
        ) as sliced,
        patch.object(database, "load_tensor", wraps=database.load_tensor) as full,
    ):
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


class LoadConfigExpertMappingTest(unittest.TestCase):
    def test_zero_redundancy_uses_identity_mapping(self):
        for ep_size, num_nodes in ((1, 4), (1, 1), (16, 4)):
            with self.subTest(ep_size=ep_size, num_nodes=num_nodes):
                mapping = LoadConfig.create_redundant_expert(
                    layer_num=3,
                    expert_num=512,
                    phy_exp_num=512,
                    ep_size=ep_size,
                    num_nodes=num_nodes,
                )
                self.assertEqual(mapping, [list(range(512)) for _ in range(3)])

    def test_zero_redundancy_layer_lists_are_independent(self):
        for ep_size, num_nodes in ((1, 4), (1, 1), (16, 4)):
            with self.subTest(ep_size=ep_size, num_nodes=num_nodes):
                mapping = LoadConfig.create_redundant_expert(
                    3, 512, 512, ep_size, num_nodes
                )
                self.assertEqual(len({id(layer) for layer in mapping}), 3)
                mapping[0][0] = 511
                self.assertEqual(mapping[1:], [list(range(512)), list(range(512))])

    def test_explicit_mapping_takes_priority(self):
        for ep_size, phy_exp_num in ((1, 32), (16, 32), (16, 48)):
            with self.subTest(ep_size=ep_size, phy_exp_num=phy_exp_num):
                expected = [
                    [(i + layer) % 32 for i in reversed(range(phy_exp_num))]
                    for layer in range(2)
                ]
                with patch(
                    "builtins.open", mock_open(read_data=json.dumps(expected))
                ) as opened:
                    mapping = LoadConfig.create_redundant_expert(
                        2, 32, phy_exp_num, ep_size, 4, phy2log_path="phy2log.json"
                    )
                opened.assert_called_once_with("phy2log.json", "r")
                self.assertEqual(mapping, expected)

    def test_ep16_redundancy_preserves_node_local_layout(self):
        mapping = LoadConfig.create_redundant_expert(2, 32, 48, 16, 4)
        # Each rank keeps two experts and copies one from the next rank in its node.
        node_layout = [0, 1, 2, 2, 3, 4, 4, 5, 6, 6, 7, 0]
        expected = [expert + node * 8 for node in range(4) for expert in node_layout]
        self.assertEqual(mapping, [expected, expected])

    def test_ep16_redundancy_preserves_append_layout(self):
        mapping = LoadConfig.create_redundant_expert(2, 31, 32, 16, 4)
        expected = list(range(31)) + [0]
        self.assertEqual(mapping, [expected, expected])


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


class Fp4PureTpPreshardTest(unittest.TestCase):
    wrappers = (PerGroupFp4Weight, MixedFp4Weight)

    def _wrap(self, wrapper, source, quark=False):
        quant = (
            MXFp4QuarkQuantConfig(is_quanted=True, group_size=32)
            if quark
            else ModelOptFp4Config(
                bits=4,
                group_size=16,
                is_quanted=True,
                mixed_attention=wrapper is MixedFp4Weight,
            )
        )
        return wrapper(source, quant, name=source.name)

    def _tensors(self, weight, intermediate=256):
        # ModelOpt packs two FP4 nibbles per byte; group scales cover 16 values.
        rows, cols = (
            (intermediate, 32) if weight.name == W.moe_w1 else (32, intermediate)
        )
        tensors = {}
        for child in weight.sub_weights.values():
            shape = (
                (rows, cols // (2 if child is weight.kernel else 16))
                if child in (weight.kernel, weight.scale)
                else (1,)
            )
            for index, ckpt in enumerate(child.weights):
                for expert in range(2):
                    offset = 71 * index + 37 * expert
                    if len(shape) == 2:
                        row = torch.arange(shape[0]).unsqueeze(1)
                        col = torch.arange(shape[1]).unsqueeze(0)
                        values = row * 7 + row // 16 * 13 + col * 11 + offset
                        values = (
                            values % 256
                            if child is weight.kernel
                            else (values % 24 + 1).float() / 8
                        )
                    else:
                        values = torch.tensor([1.0 + offset])
                    tensors[_name(ckpt, expert, False)] = values.to(child.data_type)
        return tensors

    def _assert_bytes(self, actual, expected):
        self.assertEqual(actual.shape, expected.shape)
        self.assertEqual(actual.dtype, expected.dtype)
        # Compare representations, not approximate float8 values (or NaN equality).
        self.assertTrue(
            torch.equal(
                actual.detach().cpu().contiguous().view(torch.uint8),
                expected.detach().cpu().contiguous().view(torch.uint8),
            )
        )

    def _assert_composite_parity(self, device):
        stack = mw.stack_0

        def scalar_stack_only(ts):
            self.assertTrue(all(t.ndim <= 1 for t in ts), "full expert stack")
            return stack(ts)

        for wrapper in self.wrappers:
            for source in _weights(False):
                weight = self._wrap(wrapper, source)
                tensors = self._tensors(weight)
                scalars = (weight.scale_2, weight.input_scale)
                scalar_names = [
                    _name(ckpt, expert, False)
                    for child in scalars
                    for ckpt in child.weights
                    for expert in range(2)
                ]
                for child in (weight.kernel, weight.scale):
                    self.assertTrue(child.enable_pure_tp_preshard)
                for child in scalars:
                    self.assertFalse(child.enable_pure_tp_preshard)
                for ckpt in weight.scale.weights:
                    for expert in range(2):
                        self.assertTrue(
                            torch.isfinite(
                                tensors[_name(ckpt, expert, False)].float()
                            ).all()
                        )

                scalar_reference = None
                with _database(tensors) as db:
                    tensor_source = DatabaseTensorSource(db)
                    for tp in (1, 2, 16):
                        for rank in range(tp):
                            with self.subTest(
                                wrapper=wrapper.__name__,
                                weight=source.name,
                                device=device,
                                tp=tp,
                                rank=rank,
                            ):
                                config = _config(rank, tp_size=tp, hidden_size=32)
                                legacy_config = config.model_copy(
                                    update={"moe_pure_tp_preshard": False}
                                )
                                with _track_reads(db) as (sliced, full):
                                    expected = weight.load(
                                        tensor_source, 0, device, legacy_config
                                    )
                                if scalar_reference is None:
                                    scalar_reference = {
                                        child.name: expected[child.name]
                                        for child in scalars
                                    }
                                sliced.assert_not_called()
                                self.assertCountEqual(
                                    [c.args[0] for c in full.call_args_list], tensors
                                )
                                with (
                                    _track_reads(db) as (sliced, full),
                                    patch.object(
                                        mw, "stack_0", side_effect=scalar_stack_only
                                    ),
                                ):
                                    actual = weight.load(
                                        tensor_source, 0, device, config
                                    )
                                self.assertEqual(set(actual), set(weight.sub_weights))
                                self.assertCountEqual(
                                    [c.args[0] for c in full.call_args_list],
                                    scalar_names,
                                )
                                expected_reads = []
                                for child in (weight.kernel, weight.scale):
                                    for ckpt in child.weights:
                                        for expert in range(2):
                                            name = _name(ckpt, expert, False)
                                            # W2 reads a whole expert: last-axis safetensors
                                            # slicing is strided; only the output is rank-local.
                                            where = (slice(None), slice(None))
                                            if source.name == W.moe_w1:
                                                size = tensors[name].shape[0] // tp
                                                where = (
                                                    slice(
                                                        rank * size, (rank + 1) * size
                                                    ),
                                                    slice(None),
                                                )
                                            expected_reads.append(
                                                (name, where, child.data_type)
                                            )
                                    value = actual[child.name]
                                    self.assertIsNone(value._base)
                                    self.assertEqual(value.storage_offset(), 0)
                                    self.assertEqual(
                                        value.untyped_storage().nbytes(),
                                        value.numel() * value.element_size(),
                                    )
                                    self.assertEqual(
                                        value.numel(),
                                        sum(
                                            tensors[_name(ckpt, expert, False)].numel()
                                            for ckpt in child.weights
                                            for expert in range(2)
                                        )
                                        // tp,
                                    )
                                self.assertCountEqual(
                                    [c.args for c in sliced.call_args_list],
                                    expected_reads,
                                )
                                for name in actual:
                                    self.assertEqual(actual[name].device.type, device)
                                    self._assert_bytes(actual[name], expected[name])
                                for child in scalars:
                                    self._assert_bytes(
                                        actual[child.name], scalar_reference[child.name]
                                    )
                # Outputs must own their data after the safetensors handles close.
                for name in actual:
                    self._assert_bytes(actual[name], expected[name])

    def test_composite_cpu_parity_and_rank_local_storage(self):
        self._assert_composite_parity("cpu")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_composite_cuda_parity_with_identity_device(self):
        self._assert_composite_parity("cuda")

    def test_nondivisible_group_scales_fall_back_before_reading(self):
        for wrapper in self.wrappers:
            for source in _weights(False):
                weight = self._wrap(wrapper, source)
                # W2: packed K=144 divides TP16, but K/16=18 scales do not.
                intermediate = 264 if source.name == W.moe_w1 else 288
                tensors = self._tensors(weight, intermediate)
                child = weight.scale
                with _database(tensors) as db:
                    for rank in (0, 7, 15):
                        with self.subTest(
                            wrapper=wrapper.__name__, weight=source.name, rank=rank
                        ):
                            config = _config(rank, tp_size=16, hidden_size=32)
                            expected = child._split(
                                {child.name: _legacy(child, tensors)}, config
                            )
                            with (
                                _track_reads(db) as (sliced, full),
                                self.assertLogs(level="WARNING") as logs,
                            ):
                                actual = child.load(
                                    DatabaseTensorSource(db), 0, "cpu", config
                                )
                            self.assertTrue(
                                any("not divisible" in line for line in logs.output)
                            )
                            sliced.assert_not_called()
                            self.assertEqual(full.call_count, len(child.weights) * 2)
                            self._assert_bytes(actual[child.name], expected[child.name])
                            if source.name == W.moe_w2:
                                with _track_reads(db) as (sliced, full):
                                    weight.kernel.load(
                                        DatabaseTensorSource(db), 0, "cpu", config
                                    )
                                self.assertEqual(sliced.call_count, 2)
                                full.assert_not_called()

    def test_source_optout_stacked_and_quark_do_not_enable_children(self):
        for wrapper in self.wrappers:
            for exclusion in ("optout", "stacked", "quark"):
                if exclusion == "quark" and wrapper is not MixedFp4Weight:
                    continue
                for source in _weights(exclusion == "stacked"):
                    with self.subTest(
                        wrapper=wrapper.__name__,
                        weight=source.name,
                        exclusion=exclusion,
                    ):
                        if exclusion == "optout":
                            source.enable_pure_tp_preshard = False
                        weight = self._wrap(wrapper, source, quark=exclusion == "quark")
                        if exclusion == "quark":
                            self.assertIsNone(weight.scale_2)
                            self.assertIsNone(weight.input_scale)
                            self.assertEqual(weight.scale.data_type, torch.uint8)
                        with _database({}) as db, _track_reads(db) as (sliced, full):
                            for child in weight.sub_weights.values():
                                self.assertFalse(child.enable_pure_tp_preshard)
                                self.assertIsNone(
                                    child._load_pure_tp(
                                        DatabaseTensorSource(db), 0, "cpu", _config()
                                    )
                                )
                            sliced.assert_not_called()
                            full.assert_not_called()


if __name__ == "__main__":
    unittest.main()

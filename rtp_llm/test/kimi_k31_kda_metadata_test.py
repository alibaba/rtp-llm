"""cuLA page mappings, host lifetime and nonblocking CUDA preparation."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from rtp_llm.models_py.model_desc.kimi_k3 import KimiK3Model
from rtp_llm.models_py.modules.factory.attention.cuda_kda_impl import (
    prepare_cula_kda_prefill_metadata,
)
from rtp_llm.models_py.modules.hybrid.query_layout import (
    AttentionInputView,
    QueryLayout,
)
from rtp_llm.ops import HybridAttentionType


class KdaMetadataTest(unittest.TestCase):
    def prepare(self, device):
        # Two real requests surround a virtual row. Repeated checkpoint slot 2
        # must retain only its last publication; the second request reads slot 4.
        states = torch.zeros((8, 2, 128, 128), dtype=torch.float32, device=device)
        return prepare_cula_kda_prefill_metadata(
            [0, 130, 194, 259],
            [0, 0, 64],
            [[1, 2, 2], [0], [4, 5, 6]],
            64,
            states,
        )

    def check_mapping(self, metadata):
        packed = metadata.packed
        self.assertIsNotNone(packed)
        self.assertEqual(packed.initial_blocks.cpu().tolist(), [-1, -1, 4])
        self.assertEqual(packed.checkpoint_blocks.cpu().tolist(), [1, -1, 2, -1, 5, 6])
        self.assertEqual(packed.cu_seqlens_cpu.tolist(), [0, 130, 194, 259])
        self.assertEqual(packed.checkpoint_offsets.cpu().tolist(), [0, 3, 4, 6])
        self.assertEqual(packed.padding_ranges, ((130, 194),))
        self.assertEqual(packed.initial_states.dtype, torch.float32)
        self.assertEqual(packed.checkpoints.shape, (1, 6, 2, 128, 128))
        self.assertEqual(len(packed.host_tensors), 4)
        self.assertTrue(all(t.device.type == "cpu" for t in packed.host_tensors))

    def test_exact_page_mapping_and_independent_round_lifetimes(self):
        first, second = self.prepare("cpu"), self.prepare("cpu")
        self.check_mapping(first)
        self.check_mapping(second)
        for a, b in zip(first.packed.host_tensors, second.packed.host_tensors):
            self.assertNotEqual(a.data_ptr(), b.data_ptr())

    def test_cuda_metadata_has_no_blocking_copy(self):
        if not torch.cuda.is_available():
            self.skipTest("requires CUDA")
        with torch.device("cuda"):
            warm = self.prepare("cuda")
        torch.cuda.synchronize()
        del warm
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ]
        ) as profile:
            with torch.profiler.record_function("k31.metadata.prepare"):
                with torch.device("cuda"):
                    actual = self.prepare("cuda")
        self.check_mapping(actual)
        self.assertTrue(all(t.is_pinned() for t in actual.packed.host_tensors))
        for event in profile.events():
            if event.name != "cudaStreamSynchronize":
                continue
            parent = event.cpu_parent
            while parent is not None:
                self.assertNotEqual(parent.name, "k31.metadata.prepare")
                parent = parent.cpu_parent


class _SimulatedDeviceMetadata:
    """CPU backing only: count boundary copies, not CUDA execution or ordering."""

    device = torch.device("cuda:0")

    def __init__(self, backing):
        self.backing = backing
        self.copies = 0

    def __getattr__(self, name):
        return getattr(self.backing, name)

    def cpu(self):
        self.copies += 1
        return self.backing.clone()


class KdaHostPreparationTest(unittest.TestCase):
    def fixture(self, *, device_input, cp=False, prefix=0):
        model = object.__new__(KimiK3Model)
        nn.Module.__init__(model)
        model.prefill_query_cp = cp
        model.dsa_workspace = object()
        tags = ("linear_a", "linear_a", "linear_b", "full")
        model.layers = nn.ModuleList([nn.Identity() for _ in tags])
        for layer, tag in zip(model.layers, tags):
            layer.layer_type = (
                HybridAttentionType.NONE
                if tag == "full"
                else HybridAttentionType.LINEAR
            )
        model.kv_cache = SimpleNamespace(
            get_layer_cache_groups=lambda i: [SimpleNamespace(tag=tags[i])]
        )
        lengths = torch.tensor([2, 2] if cp else [3, 2, 3], dtype=torch.int32)
        prefixes = torch.tensor(
            [prefix, 0] if cp else [prefix, 0, 0], dtype=torch.int32
        )
        info = None
        if cp:
            # Two ragged requests padded to four rows each, split across TP2.
            rankmajor = torch.tensor([0, 3, 4, 7, 1, 2, 5, 6])
            restore = torch.empty(8, dtype=torch.int64)
            restore[rankmajor] = torch.arange(8)
            info = SimpleNamespace(
                prefill_actual_input_lengths_cpu=torch.tensor(
                    [3, 2], dtype=torch.int32
                ),
                prefill_prefix_lengths_cpu=prefixes.clone(),
                prefill_qkv_restore_indice=restore,
                prefill_qkv_padding_mask=torch.tensor(
                    [True, True, True, False, True, True, False, False]
                ),
            )
        # All three table slices share storage but have different execution IDs.
        tables = torch.tensor(
            [
                [[21, 22], [23, 24], [0, 0]],
                [[1, 2], [3, 4], [0, 0]],
                [[7, 8], [9, 10], [0, 0]],
            ],
            dtype=torch.int32,
        )
        sources = {"lengths": lengths, "prefixes": prefixes}
        sources.update(
            (tag, tables[i, : lengths.numel()])
            for i, tag in enumerate(("full", "linear_a", "linear_b"))
        )
        if device_input:
            sources = {
                key: _SimulatedDeviceMetadata(value) for key, value in sources.items()
            }
        device_cu = torch.cat(
            (torch.zeros(1, dtype=torch.int32), lengths.cumsum(0, dtype=torch.int32))
        )
        cu = torch.empty(0, dtype=torch.int32) if device_input else device_cu.clone()
        groups = {}
        for tag in ("full", "linear_b", "linear_a"):
            physical = torch.full((lengths.numel(), 2), 101, dtype=torch.int32)
            groups[tag] = AttentionInputView(
                SimpleNamespace(
                    is_prefill=True,
                    is_target_verify=False,
                    is_mtp_draft_update=False,
                    input_lengths=sources["lengths"],
                    input_lengths_device=sources["lengths"],
                    prefix_lengths=sources["prefixes"],
                    prefix_lengths_device=sources["prefixes"],
                    cu_seqlens=cu,
                    cu_seqlens_device=device_cu,
                    kv_cache_kernel_block_id=sources[tag],
                    kv_cache_kernel_block_id_device=sources[tag],
                    kv_cache_block_id=physical,
                    kv_cache_block_id_device=physical.clone(),
                    cache_store_inputs=SimpleNamespace(
                        host_kv_cache_offset=physical[:2]
                    ),
                    logical_token_count=5,
                    physical_token_count=4 if cp else 8,
                    logical_request_count=2,
                    physical_request_count=lengths.numel(),
                    valid_token_mask=(
                        None if cp else torch.tensor([True] * 5 + [False] * 3)
                    ),
                    context_parallel_info=info,
                )
            )
        return model, SimpleNamespace(attention_inputs=groups), sources

    def prepare(self, model, inputs):
        # Exercise the production prepare entry, stopping before DSA kernels.
        with patch.object(model, "_prepare_dsa", return_value=object()) as dsa:
            result = model.prepare_fmha_impl(inputs)
        dsa.assert_called_once_with(inputs, False)
        self.assertIs(result, dsa.return_value)

    def test_host_contract_for_cp_and_padded_non_cp(self):
        from rtp_llm.models_py.modules.factory.attention.kda_impl import KdaPrefill

        for device_input in (False, True):
            for cp in (False, True):
                for prefix in (0, 64):
                    with self.subTest(device_input=device_input, cp=cp, prefix=prefix):
                        model, inputs, sources = self.fixture(
                            device_input=device_input, cp=cp, prefix=prefix
                        )
                        # Even the CPU-input route must not call .cpu() on host tensors.
                        with patch.object(
                            torch.Tensor, "cpu", side_effect=AssertionError("host D2H")
                        ):
                            self.prepare(model, inputs)
                        groups = inputs.attention_inputs
                        primary = groups["full"]
                        layout = (
                            QueryLayout(primary.context_parallel_info, 2, 0, "cpu")
                            if cp
                            else None
                        )
                        for tag, group in groups.items():
                            original = group.original
                            for field in (
                                "input_lengths_device",
                                "prefix_lengths_device",
                                "cu_seqlens_device",
                                "kv_cache_kernel_block_id_device",
                                "kv_cache_block_id_device",
                                "kv_cache_block_id",
                                "cache_store_inputs",
                                "valid_token_mask",
                            ):
                                self.assertIs(
                                    getattr(group, field), getattr(original, field)
                                )
                            for field in (
                                "logical_token_count",
                                "physical_token_count",
                                "logical_request_count",
                                "physical_request_count",
                            ):
                                self.assertEqual(
                                    getattr(group, field), getattr(original, field)
                                )
                            if cp:
                                self.assertIs(
                                    group.input_lengths, original.input_lengths
                                )
                                self.assertIs(group.cu_seqlens, original.cu_seqlens)
                            else:
                                self.assertEqual(
                                    group.input_lengths.tolist(), [3, 2, 3]
                                )
                                self.assertEqual(group.input_lengths.dtype, torch.int32)
                                self.assertEqual(
                                    group.cu_seqlens.tolist(), [0, 3, 5, 8]
                                )
                                torch.testing.assert_close(
                                    group.cu_seqlens, group.cu_seqlens_device
                                )
                                self.assertEqual(group.cu_seqlens.dtype, torch.int32)
                                self.assertIs(group.cu_seqlens, primary.cu_seqlens)
                            if tag == "full":
                                self.assertIs(
                                    group.kv_cache_kernel_block_id, sources[tag]
                                )
                                continue
                            temporal = layout.natural_attention(group) if cp else group
                            self.assertEqual(
                                temporal.prefix_lengths.tolist(),
                                [prefix, 0] if cp else [prefix, 0, 0],
                            )
                            self.assertEqual(temporal.prefix_lengths.dtype, torch.int32)
                            self.assertEqual(
                                temporal.kv_cache_kernel_block_id.dtype, torch.int32
                            )
                            expected = (
                                sources[tag].backing if device_input else sources[tag]
                            )
                            torch.testing.assert_close(
                                temporal.kv_cache_kernel_block_id, expected
                            )
                            KdaPrefill._validate_prefix_state(temporal, 64)
                            if cp:
                                self.assertEqual(
                                    temporal.cu_seqlens.tolist(), [0, 3, 5]
                                )
                                self.assertEqual(
                                    temporal.input_lengths.tolist(), [3, 2]
                                )
                                self.assertIs(layout.natural_attention(group), temporal)
                        if device_input:
                            self.assertEqual(sources["lengths"].copies, 0 if cp else 1)
                            self.assertEqual(sources["prefixes"].copies, 0 if cp else 1)
                            self.assertEqual(sources["full"].copies, 0)
                            self.assertEqual(sources["linear_a"].copies, 1)
                            self.assertEqual(sources["linear_b"].copies, 1)
                            # Re-entering prepare with already-host mirrors is idempotent.
                            before = {
                                key: value.copies for key, value in sources.items()
                            }
                            self.prepare(model, inputs)
                            self.assertEqual(
                                before,
                                {key: value.copies for key, value in sources.items()},
                            )

    def test_same_device_view_is_copied_once_across_distinct_tags(self):
        model, inputs, sources = self.fixture(device_input=True)
        alias = _SimulatedDeviceMetadata(
            sources["linear_a"].backing.view_as(sources["linear_a"].backing)
        )
        shared = inputs.attention_inputs["linear_b"].original
        shared.kv_cache_kernel_block_id = alias
        shared.kv_cache_kernel_block_id_device = alias
        self.prepare(model, inputs)
        self.assertEqual(sources["linear_a"].copies + alias.copies, 1)
        self.assertIs(
            inputs.attention_inputs["linear_a"].kv_cache_kernel_block_id,
            inputs.attention_inputs["linear_b"].kv_cache_kernel_block_id,
        )

    def test_next_round_refreshes_metadata_even_when_device_storage_is_reused(self):
        model, inputs, sources = self.fixture(device_input=True)
        self.prepare(model, inputs)
        old = inputs.attention_inputs["linear_a"]
        sources["lengths"].backing.copy_(torch.tensor([2, 3, 3], dtype=torch.int32))
        sources["prefixes"].backing[0] = 64
        sources["linear_a"].backing[0, 0] = 11
        # Simulate the next C++ preparation refreshing its device cu-seqlens too.
        for group in inputs.attention_inputs.values():
            group.original.cu_seqlens_device = torch.tensor(
                [0, 2, 5, 8], dtype=torch.int32
            )
        next_inputs = SimpleNamespace(
            attention_inputs={
                tag: AttentionInputView(group.original)
                for tag, group in inputs.attention_inputs.items()
            }
        )
        self.prepare(model, next_inputs)
        new = next_inputs.attention_inputs["linear_a"]
        self.assertEqual(old.prefix_lengths.tolist(), [0, 0, 0])
        self.assertEqual(old.kv_cache_kernel_block_id[0, 0].item(), 1)
        self.assertEqual(new.prefix_lengths.tolist(), [64, 0, 0])
        self.assertEqual(new.kv_cache_kernel_block_id[0, 0].item(), 11)
        self.assertEqual(new.cu_seqlens.tolist(), [0, 2, 5, 8])
        torch.testing.assert_close(new.cu_seqlens, new.cu_seqlens_device)
        self.assertEqual(sources["prefixes"].copies, 2)
        self.assertEqual(sources["linear_a"].copies, 2)

    def test_decode_verify_draft_update_and_mla_only_keep_device_fast_path(self):
        for excluded in ("decode", "verify", "draft_update", "mla_only"):
            with self.subTest(excluded=excluded):
                model, inputs, sources = self.fixture(device_input=True)
                for group in inputs.attention_inputs.values():
                    group.is_prefill = excluded != "decode"
                    group.is_target_verify = excluded == "verify"
                    group.is_mtp_draft_update = excluded == "draft_update"
                if excluded == "mla_only":
                    for layer in model.layers:
                        layer.layer_type = HybridAttentionType.NONE
                self.prepare(model, inputs)
                self.assertTrue(all(value.copies == 0 for value in sources.values()))
                for group in inputs.attention_inputs.values():
                    self.assertIs(group.input_lengths, group.original.input_lengths)
                    self.assertIs(group.prefix_lengths, group.original.prefix_lengths)
                    self.assertIs(
                        group.kv_cache_kernel_block_id,
                        group.original.kv_cache_kernel_block_id,
                    )
                    self.assertIs(group.cu_seqlens, group.original.cu_seqlens)

    def test_bound_single_group_keeps_readonly_device_fields(self):
        from rtp_llm.ops.compute_ops import PyAttentionInputs

        model, _, _ = self.fixture(device_input=False)
        model.kv_cache = None
        attention = PyAttentionInputs()
        attention.is_prefill = True
        attention.input_lengths = torch.tensor([3, 2, 3], dtype=torch.int32)
        attention.prefix_lengths = torch.zeros(3, dtype=torch.int32)
        attention.cu_seqlens = torch.empty(0, dtype=torch.int32)
        attention.kv_cache_kernel_block_id = torch.tensor(
            [[1], [2], [0]], dtype=torch.int32
        )
        devices = (attention.input_lengths_device, attention.prefix_lengths_device)
        self.prepare(model, SimpleNamespace(attention_inputs=attention))
        self.assertEqual(attention.cu_seqlens.tolist(), [0, 3, 5, 8])
        self.assertIs(attention.input_lengths_device, devices[0])
        self.assertIs(attention.prefix_lengths_device, devices[1])


if __name__ == "__main__":
    unittest.main()

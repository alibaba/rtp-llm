"""CPU coverage for encoder graph planning; no native RTP ops are needed."""

import importlib.util
import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch


def _load_graph_module():
    # Isolate the Python metadata contract from the GPU/native binding loader.
    # All tensor operations in the tests still use real CPU PyTorch.
    base = types.ModuleType(
        "rtp_llm.models_py.modules.factory.attention.fmha_impl_base"
    )
    base.FMHAImplBase = object
    ops = types.ModuleType("rtp_llm.ops.compute_ops")
    ops.ParamsBase = object
    path = (
        Path(__file__).resolve().parents[2]
        / "modules/factory/attention/bert_encoder_graph.py"
    )
    spec = importlib.util.spec_from_file_location(
        "encoder_graph_metadata_test_impl", path
    )
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {base.__name__: base, ops.__name__: ops}):
        spec.loader.exec_module(module)
    return module


class BertEncoderGraphMetadataTest(unittest.TestCase):
    def setUp(self):
        cls = _load_graph_module().BertEncoderGraphImpl
        self.impl = cls.__new__(cls)
        self.impl.request_capacity = 3
        self.impl.token_capacity = 8
        self.impl.max_sequence_length = None
        self.impl.heads = self.impl.kv_heads = 1
        self.impl.dim = 64
        self.impl.dtype = torch.bfloat16
        self.impl.cu_seqlens_host = torch.empty(5, dtype=torch.int32)
        self.impl.wrapper = Mock()

    @staticmethod
    def _inputs(lengths, prefix=None):
        return SimpleNamespace(
            input_lengths=torch.tensor(lengths, dtype=torch.int32),
            prefix_lengths=torch.tensor(
                prefix or [0] * len(lengths), dtype=torch.int32
            ),
        )

    def test_layout_changes_keep_storage_and_isolate_padding(self):
        pointer = self.impl.cu_seqlens_host.data_ptr()
        cases = [
            ([8, 0, 0], [0, 8, 8, 8, 8]),
            ([2, 3, 0], [0, 2, 5, 5, 8]),
            ([1, 1, 1], [0, 1, 2, 3, 8]),
            ([0, 0, 0], [0, 0, 0, 0, 8]),
            ([3, 0, 4], [0, 3, 3, 7, 8]),
        ]
        for lengths, expected in cases:
            with self.subTest(lengths=lengths):
                self.impl.prepare_cuda_graph(self._inputs(lengths))
                self.assertEqual(self.impl.cu_seqlens_host.tolist(), expected)
                self.assertEqual(self.impl.cu_seqlens_host.data_ptr(), pointer)
                kwargs = self.impl.wrapper.plan.call_args.kwargs
                self.assertFalse(kwargs["causal"])
                self.assertTrue(kwargs["disable_split_kv"])

    def test_incompatible_profiles_are_rejected(self):
        for lengths, prefix in [
            ([9, 0, 0], None),
            ([-1, 2, 0], None),
            ([1, 2, 0], [0, 1, 0]),
            ([1, 2], None),
        ]:
            with self.subTest(lengths=lengths, prefix=prefix):
                with self.assertRaises(ValueError):
                    self.impl.prepare_cuda_graph(self._inputs(lengths, prefix))

    def test_32_item_batch_uses_aggregate_bucket(self):
        self.impl.request_capacity = 32
        self.impl.token_capacity = 6144
        self.impl.max_sequence_length = 512
        self.impl.cu_seqlens_host = torch.empty(34, dtype=torch.int32)
        # Same item text lengths as mock_request/bert/request.json, plus four
        # visual rows and a seven-token query per item.
        item_lengths = [
            182,
            192,
            149,
            114,
            192,
            161,
            165,
            140,
            150,
            136,
            136,
            167,
            141,
            138,
            153,
            151,
            192,
            178,
            164,
            188,
            177,
            156,
            171,
            151,
            149,
            136,
            116,
            174,
            180,
            142,
            185,
            124,
        ]
        lengths = [length + 4 + 7 for length in item_lengths]
        self.impl.prepare_cuda_graph(self._inputs(lengths))
        self.assertEqual(int(self.impl.cu_seqlens_host[-2]), 5402)
        self.assertEqual(int(self.impl.cu_seqlens_host[-1]), 6144)
        self.assertEqual(self.impl.cu_seqlens_host.numel(), 34)

    def test_single_sequence_limit_is_independent_of_total_capacity(self):
        self.impl.token_capacity = 6144
        self.impl.max_sequence_length = 512
        with self.assertRaisesRegex(ValueError, "position-table"):
            self.impl.prepare_cuda_graph(self._inputs([513, 0, 0]))
        # The long synthetic sequence is legal only during initial capture.
        self.impl._prepare(self._inputs([6144, 0, 0]), for_capture=True)
        self.assertEqual(
            self.impl.cu_seqlens_host.tolist(), [0, 6144, 6144, 6144, 6144]
        )


if __name__ == "__main__":
    unittest.main()

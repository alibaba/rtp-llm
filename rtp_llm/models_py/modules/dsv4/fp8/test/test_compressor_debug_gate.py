"""Regression tests for the production compressor diagnostic gate."""

import unittest
from unittest import mock

from rtp_llm.models_py.modules.dsv4.fp8.compressor import CompressorFP8


class _ExplodingMeta:
    @property
    def positions(self):
        raise AssertionError("disabled diagnostics touched a CUDA tensor")

    @property
    def kv_slots(self):
        raise AssertionError("disabled diagnostics touched a CUDA tensor")


class CompressorDebugGateTest(unittest.TestCase):
    def test_disabled_recorder_returns_before_tensor_operations(self) -> None:
        compressor = CompressorFP8.__new__(CompressorFP8)
        compressor._profile_label = "L06.attn_cmp"

        with mock.patch(
            "rtp_llm.models_py.modules.dsv4.fp8.compressor._rt.should_record_layer",
            return_value=False,
        ) as should_record:
            compressor._debug_record_readonly_inputs(_ExplodingMeta(), object())

        should_record.assert_called_once_with(6)

    def test_non_layer6_returns_without_querying_recorder(self) -> None:
        compressor = CompressorFP8.__new__(CompressorFP8)
        compressor._profile_label = "L05.attn_cmp"

        with mock.patch(
            "rtp_llm.models_py.modules.dsv4.fp8.compressor._rt.should_record_layer"
        ) as should_record:
            compressor._debug_record_readonly_inputs(_ExplodingMeta(), object())

        should_record.assert_not_called()

    def test_disabled_pool_recorder_returns_before_tensor_operations(self) -> None:
        compressor = CompressorFP8.__new__(CompressorFP8)
        compressor._profile_label = "L17.csa_main"

        with mock.patch(
            "rtp_llm.models_py.modules.dsv4.fp8.compressor._rt.should_record_layer",
            return_value=False,
        ) as should_record:
            compressor._debug_record_pool_rows(_ExplodingMeta())

        should_record.assert_called_once_with(17)

    def test_unparseable_pool_label_returns_without_querying_recorder(self) -> None:
        compressor = CompressorFP8.__new__(CompressorFP8)
        compressor._profile_label = "compressor"

        with mock.patch(
            "rtp_llm.models_py.modules.dsv4.fp8.compressor._rt.should_record_layer"
        ) as should_record:
            compressor._debug_record_pool_rows(_ExplodingMeta())

        should_record.assert_not_called()


if __name__ == "__main__":
    unittest.main()

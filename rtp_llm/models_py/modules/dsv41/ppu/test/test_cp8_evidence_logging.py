"""Disabled INFO must not eagerly build CP device-backed YOCO evidence.

The actual CP compact activation and restore-head methods run on CPU. Only
the pinned upload and row restoration transport boundaries are doubled;
explicit evidence still evaluates its original position/count expressions.
"""

from contextlib import ExitStack, contextmanager
import logging
import os
import unittest
from unittest.mock import Mock, patch

import torch

from rtp_llm.models_py.modules.dsv41.ppu.test.test_cp8_cross_layer_mhc import _activate, _plan
from rtp_llm.models_py.modules.dsv41.ppu.test.test_cp8_metadata_reuse import ENV, _fixture


@contextmanager
def _environment(values):
    absent = object()
    previous = {name: os.environ.get(name, absent) for name in values}
    try:
        os.environ.update(values)
        yield
    finally:
        for name, value in previous.items():
            if value is absent:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


@contextmanager
def _logging_state(level, *, disable=logging.NOTSET, logger_disabled=False):
    logger = logging.getLogger()
    original = logger.level, logger.disabled, logger.manager.disable
    try:
        logger.setLevel(level)
        logger.disabled = logger_disabled
        logging.disable(disable)
        yield logger
    finally:
        logger.setLevel(original[0])
        logger.disabled = original[1]
        logging.disable(original[2])


def _prepared():
    state, _ = _fixture(rank=0, lengths=(256,) * 8, prefixes=(1024,) * 8)
    return state, _plan(state)


class CP8EvidenceLoggingContracts(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(torch.no_grad())
        self.stack.enter_context(_environment(ENV))
        self.stack.enter_context(patch("rtp_llm.models_py.modules.dsv41.ppu._quant.is_ppu", return_value=True))
        self.stack.enter_context(patch.object(torch.cuda, "is_current_stream_capturing", return_value=False))
        self.stack.enter_context(patch.object(torch.cuda, "current_stream",
                                             side_effect=AssertionError("CPU fixture must not initialize CUDA")))
        self.stack.enter_context(patch(
            "rtp_llm.models_py.modules.dsv41.fp8._v41_swa_metadata._upload",
            side_effect=lambda values, device, dtype: torch.tensor(values, device=device, dtype=dtype)))

    def test_warning_and_disable_skip_evidence_in_real_compact_and_restore_methods(self):
        for settings in (
            {"level": logging.WARNING},
            {"level": logging.INFO, "disable": logging.INFO},
            {"level": logging.INFO, "logger_disabled": True},
        ):
            with self.subTest(settings=settings), _logging_state(**settings) as logger:
                state, plan = _prepared()
                self.assertIsNotNone(state._cp_full_rows_birth)
                self.assertFalse(logger.isEnabledFor(logging.INFO))
                reduced = torch.empty((plan.context.chunk_length, 2), dtype=torch.bfloat16)
                restored = torch.empty((plan.original_context.chunk_length, 2), dtype=torch.bfloat16)
                evidence = Mock(side_effect=AssertionError("disabled INFO tried to read CP evidence from device"))
                with patch.object(state, "evidence", new=evidence), patch.object(logging, "info") as info:
                    _activate(state, plan)
                    self.assertTrue(state.compacted)
                    self.assertIsNotNone(state._cp_metadata_compact_birth)
                    with patch.object(plan, "restore_rows", return_value=restored) as restore:
                        result = state.restore_head(reduced)
                    restore.assert_called_once_with(reduced)
                    self.assertIs(result, restored)
                    self.assertTrue(state.head_restored)
                evidence.assert_not_called()
                info.assert_not_called()

    def test_info_preserves_original_payload_format_order_and_one_call_at_each_boundary(self):
        with _logging_state(logging.INFO):
            state, plan = _prepared()
            real_evidence = state.evidence
            reduced = torch.empty((plan.context.chunk_length, 2), dtype=torch.bfloat16)
            restored = torch.empty((plan.original_context.chunk_length, 2), dtype=torch.bfloat16)
            with patch.object(state, "evidence", wraps=real_evidence) as evidence, \
                 patch.object(logging, "info") as info:
                _activate(state, plan)
                self.assertEqual(evidence.call_count, 1)
                expected_compact = real_evidence()
                info.assert_called_once_with("PPU shared YOCO compact %s", expected_compact)
                with patch.object(plan, "restore_rows", return_value=restored) as restore:
                    self.assertIs(state.restore_head(reduced), restored)
                restore.assert_called_once_with(reduced)
                expected_restored = real_evidence()
                self.assertEqual(evidence.call_count, 2)
                self.assertEqual(info.call_count, 2)
                self.assertEqual(info.call_args_list[0].args, ("PPU shared YOCO compact %s", expected_compact))
                self.assertEqual(info.call_args_list[1].args, ("PPU shared YOCO restored %s", expected_restored))
                self.assertFalse(expected_compact["restored"])
                self.assertTrue(expected_restored["restored"])

    def test_explicit_evidence_keeps_position_and_valid_output_data_when_logging_is_disabled(self):
        for settings in ({"level": logging.WARNING}, {"level": logging.INFO, "disable": logging.CRITICAL}):
            with self.subTest(settings=settings), _logging_state(**settings):
                state, plan = _prepared()
                with patch.object(state, "evidence", side_effect=AssertionError("suppressed activation evidence")):
                    _activate(state, plan)
                self.assertIsNone(state._compact_host_evidence(), "this CP path still uses the original explicit readback")
                expected_positions = plan.context.global_positions.tolist()
                expected_valid = int(state.valid_output_rows.sum())
                with patch.object(logging, "info") as info:
                    result = state.evidence()
                info.assert_not_called()
                self.assertEqual(result["local_positions"], expected_positions)
                self.assertEqual(result["local_valid_output_rows"], expected_valid)
                self.assertEqual(result["cp_size"], 8)
                self.assertEqual(result["cp_rank"], 0)
                self.assertEqual(len(result["local_positions"]), 144)
                self.assertTrue(result["compacted"])
                self.assertFalse(result["restored"])

    def test_uncompacted_restore_is_still_identity_without_logging_or_evidence(self):
        with _logging_state(logging.INFO):
            state, _ = _prepared()
            reduced = torch.empty((state.context.chunk_length, 2), dtype=torch.bfloat16)
            with patch.object(state, "evidence", side_effect=AssertionError("no compact restoration")) as evidence, \
                 patch.object(logging, "info") as info:
                self.assertIs(state.restore_head(reduced), reduced)
            evidence.assert_not_called()
            info.assert_not_called()
            self.assertFalse(state.head_restored)


if __name__ == "__main__":
    unittest.main()

"""CPU-only contracts for the optional TP FP8 all-reduce configuration."""

import os
import unittest
from contextlib import nullcontext
from unittest.mock import MagicMock, patch

import torch
import torch.distributed as dist

from rtp_llm.models_py.distributed import collective_torch as collective
from rtp_llm.models_py.distributed import tp_fp8_all_reduce as fp8_ar


class TpFp8AllReduceConfigTest(unittest.TestCase):
    def test_defaults_are_disabled_and_have_documented_byte_bounds(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(fp8_ar.read_config_values(), ("0", "2097152", "134217728"))
            self.assertEqual(
                fp8_ar.TpFp8AllReduceConfig.from_values(fp8_ar.read_config_values()),
                fp8_ar.TpFp8AllReduceConfig(),
            )

    def test_enabled_is_strict_binary_switch(self):
        enabled = fp8_ar.TpFp8AllReduceConfig.from_values(("1", "64", "128"))
        self.assertTrue(enabled.enabled)
        for value in ("", "true", "True", "-1", "2"):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    fp8_ar.TpFp8AllReduceConfig.from_values((value, "64", "128"))

    def test_byte_bounds_reject_invalid_or_ambiguous_launch_ranges(self):
        invalid_values = (
            ("1", "63", "128"),
            ("1", "128", "127"),
            ("1", "64", "129"),
            ("1", "not-a-number", "128"),
        )
        for values in invalid_values:
            with self.subTest(values=values):
                with self.assertRaises(ValueError):
                    fp8_ar.TpFp8AllReduceConfig.from_values(values)
        self.assertEqual(
            fp8_ar.TpFp8AllReduceConfig.from_values(("1", "65", "128")).min_bytes,
            65,
        )

    def test_rank_mismatch_is_detected_before_any_low_precision_collective(self):
        local = ("1", "64", "128")

        def gather(output, value, group):
            self.assertEqual(value, local)
            output[:] = [local, ("1", "64", "256")]

        with (
            patch.dict(
                os.environ,
                {
                    "RTP_LLM_TP_FP8_ALLREDUCE": local[0],
                    "RTP_LLM_TP_FP8_ALLREDUCE_MIN_BYTES": local[1],
                    "RTP_LLM_TP_FP8_ALLREDUCE_MAX_BYTES": local[2],
                },
                clear=True,
            ),
            patch.object(dist, "get_world_size", return_value=2),
            patch.object(dist, "all_gather_object", side_effect=gather),
        ):
            with self.assertRaisesRegex(ValueError, "differs across ranks"):
                fp8_ar.validate_config(object())

    def test_matching_disabled_ranks_participate_in_the_same_validation(self):
        local = ("0", "2097152", "134217728")

        def gather(output, value, group):
            self.assertEqual(value, local)
            output[:] = [local, local]

        with (
            patch.dict(os.environ, {}, clear=True),
            patch.object(dist, "get_world_size", return_value=2),
            patch.object(dist, "all_gather_object", side_effect=gather) as gathered,
        ):
            config = fp8_ar.validate_config(object())
        self.assertFalse(config.enabled)
        gathered.assert_called_once()

    def test_single_rank_still_validates_the_common_raw_config_protocol(self):
        with (
            patch.dict(os.environ, {}, clear=True),
            patch.object(dist, "get_world_size", return_value=1),
            patch.object(
                dist,
                "all_gather_object",
                side_effect=lambda output, value, **_: output.__setitem__(
                    slice(None), [value]
                ),
            ) as gather,
        ):
            config = fp8_ar.validate_config(object())
        self.assertFalse(config.enabled)
        gather.assert_called_once()

    def test_default_off_does_not_construct_or_register_a_communicator(self):
        fp8_ar.destroy_tp_fp8_allreduce()
        with (
            patch.object(
                fp8_ar, "validate_config", return_value=fp8_ar.TpFp8AllReduceConfig()
            ),
            patch.object(
                fp8_ar,
                "TpFp8AllReduceCommunicator",
                side_effect=AssertionError(
                    "default-off must not allocate an IPC communicator"
                ),
            ),
        ):
            self.assertIsNone(fp8_ar.init_tp_fp8_allreduce(object(), "cuda:0"))
            self.assertIsNone(fp8_ar.get_tp_fp8_allreduce())


class TpFp8AllReduceApiUnitTest(unittest.TestCase):
    def _communicator(self):
        communicator = fp8_ar.TpFp8AllReduceCommunicator.__new__(
            fp8_ar.TpFp8AllReduceCommunicator
        )
        communicator._closed = False
        communicator.device = torch.device("cuda", 0)
        communicator.min_bytes = 64
        communicator.max_bytes = 128
        communicator.rank = 0
        return communicator

    def _tensor(self, **changes):
        tensor = MagicMock(spec=torch.Tensor)
        tensor.is_cuda = True
        tensor.device = torch.device("cuda", 0)
        tensor.dtype = torch.bfloat16
        tensor.is_contiguous.return_value = True
        tensor.numel.return_value = 32
        tensor.element_size.return_value = 2
        tensor.data_ptr.return_value = 16
        for name, value in changes.items():
            setattr(tensor, name, value)
        return tensor

    def test_should_use_uses_only_cross_rank_logical_tensor_gates(self):
        communicator = self._communicator()
        valid = self._tensor()
        with patch.object(
            torch.cuda, "is_current_stream_capturing", return_value=False
        ):
            self.assertTrue(communicator.should_use(valid))

        invalid = {
            "cpu": self._tensor(is_cuda=False),
            "fp16": self._tensor(dtype=torch.float16),
            "unaligned_numel": self._tensor(numel=lambda: 31),
            "large_bytes": self._tensor(numel=lambda: 96),
        }
        with patch.object(
            torch.cuda, "is_current_stream_capturing", return_value=False
        ):
            for name, tensor in invalid.items():
                with self.subTest(name=name):
                    self.assertFalse(communicator.should_use(tensor))
            for name, tensor in {
                "noncontiguous": self._tensor(is_contiguous=lambda: False),
                "unaligned_ptr": self._tensor(data_ptr=lambda: 17),
            }.items():
                with self.subTest(name=name):
                    self.assertTrue(communicator.should_use(tensor))
        with patch.object(torch.cuda, "is_current_stream_capturing", return_value=True):
            self.assertFalse(communicator.should_use(valid))
        with self.assertRaisesRegex(ValueError, "bound communicator"):
            communicator.should_use(self._tensor(device=torch.device("cuda", 1)))

        communicator.min_bytes = 128
        with patch.object(
            torch.cuda, "is_current_stream_capturing", return_value=False
        ):
            self.assertFalse(communicator.should_use(valid))

    def test_async_stages_noncontiguous_and_misaligned_input_and_output(self):
        communicator = self._communicator()
        communicator.calls = 0
        communicator._stream = MagicMock()
        communicator._native = MagicMock()
        tensor = self._tensor(is_contiguous=lambda: False, data_ptr=lambda: 17)
        tensor.shape = (4, 8)
        output = self._tensor(is_contiguous=lambda: False, data_ptr=lambda: 19)
        output.shape = tensor.shape
        staged_input = self._tensor()
        staged_output = self._tensor()
        staged_input.shape = tensor.shape
        staged_output.shape = tensor.shape
        producer, event = MagicMock(), MagicMock()
        with (
            patch.object(torch.cuda, "is_current_stream_capturing", return_value=False),
            patch.object(torch.cuda, "device", return_value=nullcontext()),
            patch.object(torch.cuda, "current_stream", return_value=producer),
            patch.object(torch.cuda, "stream", return_value=nullcontext()),
            patch.object(torch.cuda, "Event", return_value=event),
            patch.object(
                torch, "empty_like", side_effect=[staged_input, staged_output]
            ),
        ):
            pending = communicator.all_reduce_async(tensor, output)
        self.assertIs(pending.tensor, output)
        communicator._native.all_reduce.assert_called_once_with(
            staged_input, staged_output
        )
        output.copy_.assert_called_once_with(staged_output)
        communicator._stream.wait_stream.assert_called_once_with(producer)
        event.record.assert_called_once_with(communicator._stream)

    def test_pending_wait_records_each_consumer_stream(self):
        tensor = MagicMock(spec=torch.Tensor)
        tensor.device = torch.device("cuda", 0)
        event = MagicMock()
        first, second = MagicMock(), MagicMock()
        pending = fp8_ar.PendingTpFp8AllReduce(tensor, event)
        with patch.object(torch.cuda, "current_stream", side_effect=[first, second]):
            self.assertIs(pending.wait(), tensor)
            self.assertIs(pending.wait(), tensor)
        self.assertEqual(event.wait.call_count, 0)
        self.assertEqual(first.wait_event.call_count, 1)
        self.assertEqual(second.wait_event.call_count, 1)
        self.assertEqual(
            [call.args[0] for call in tensor.record_stream.call_args_list],
            [first, second],
        )

    def test_collective_sync_uses_fp8_only_when_eligible_and_preserves_alias(self):
        tensor = self._tensor()
        fp8 = MagicMock()
        fp8.should_use.return_value = True
        module = MagicMock()
        module.get_tp_fp8_allreduce.return_value = fp8
        with (
            patch.object(collective, "_get_rocm_rccl", return_value=None),
            patch.object(collective, "_get_tp_fp8_allreduce", return_value=module),
        ):
            result = collective.all_reduce(tensor, collective.Group.TP, inplace=True)
            self.assertIs(result, fp8.all_reduce.return_value)
            fp8.all_reduce.assert_called_once_with(tensor, out=tensor)
            fp8.reset_mock()
            result = collective.all_reduce(tensor, collective.Group.TP, inplace=False)
            self.assertIs(result, fp8.all_reduce.return_value)
            fp8.all_reduce.assert_called_once_with(tensor, out=None)

    def test_collective_async_uses_fp8_only_when_eligible_and_preserves_alias(self):
        tensor = self._tensor()
        fp8 = MagicMock()
        fp8.should_use.return_value = True
        module = MagicMock()
        module.get_tp_fp8_allreduce.return_value = fp8
        process_group = object()
        with (
            patch.object(torch.version, "hip", None),
            patch.object(torch.cuda, "is_current_stream_capturing", return_value=False),
            patch.object(collective, "_get_group", return_value=process_group),
            patch.object(dist, "get_backend", return_value="nccl"),
            patch.object(collective, "_get_tp_fp8_allreduce", return_value=module),
        ):
            self.assertIs(
                collective.all_reduce_async(tensor, collective.Group.TP, inplace=True),
                fp8.all_reduce_async.return_value,
            )
            fp8.all_reduce_async.assert_called_once_with(tensor, out=tensor)
            fp8.reset_mock()
            self.assertIs(
                collective.all_reduce_async(tensor, collective.Group.TP, inplace=False),
                fp8.all_reduce_async.return_value,
            )
            fp8.all_reduce_async.assert_called_once_with(tensor, out=None)

    def test_collective_ineligible_fp8_falls_back_to_nccl(self):
        tensor = self._tensor()
        fp8 = MagicMock()
        fp8.should_use.return_value = False
        module = MagicMock()
        module.get_tp_fp8_allreduce.return_value = fp8
        process_group, work = object(), MagicMock()
        with (
            patch.object(torch.version, "hip", None),
            patch.object(torch.cuda, "is_current_stream_capturing", return_value=False),
            patch.object(collective, "_get_group", return_value=process_group),
            patch.object(dist, "get_backend", return_value="nccl"),
            patch.object(collective, "_get_tp_fp8_allreduce", return_value=module),
            patch.object(dist, "all_reduce", return_value=work) as all_reduce,
        ):
            pending = collective.all_reduce_async(tensor, collective.Group.TP)
        self.assertIsInstance(pending, collective.PendingAllReduce)
        self.assertIs(pending.tensor, tensor)
        all_reduce.assert_called_once_with(
            tensor,
            op=dist.ReduceOp.SUM,
            group=process_group,
            async_op=True,
        )

    def test_collective_sync_ineligible_fp8_uses_full_precision_nccl(self):
        tensor = self._tensor()
        fp8 = MagicMock()
        fp8.should_use.return_value = False
        module = MagicMock()
        module.get_tp_fp8_allreduce.return_value = fp8
        process_group = object()
        disabled_custom = MagicMock()
        disabled_custom.get_flashinfer_allreduce.return_value = None
        disabled_symm = MagicMock()
        disabled_symm.get_symm_mem_communicator.return_value = None
        with (
            patch.object(collective, "_get_rocm_rccl", return_value=None),
            patch.object(collective, "_get_tp_fp8_allreduce", return_value=module),
            patch.object(
                collective, "_get_flashinfer_allreduce", return_value=disabled_custom
            ),
            patch.object(collective, "_get_symm_mem", return_value=disabled_symm),
            patch.object(collective, "_get_group", return_value=process_group),
            patch.object(dist, "all_reduce") as all_reduce,
        ):
            self.assertIs(collective.all_reduce(tensor, collective.Group.TP), tensor)
        all_reduce.assert_called_once_with(
            tensor, op=dist.ReduceOp.SUM, group=process_group
        )


if __name__ == "__main__":
    unittest.main()

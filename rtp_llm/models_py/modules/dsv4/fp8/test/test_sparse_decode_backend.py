"""Device binding and planner lifetime for sparse decode backends."""

import unittest
from unittest.mock import Mock, patch

import torch

from rtp_llm.models_py.modules.dsv4.fp8.decode import fp8_sparse_attn_decode_op
from rtp_llm.models_py.modules.dsv4.fp8.decode.fp8_sparse_attn_decode_op import (
    SparseAttnV4DecodeFp8Op,
)


class SparseDecodeBackendTest(unittest.TestCase):
    def test_construction_does_not_load_provider_or_choose_current_device(self):
        with (
            patch.object(fp8_sparse_attn_decode_op, "_load_flash_mla") as load,
            patch.object(
                torch.cuda,
                "current_device",
                side_effect=AssertionError("implicit device"),
            ),
        ):
            SparseAttnV4DecodeFp8Op(8, 512, 1.0)
        load.assert_not_called()

    def test_metadata_and_forward_share_binding_but_not_step_metadata(self):
        op = SparseAttnV4DecodeFp8Op(8, 512, 1.0)
        query = torch.zeros(1, 1, 8, 512, dtype=torch.bfloat16)
        cache, sink, indices = object(), object(), object()
        step_metadata = [object(), object()]
        schedulers = [object(), object()]
        output = object()
        with (
            patch.object(
                fp8_sparse_attn_decode_op, "_load_flash_mla", return_value=True
            ) as load,
            patch.object(op, "_forward_flash_mla", return_value=output) as forward,
            patch(
                "rtp_llm.models_py.modules.dsv4.fp8.decode.decode_attn_metadata.get_or_build_sched_meta",
                side_effect=schedulers,
            ) as build,
        ):
            for metadata, expected in zip(step_metadata, schedulers):
                scheduler = op.get_sched_meta(
                    query.device,
                    metadata,
                    batch_size=1,
                    q_len=1,
                    topk=128,
                    extra_attn_type="HCA_KV",
                )
                self.assertIs(scheduler, expected)
                self.assertIs(
                    op.forward(query, cache, sink, indices, scheduler), output
                )
                build.assert_called_with(
                    metadata,
                    batch_size=1,
                    q_len=1,
                    num_heads=8,
                    topk=128,
                    extra_attn_type="HCA_KV",
                )
            load.assert_called_once_with()
            self.assertEqual(build.call_count, 2)
            self.assertEqual(forward.call_count, 2)
            self.assertIs(forward.call_args.kwargs["sched_meta"], schedulers[1])
            self.assertIs(forward.call_args.kwargs["q"], query)

    def test_device_change_fails_before_backend_execution(self):
        op = SparseAttnV4DecodeFp8Op(8, 512, 1.0)
        with (
            patch.object(
                fp8_sparse_attn_decode_op, "_load_flash_mla", return_value=True
            ),
            patch.object(op, "_forward_flash_mla") as forward,
        ):
            op.bind_device(torch.device("cpu"))
            query = Mock(device=torch.device("cuda:1"))
            with self.assertRaisesRegex(ValueError, "bound to cpu, got cuda:1"):
                op.forward(query, None, None, None, None)
            forward.assert_not_called()

    def test_indexless_cuda_device_is_not_inferred_from_current_device(self):
        op = SparseAttnV4DecodeFp8Op(8, 512, 1.0)
        with (
            patch.object(op, "_prepare_backend") as prepare,
            patch.object(
                torch.cuda,
                "current_device",
                side_effect=AssertionError("implicit device"),
            ),
        ):
            with self.assertRaisesRegex(ValueError, "explicit CUDA device index"):
                op.bind_device(torch.device("cuda"))
        prepare.assert_not_called()

    def test_missing_provider_fails_before_planning_and_does_not_bind(self):
        op = SparseAttnV4DecodeFp8Op(8, 512, 1.0)
        with (
            patch.object(
                fp8_sparse_attn_decode_op, "_load_flash_mla", return_value=False
            ),
            patch(
                "rtp_llm.models_py.modules.dsv4.fp8.decode.decode_attn_metadata.get_or_build_sched_meta"
            ) as build,
        ):
            with self.assertRaisesRegex(RuntimeError, "FlashMLA is required"):
                op.get_sched_meta(
                    torch.device("cpu"),
                    object(),
                    batch_size=1,
                    q_len=1,
                    topk=128,
                    extra_attn_type=None,
                )
            build.assert_not_called()
        # Failed initialization must not leave a partially initialized op.
        with patch.object(
            fp8_sparse_attn_decode_op, "_load_flash_mla", return_value=True
        ) as load:
            op.bind_device(torch.device("cpu"))
            load.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()

"""SM120 backend selection uses the query device for planning and execution."""

import unittest
from unittest.mock import Mock, patch

import torch

from rtp_llm.models_py.modules.dsv4.fp8 import (
    attention_backend,
    backend_selection,
    sm120_sparse_mla,
)
from rtp_llm.models_py.modules.dsv4.fp8.decode import fp8_sparse_attn_decode_op
from rtp_llm.models_py.modules.dsv4.fp8.decode.fp8_sparse_attn_decode_op import (
    SparseAttnV4DecodeFp8Op,
)


class SparseDecodeBackendSm120Test(unittest.TestCase):
    def setUp(self):
        for cached in (
            backend_selection._is_cuda_runtime,
            backend_selection._is_sm120_index,
            attention_backend._backend_for_cuda,
        ):
            cached.cache_clear()
            self.addCleanup(cached.cache_clear)
        runtime = patch.object(backend_selection, "is_cuda", return_value=True)
        runtime.start()
        self.addCleanup(runtime.stop)

    @patch(
        "rtp_llm.models_py.modules.dsv4.fp8.decode.decode_attn_metadata."
        "get_or_build_sched_meta"
    )
    def test_sm120_decode_does_not_build_flash_mla_metadata(self, build):
        device = torch.device("cuda:7")
        op = SparseAttnV4DecodeFp8Op(8, 512, 1.0)
        query = Mock(device=device)
        output = object()
        with (
            patch.object(
                torch.cuda,
                "current_device",
                side_effect=AssertionError("implicit device"),
            ),
            patch.object(
                backend_selection, "is_sm120", side_effect=lambda d: d == device
            ) as select,
            patch.object(fp8_sparse_attn_decode_op, "_load_flash_mla") as flash_mla,
            patch.object(
                sm120_sparse_mla, "_load_sm120_sparse_mla", return_value=object()
            ) as provider,
            patch.object(sm120_sparse_mla, "warmup") as warmup,
            patch.object(
                sm120_sparse_mla, "forward_decode", return_value=output
            ) as forward,
        ):
            result = op.get_sched_meta(
                device,
                object(),
                batch_size=2,
                q_len=1,
                topk=64,
                extra_attn_type=None,
            )
            self.assertIsNone(result)
            for _ in range(2):
                self.assertIs(op.forward(query, None, None, None, result), output)
            select.assert_called_once_with(device)
            provider.assert_called_once_with()
            warmup.assert_called_once_with(device)
            flash_mla.assert_not_called()
            build.assert_not_called()
            self.assertEqual(forward.call_count, 2)
            self.assertIs(forward.call_args.kwargs["q"], query)

    @patch(
        "rtp_llm.models_py.modules.dsv4.fp8.decode.decode_attn_metadata."
        "get_or_build_sched_meta"
    )
    def test_non_sm120_decode_builds_flash_mla_metadata(self, build):
        sentinel, metadata = object(), object()
        build.return_value = sentinel
        device = torch.device("cuda:3")
        op = SparseAttnV4DecodeFp8Op(8, 512, 1.0)
        with (
            patch.object(backend_selection, "is_sm120", return_value=False) as select,
            patch.object(
                fp8_sparse_attn_decode_op, "_load_flash_mla", return_value=True
            ),
            patch.object(sm120_sparse_mla, "_load_sm120_sparse_mla") as provider,
            patch.object(
                torch.cuda,
                "current_device",
                side_effect=AssertionError("implicit device"),
            ),
        ):
            result = op.get_sched_meta(
                device,
                metadata,
                batch_size=2,
                q_len=1,
                topk=64,
                extra_attn_type="HCA_KV",
            )
            self.assertIs(result, sentinel)
            select.assert_called_once_with(device)
            provider.assert_not_called()
        build.assert_called_once_with(
            metadata,
            batch_size=2,
            q_len=1,
            num_heads=8,
            topk=64,
            extra_attn_type="HCA_KV",
        )

    def test_missing_sm120_provider_fails_before_workspace_allocation(self):
        op = SparseAttnV4DecodeFp8Op(8, 512, 1.0)
        with (
            patch.object(backend_selection, "is_sm120", return_value=True),
            patch.object(sm120_sparse_mla, "_load_sm120_sparse_mla", return_value=None),
            patch.object(sm120_sparse_mla, "warmup") as warmup,
        ):
            with self.assertRaisesRegex(RuntimeError, "requires FlashInfer"):
                op.bind_device(torch.device("cuda:2"))
            warmup.assert_not_called()

    def test_workspace_failure_does_not_publish_binding(self):
        op = SparseAttnV4DecodeFp8Op(8, 512, 1.0)
        device = torch.device("cuda:2")
        with (
            patch.object(backend_selection, "is_sm120", return_value=True),
            patch.object(
                sm120_sparse_mla, "_load_sm120_sparse_mla", return_value=object()
            ),
            patch.object(
                sm120_sparse_mla,
                "warmup",
                side_effect=[RuntimeError("workspace"), None],
            ) as warmup,
        ):
            with self.assertRaisesRegex(RuntimeError, "workspace"):
                op.bind_device(device)
            op.bind_device(device)
            self.assertEqual(warmup.call_count, 2)
            warmup.assert_called_with(device)


if __name__ == "__main__":
    unittest.main()

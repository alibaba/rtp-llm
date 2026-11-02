from types import SimpleNamespace
from unittest import TestCase, main, skipUnless
from unittest.mock import patch

import torch

from rtp_llm.models.qwen4_exp.qwen4_exp_kv_cache import PLE_STATE_TAG
from rtp_llm.models_py.model_desc.qwen4_exp_runtime.ple_cache_views import PLECacheViews


@skipUnless(torch.cuda.is_available(), "CUDA page-table validation is required")
class PLEHostPagePairsTest(TestCase):
    def test_trusted_prefill_host_table_avoids_gpu_scalar_read(self):
        from rtp_llm.ops.compute_ops import PyAttentionInputs

        host_table = torch.tensor(
            [[1, 2], [3, 4]], device="cpu", dtype=torch.int32
        ).pin_memory()
        inputs = PyAttentionInputs()
        inputs.is_prefill = True
        inputs.kv_cache_block_id = host_table
        inputs.kv_cache_block_id_device = host_table.to("cuda")
        rows = torch.tensor([0, 1], dtype=torch.long)
        pages = torch.tensor([1, 0], dtype=torch.long)
        original_item = torch.Tensor.item

        def no_gpu_item(tensor, *args, **kwargs):
            if tensor.is_cuda:
                raise AssertionError("trusted prefill performed a GPU scalar read")
            return original_item(tensor, *args, **kwargs)

        with patch.object(torch.Tensor, "item", no_gpu_item):
            blocks, writable = PLECacheViews._physical_blocks_for_pairs(
                inputs,
                PLE_STATE_TAG,
                rows,
                pages,
                5,
                torch.device("cuda"),
                trusted_host_table=True,
            )
            host_table[0, 1] = 5
            with self.assertRaisesRegex(RuntimeError, "physical block id exceeds"):
                PLECacheViews._physical_blocks_for_pairs(
                    inputs,
                    PLE_STATE_TAG,
                    rows,
                    pages,
                    5,
                    torch.device("cuda"),
                    trusted_host_table=True,
                )
        self.assertEqual(blocks.tolist(), [2, 3])
        self.assertEqual(writable.tolist(), [True, True])

    def test_host_staging_with_cuda_as_default_device(self):
        from rtp_llm.models_py.triton_kernels.causal_conv1d.causal_conv1d import (
            prepare_causal_conv1d_metadata,
        )

        rows = torch.tensor([0, 1], device="cpu", dtype=torch.long)
        pages = torch.tensor([1, 0], device="cpu", dtype=torch.long)
        cu = torch.tensor([0, 3, 20], device="cpu", dtype=torch.int32)
        table = torch.tensor([[1, 2], [3, 4]], device="cuda", dtype=torch.int32)
        inputs = SimpleNamespace(kv_cache_block_id_device=table)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream), torch.device("cuda"):
            blocks, writable = PLECacheViews._physical_blocks_for_pairs(
                inputs, PLE_STATE_TAG, rows, pages, 5, table.device
            )
            metadata = prepare_causal_conv1d_metadata(cu, table.device)
        stream.synchronize()
        self.assertEqual(blocks.tolist(), [2, 3])
        self.assertEqual(writable.tolist(), [True, True])
        self.assertEqual(metadata.total, 4)
        self.assertEqual(metadata.batch_ptr.tolist(), [0, 1, 1, 1, -1])
        self.assertEqual(metadata.token_chunk_offset_ptr.tolist(), [0, 0, 1, 2, -1])

    def test_host_pairs_on_nondefault_stream_and_device_fallback(self):
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            storage = torch.full((2, 4), 99, device="cuda", dtype=torch.int32)
            table = storage[:, :3]
            table.copy_(torch.tensor([[1, 0, -1], [2, 3, 4]], dtype=torch.int32))
            inputs = SimpleNamespace(kv_cache_block_id_device=table)
            for source in ("cpu", "cuda"):
                rows = torch.tensor([0, 0, 0, 1], dtype=torch.long, device=source)
                pages = torch.tensor([0, 1, 2, 2], dtype=torch.long, device=source)
                blocks, writable = PLECacheViews._physical_blocks_for_pairs(
                    inputs, PLE_STATE_TAG, rows, pages, 5, table.device
                )
                self.assertEqual(blocks.tolist(), [1, 0, -1, 4])
                self.assertEqual(writable.tolist(), [True, False, False, True])
            empty = torch.empty(0, dtype=torch.long)
            blocks, writable = PLECacheViews._physical_blocks_for_pairs(
                inputs, PLE_STATE_TAG, empty, empty, 5, table.device
            )
            self.assertEqual(blocks.numel(), 0)
            self.assertEqual(writable.numel(), 0)
            self.assertEqual(storage[:, 3].tolist(), [99, 99])
        stream.synchronize()

    def test_invalid_host_indices_and_live_device_blocks_are_rejected(self):
        table = torch.tensor([[1, 0, -1], [2, 3, 4]], device="cuda", dtype=torch.int32)
        inputs = SimpleNamespace(kv_cache_block_id_device=table)
        original = table.clone()
        for row, page, message in (
            (-1, 0, "block-table row"),
            (2, 0, "block-table row"),
            (0, -1, "invalid logical page"),
            (0, 3, "logical page exceeds"),
        ):
            with self.subTest(row=row, page=page):
                with self.assertRaisesRegex(RuntimeError, message):
                    PLECacheViews._physical_blocks_for_pairs(
                        inputs,
                        PLE_STATE_TAG,
                        torch.tensor([row]),
                        torch.tensor([page]),
                        5,
                        table.device,
                    )
        torch.testing.assert_close(table, original, rtol=0, atol=0)
        table[0, 0] = 5
        with self.assertRaisesRegex(RuntimeError, "physical block id exceeds"):
            PLECacheViews._physical_blocks_for_pairs(
                inputs,
                PLE_STATE_TAG,
                torch.tensor([0]),
                torch.tensor([0]),
                5,
                table.device,
            )


if __name__ == "__main__":
    main()

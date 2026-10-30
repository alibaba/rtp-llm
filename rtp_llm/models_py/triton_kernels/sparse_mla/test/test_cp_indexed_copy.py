import unittest

import torch
from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.cp_kv_prefetch import (
    CrossLayerKvPrefetch,
)
from rtp_llm.models_py.triton_kernels.sparse_mla.cp_indexed_copy import (
    try_indexed_copy,
)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class CPIndexedCopyTest(unittest.TestCase):

    def test_indexed_copy_preserves_other_rows(self):
        for dtype in (torch.bfloat16, torch.float16, torch.float32):
            source = torch.randn(521, 593, device="cuda", dtype=dtype)[:, :576]
            source[0, :3] = torch.tensor(
                [float("nan"), float("inf"), -0.0], device="cuda"
            )
            src_ids = torch.arange(517, device="cuda") % 521
            dst_ids = torch.arange(517, device="cuda") * 2 + 1
            out = torch.randn(1051, 587, device="cuda", dtype=dtype)[:, :576]
            ref = out.clone()
            ref.index_copy_(0, dst_ids, source.index_select(0, src_ids))
            self.assertTrue(try_indexed_copy(source, src_ids, dst_ids, out))
            self.assertTrue(
                torch.equal(out.contiguous().view(torch.uint8), ref.view(torch.uint8))
            )
            self.assertFalse(try_indexed_copy(source, src_ids, src_ids, source))

    def test_consume_after_history_event(self):
        # Exercise the real consume method: producer writes prefix + holes;
        # compute must preserve that prefix while replacing only new rows.
        producer = torch.cuda.Stream()
        compute = torch.cuda.current_stream()
        source = torch.randn(521, 576, device="cuda", dtype=torch.bfloat16)
        src_ids = torch.arange(517, device="cuda")
        dst_ids = src_ids * 2 + 1
        out = torch.empty(1051, 576, device="cuda", dtype=torch.bfloat16)
        pipeline = object.__new__(CrossLayerKvPrefetch)
        pipeline.fuse_new_restore = True
        pipeline.buffers = [out, out]
        pipeline.plan = {"new_restore": src_ids, "new_dest": dst_ids}
        producer.wait_stream(compute)
        for layer in range(1, 5):
            producer.wait_stream(compute)
            with torch.cuda.stream(producer):
                out.fill_(layer)
                ready = torch.cuda.Event()
                ready.record()
            pipeline.owners = [layer, layer]
            pipeline.pending_new = source
            compute.wait_event(ready)
            pipeline.consume(layer)
            self.assertIsNone(pipeline.pending_new)
            expected = torch.full_like(out, layer)
            expected.index_copy_(0, dst_ids, source.index_select(0, src_ids))
            self.assertTrue(torch.equal(out, expected))


if __name__ == "__main__":
    unittest.main()

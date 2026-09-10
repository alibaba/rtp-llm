import unittest

import torch
from k3_trace_buffers import K3SharedTraceBuffers


class SharedTraceTest(unittest.TestCase):
    def test_snapshot_masks_unwritten_rows_and_survives_buffer_reuse(self):
        trace = K3SharedTraceBuffers(3, 128, "cpu")
        trace.buffer.fill_(255)
        trace.reset()
        values = torch.linspace(-8, 8, 256).reshape(2, 128)
        trace.tensors["fc1_accumulator"][1].copy_(values)
        trace.tensors["fc1_rounded"][1].copy_(values)
        trace.valid[1] = 1
        snapshot = trace.snapshot()
        trace.buffer.fill_(255)
        self.assertEqual(snapshot["valid"].tolist(), [False, True, False])
        for name, dtype in (
            ("fc1_accumulator", torch.float32),
            ("fc1_rounded", torch.bfloat16),
        ):
            torch.testing.assert_close(
                snapshot[name][1], values.to(dtype), rtol=0, atol=0
            )
            self.assertEqual(torch.count_nonzero(snapshot[name][[0, 2]]).item(), 0)
        trace.reset()
        self.assertEqual(trace.valid.tolist(), [0, 0, 0])
        self.assertEqual(trace.overflow.item(), 0)

    def test_rejects_invalid_capacity_and_channel_alignment(self):
        for capacity, width in ((0, 128), (-1, 128), (1, 0), (1, 129)):
            with self.assertRaises(ValueError):
                K3SharedTraceBuffers(capacity, width, "cpu")


if __name__ == "__main__":
    unittest.main()

import os, sys, unittest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from sparse_kv_fp8 import FP8_KV_DIM, pack_kv_fp8, unpack_kv_fp8
class TestSparseKvFp8(unittest.TestCase):
    def test_roundtrip_error_small(self):
        torch.manual_seed(0)
        kv = (torch.randn(4096, 576, device="cuda") * 0.1).bfloat16()
        buf = pack_kv_fp8(kv)
        self.assertEqual(tuple(buf.shape), (4096, FP8_KV_DIM))
        self.assertEqual(buf.dtype, torch.uint8)
        deq = unpack_kv_fp8(buf).float()
        rel = ((deq - kv.float()).norm() / kv.float().norm()).item()
        self.assertLess(rel, 0.05, f"relative Frobenius error too large: {rel}")
        self.assertLess((deq[:, 512:] - kv.float()[:, 512:]).abs().max().item(), 1e-3)
    def test_shapes_rejected(self):
        kv = torch.zeros(4, 100, device="cuda")
        with self.assertRaises(ValueError):
            pack_kv_fp8(kv)
if __name__ == "__main__":
    unittest.main()

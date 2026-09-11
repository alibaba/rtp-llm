import unittest

import torch

from rtp_llm.cpp.model_rpc.proto.model_rpc_service_pb2 import TensorPB
from rtp_llm.utils.grpc_util import trans_from_tensor, trans_tensor


class GrpcUtilTensorTest(unittest.TestCase):
    def test_empty_tensor_shapes_round_trip(self):
        for shape in ((0,), (0, 4), (1, 1, 0)):
            with self.subTest(shape=shape):
                tensor = torch.empty(shape, dtype=torch.bfloat16)
                restored = trans_tensor(trans_from_tensor(tensor))
                self.assertEqual(restored.shape, shape)
                self.assertEqual(restored.dtype, torch.bfloat16)
                self.assertEqual(restored.numel(), 0)

    def test_bfloat16_multidimensional_round_trip(self):
        tensor = torch.arange(6, dtype=torch.int16).view(torch.bfloat16).reshape(2, 3)
        restored = trans_tensor(trans_from_tensor(tensor))
        self.assertEqual(restored.dtype, torch.bfloat16)
        self.assertEqual(restored.shape, (2, 3))
        self.assertTrue(torch.equal(restored, tensor))

    def test_rejects_payload_for_zero_element_shape(self):
        for shape in ((0,), (0, 4), (1, 1, 0)):
            with self.subTest(shape=shape):
                tensor = TensorPB(data_type=TensorPB.DataType.BF16, shape=shape)
                tensor.bf16_data = b"\x00\x00"
                with self.assertRaisesRegex(ValueError, "zero-element"):
                    trans_tensor(tensor)

    def test_rejects_malformed_non_empty_payload(self):
        tensor = TensorPB(data_type=TensorPB.DataType.BF16, shape=[2, 3])
        tensor.bf16_data = b"\x00" * 10
        with self.assertRaisesRegex(ValueError, "byte length mismatch"):
            trans_tensor(tensor)


if __name__ == "__main__":
    unittest.main()

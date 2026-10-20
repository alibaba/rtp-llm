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

    def test_rejects_unsupported_dtype_before_decoding_payload(self):
        for shape, payload in (([], b""), ([1], b"\x00\x00")):
            with self.subTest(shape=shape):
                tensor = TensorPB(data_type=99, shape=shape, bf16_data=payload)
                with self.assertRaisesRegex(ValueError, "unsupported TensorPB data_type"):
                    trans_tensor(tensor)

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

    def test_trans_from_tensor_detaches_autograd_tensor(self):
        tensor = torch.arange(6, dtype=torch.float16, requires_grad=True).reshape(2, 3)

        tensor_pb = trans_from_tensor(tensor)
        restored = trans_tensor(tensor_pb)

        self.assertFalse(restored.requires_grad)
        torch.testing.assert_close(restored, tensor.detach())

    def test_noncontiguous_bf16_roundtrip(self):
        base = torch.arange(24, dtype=torch.bfloat16).reshape(4, 6)
        for tensor in (base.transpose(0, 1), base[:, ::2]):
            with self.subTest(stride=tensor.stride()):
                self.assertFalse(tensor.is_contiguous())
                torch.testing.assert_close(
                    trans_tensor(trans_from_tensor(tensor)), tensor
                )

    def test_trans_from_tensor_fills_existing_message(self):
        tensor = torch.arange(6, dtype=torch.bfloat16).reshape(2, 3)
        tensor_pb = trans_from_tensor(torch.ones(1, dtype=torch.float32))
        tensor_pb.fp32_data = b"stale"

        returned = trans_from_tensor(tensor, tensor_pb)

        self.assertIs(returned, tensor_pb)
        self.assertEqual(trans_from_tensor(tensor).data_type, tensor_pb.data_type)
        self.assertEqual([2, 3], list(tensor_pb.shape))
        self.assertEqual(b"", tensor_pb.fp32_data)
        torch.testing.assert_close(trans_tensor(tensor_pb), tensor)


if __name__ == "__main__":
    unittest.main()

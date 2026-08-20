from unittest import TestCase, main

import torch

from rtp_llm.utils.grpc_util import trans_from_tensor, trans_tensor


class GrpcUtilTest(TestCase):
    def test_trans_from_tensor_detaches_autograd_tensor(self):
        tensor = torch.arange(6, dtype=torch.float16, requires_grad=True).reshape(2, 3)

        tensor_pb = trans_from_tensor(tensor)
        restored = trans_tensor(tensor_pb)

        self.assertFalse(restored.requires_grad)
        torch.testing.assert_close(restored, tensor.detach())

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
    main()

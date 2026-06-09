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


if __name__ == "__main__":
    main()

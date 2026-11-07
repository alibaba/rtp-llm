import unittest
import torch
from rtp_llm.multimodal.kvcm.registered_transport import RegisteredObjectTransport


class RegisteredObjectTransportTest(unittest.TestCase):
    def test_invalid_configuration_does_not_construct_client(self):
        def forbidden(**kw):
            self.fail('invalid configuration constructed client')
        for capacity in (0, -1, 1024**3 + 1):
            with self.assertRaises(ValueError):
                RegisteredObjectTransport(forbidden, max_object_bytes=capacity, device='cuda:0')
        with self.assertRaises(ValueError):
            RegisteredObjectTransport(forbidden, max_object_bytes=1024, device='cpu')

    @unittest.skipUnless(torch.cuda.is_available(), 'requires CUDA')
    def test_registered_range_roundtrip_bounds_and_close(self):
        seen = {}
        class Client:
            instance_id = 'test'
            instance_group = 'test'
            def __init__(self, gpu_tensors):
                self.arena = gpu_tensors[0]
                seen['client'] = self
                self.closed = 0
            def save_one(self, key, value, **kwargs):
                assert value.device.type == 'cuda'
                assert value.data_ptr() >= self.arena.data_ptr() + 65536
                assert value.data_ptr() + value.numel() <= self.arena.data_ptr() + self.arena.numel()
                self.value = value.clone()
            def load_one(self, key, value, **kwargs):
                value.copy_(self.value)
            def close(self):
                self.closed += 1
        transport = RegisteredObjectTransport(Client, max_object_bytes=1024, device='cuda:0')
        source = torch.arange(1024).to(torch.uint8)
        target = torch.empty_like(source)
        transport.save_one('x', source)
        transport.load_one('x', target)
        self.assertTrue(torch.equal(source, target))
        for bad in (torch.empty(1025,dtype=torch.uint8), torch.empty(0,dtype=torch.uint8), torch.empty(3), torch.empty((2,2),dtype=torch.uint8)):
            with self.assertRaises(ValueError): transport.save_one('x',bad)
        transport.close(); transport.close()
        self.assertEqual(seen['client'].closed,1)
        self.assertIsNone(transport._arena)
        with self.assertRaises(RuntimeError): transport.load_one('x',target)


if __name__ == '__main__': unittest.main()

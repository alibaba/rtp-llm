"""Mixed-precision K3 convolution preserves BF16 activation and cache storage."""
import unittest
import torch
import torch.nn.functional as F
from rtp_llm.models_py.triton_kernels.causal_conv1d import causal_conv1d_fn, causal_conv1d_update


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA convolution test')
class KdaConvPrecisionTest(unittest.TestCase):
    def test_reserved_zero_page_is_opt_in(self):
        x = torch.arange(16 * 8, device="cuda", dtype=torch.float32).reshape(8, 16).bfloat16()
        weight = torch.ones((16, 4), device="cuda", dtype=torch.float32)
        ptr = torch.tensor([0, 5, 8], device="cuda", dtype=torch.int32)
        table = torch.tensor([[1], [0]], device="cuda", dtype=torch.int32)
        prefix = torch.zeros(2, device="cuda", dtype=torch.int32)
        outputs = []
        for reserved in (None, 0):
            state = torch.full((2, 3, 16), -7, device="cuda", dtype=torch.bfloat16)
            kwargs = {} if reserved is None else {"reserved_cache_block_id": reserved}
            outputs.append(causal_conv1d_fn(x.T, weight, None, state.transpose(1, 2),
                ptr, table, prefix, 128, preserve_input_dtype=True, **kwargs))
            torch.testing.assert_close(state[1], x[2:5], rtol=0, atol=0)
            expected_zero = x[5:8] if reserved is None else torch.full_like(state[0], -7)
            torch.testing.assert_close(state[0], expected_zero, rtol=0, atol=0)
        torch.testing.assert_close(outputs[0], outputs[1], rtol=0, atol=0)

    def test_fp32_weights_bf16_cache_prefill_and_decode(self):
        generator = torch.Generator().manual_seed(93)
        source = torch.rand(83,16,generator=generator).bfloat16()
        weights = torch.rand(16,4,generator=generator).float()
        expected = F.conv1d(F.pad(source.T.double()[None],(3,0)),
                            weights.double()[:,None],groups=16)
        expected = F.silu(expected)[0].T.bfloat16()
        x,w = source.cuda(),weights.cuda()
        table = torch.tensor([[1,2,3]],dtype=torch.int32,device='cuda')
        def cache():
            return torch.zeros((4,3,16),dtype=torch.bfloat16,device='cuda').transpose(1,2)
        def prefill(inputs,state,prefix=0):
            return causal_conv1d_fn(inputs.T,w,None,state,
                torch.tensor([0,len(inputs)],dtype=torch.int32,device='cuda'),table,
                torch.tensor([prefix],dtype=torch.int32,device='cuda'),64,
                preserve_input_dtype=True).T
        full_cache = cache()
        full = prefill(x,full_cache)
        self.assertEqual(full.dtype,torch.bfloat16)
        self.assertEqual(full_cache.dtype,torch.bfloat16)
        # FP32 arithmetic may choose either neighboring BF16 value at a tie.
        lower = torch.nextafter(expected,torch.full_like(expected,-float('inf')))
        upper = torch.nextafter(expected,torch.full_like(expected,float('inf')))
        self.assertTrue(bool(((full.cpu()>=lower)&(full.cpu()<=upper)).all()))
        split_cache = cache()
        prefill(x[:64],split_cache)
        tail = prefill(x[64:],split_cache,64)
        torch.testing.assert_close(tail,full[64:],rtol=0,atol=0)
        torch.testing.assert_close(split_cache,full_cache,rtol=0,atol=0)
        decode_cache = cache()
        prefill(x[:-1],decode_cache)
        decoded = causal_conv1d_update(x[-1:].T[None],decode_cache,w,activation='silu',
            block_map=table,seq_size_per_block=64,
            sequence_lengths=torch.tensor([83],dtype=torch.int32,device='cuda'))[0].T
        self.assertEqual(decoded.dtype,torch.bfloat16)
        self.assertTrue(bool(((decoded.cpu()>=lower[-1:])&(decoded.cpu()<=upper[-1:])).all()))
        torch.testing.assert_close(decode_cache[2].T,x[-3:],rtol=0,atol=0)
        torch.testing.assert_close(full_cache[2].T,x[-3:],rtol=0,atol=0)
        torch.testing.assert_close(x.cpu(),source,rtol=0,atol=0)


if __name__ == '__main__':
    unittest.main()

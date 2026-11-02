"""Known zigzag permutation, temporal unpadding and packed-scale transport."""
import unittest
from types import SimpleNamespace
import torch

from rtp_llm.models_py.modules.hybrid.query_layout import QueryLayout


def fixture(device="cpu", buffers=None):
    # Four ranks own [front pair, back pair] in this independently specified
    # 16-row fixture. Only natural rows0..10 are real; prefix starts at100.
    rankmajor = torch.tensor([0,1,14,15,2,3,12,13,4,5,10,11,6,7,8,9], device=device)
    restore = torch.empty(16, dtype=torch.int64, device=device)
    restore[rankmajor] = torch.arange(16, device=device)
    info = SimpleNamespace(
        prefill_actual_input_lengths_cpu=torch.tensor([11], dtype=torch.int32),
        prefill_prefix_lengths_cpu=torch.tensor([100], dtype=torch.int32),
        prefill_qkv_restore_indice=restore,
        prefill_qkv_padding_mask=torch.arange(16, device=device)<11,
    )
    return QueryLayout(info,4,1,device,buffers=buffers),rankmajor


class QueryLayoutTest(unittest.TestCase):
    def test_natural_attention_identity_is_shared_across_kda_layers(self):
        # KdaPagedPrefill keys the round's cuLA metadata by this object. Each
        # layer must reuse it, while distinct cache groups stay independent.
        layout, _ = fixture()
        first, second = object(), object()
        views = [layout.natural_attention(first) for _ in range(3)]
        self.assertTrue(all(view is views[0] for view in views))
        other = layout.natural_attention(second)
        self.assertIsNot(other, views[0])
        self.assertIs(other.original, second)
        self.assertIs(views[0].cu_seqlens, layout.cu_host)
        self.assertIs(views[0].valid_token_mask, layout.natural_valid)
        next_round, _ = fixture()
        self.assertIsNot(next_round.natural_attention(first), views[0])

    def test_all_projection_branches_restore_together(self):
        layout,rankmajor = fixture()
        fused=rankmajor[:,None]*100+torch.arange(8)[None,:]
        restored=layout.gather_natural(fused)
        expected=torch.arange(11)[:,None]*100+torch.arange(8)[None,:]
        torch.testing.assert_close(restored,expected,rtol=0,atol=0)
        for actual,reference in zip(restored.split([3,2,2,1],-1),expected.split([3,2,2,1],-1)):
            torch.testing.assert_close(actual,reference,rtol=0,atol=0)
        self.assertEqual(layout.local_positions.tolist(),[102,103,-1,-1])
        self.assertEqual(layout.local_valid.tolist(),[True,True,False,False])
        self.assertEqual(layout.cu_host.tolist(),[0,11])

    def test_restore_before_output_projection_and_zero_padding(self):
        layout,rankmajor=fixture()
        values=torch.randn(11,5)
        output,_=layout.scatter_rankmajor(values)
        weight=torch.randn(5,7)
        natural_projected=values@weight
        expected=torch.zeros(16,7)
        expected[layout.natural_to_rankmajor]=natural_projected
        torch.testing.assert_close(output@weight,expected)
        pointer=output.data_ptr()
        second,_=layout.scatter_rankmajor(values+1)
        self.assertEqual(pointer,second.data_ptr())
        self.assertEqual(second[rankmajor>=11].count_nonzero().item(),0)

    def test_shared_prepared_capacity_and_packed_scale_stride(self):
        buffers={};layout,_=fixture(buffers=buffers)
        prepared=layout.reserve("projection",11,16,torch.bfloat16)
        smaller=layout.gather_natural(torch.randn(16,8).bfloat16())
        self.assertEqual(prepared.data_ptr(),smaller.data_ptr())
        self.assertEqual(smaller.stride(),(16,1))
        values=torch.randn(11,128).to(torch.float8_e4m3fn)
        scales=torch.empty_strided((11,1),(1,32),dtype=torch.int32)
        scales[:,0]=torch.arange(11,dtype=torch.int32)+0x7F7F7F7F
        out,sf=layout.scatter_rankmajor(values,scales)
        torch.testing.assert_close(out.view(torch.uint8)[layout.natural_to_rankmajor],values.view(torch.uint8))
        torch.testing.assert_close(sf[layout.natural_to_rankmajor],scales)
        self.assertEqual(sf.stride(0),1)
        self.assertEqual(sf.stride(1)%32,0)
        self.assertTrue((sf[~torch.isin(torch.arange(16),layout.natural_to_rankmajor)]==0x7F7F7F7F).all())


if __name__=="__main__":unittest.main()

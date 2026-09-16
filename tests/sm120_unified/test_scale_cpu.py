"""CPU execution of the carried UE8M0 decode repair; CUDA is never used."""
import unittest
import torch
from test_integration_policy import source_fn

class ScaleCPU(unittest.TestCase):
    def test_size_one_mn_major_scale_view(self):
        fn = source_fn('rtp_llm/models_py/modules/dsv4/utils.py','_decode_ue8m0',{'torch':torch})
        # is_contiguous is true even though the singleton last-dim stride is 32.
        x=torch.full((8,),0x7F7F7F7F,dtype=torch.int32).as_strided((8,1),(1,32))
        self.assertTrue(x.is_contiguous())
        with self.assertRaises(RuntimeError): x.contiguous().view(torch.uint8)
        self.assertTrue(torch.equal(fn(x,4),torch.ones(8,4)))

    def test_regular_shapes_preserve_original_values(self):
        fn=source_fn('rtp_llm/models_py/modules/dsv4/utils.py','_decode_ue8m0',{'torch':torch})
        torch.manual_seed(14)
        for n,g in ((8,4),(2048,8),(64,16),(32,32)):
            raw=torch.randint(100,140,(n,g),dtype=torch.uint8)
            packed=raw.view(torch.int32)
            for scale in (packed,packed.t().contiguous().t()):
                want=(raw.to(torch.int32)-127).float().exp2()
                self.assertTrue(torch.equal(fn(scale,g),want))

if __name__=='__main__':unittest.main(verbosity=2)

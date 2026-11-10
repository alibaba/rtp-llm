import types,unittest
import q36_bounded_clip_candidate as c
class Storage:
 def __init__(self,p):self.p=p
 def data_ptr(self):return self.p
class Grad:
 def __init__(self,ptr,off=0,n=1,contig=True):self.ptr=ptr;self.off=off;self.n=n;self.contig=contig;self.device='cuda:0'
 def numel(self):return self.n
 def is_contiguous(self):return self.contig
 def untyped_storage(self):return Storage(self.ptr)
 def storage_offset(self):return self.off
 def element_size(self):return 2
class Param:
 def __init__(self,g):self.decoupled_grad=g;self.grad=None
class Tests(unittest.TestCase):
 def setUp(self):
  self.calls=[]
  def original(params,max_norm,total_norm,use_decoupled_grad=False):self.calls.append((list(params),max_norm,total_norm,use_decoupled_grad))
  self.cm=types.SimpleNamespace(clip_grad_by_total_norm_fp32=original);self.om=types.SimpleNamespace(clip_grad_by_total_norm_fp32=original)
  c.install(self.om,self.cm);c.install(self.om,self.cm)
 def test_default_preserved(self):
  ps=[Param(Grad(1)) for _ in range(260)];self.om.clip_grad_by_total_norm_fp32(ps,1,2,True)
  self.assertEqual(len(self.calls),1);self.assertEqual(len(self.calls[0][0]),260)
 def test_chunks_same_norm_and_empty(self):
  ps=[Param(Grad(123,off=i)) for i in range(300)]+[Param(None) for _ in range(128)]
  with c.scope():self.om.clip_grad_by_total_norm_fp32(ps,1,2,True)
  self.assertEqual([len(x[0]) for x in self.calls],[128,128,128]);self.assertEqual(sum(p.decoupled_grad is not None for x in self.calls for p in x[0]),300);self.assertTrue(all(x[1:]==(1,2,True) for x in self.calls))
 def test_rejects_alias_noncontiguous_nonfinite(self):
  for ps,n in [([Param(Grad(1,0,2)),Param(Grad(1,1,2))],2),([Param(Grad(1,contig=False))],2),([Param(Grad(1))],float('nan'))]:
   with self.subTest(norm=n),self.assertRaises(ValueError),c.scope():self.om.clip_grad_by_total_norm_fp32(ps,1,n,True)
  self.assertEqual(self.calls,[])
 def test_scope_reset_after_failure(self):
  with self.assertRaises(ValueError),c.scope():self.om.clip_grad_by_total_norm_fp32([Param(Grad(1))],1,float('inf'),True)
  self.om.clip_grad_by_total_norm_fp32([Param(Grad(1))],1,2,True);self.assertEqual(len(self.calls),1)
if __name__=='__main__':unittest.main()

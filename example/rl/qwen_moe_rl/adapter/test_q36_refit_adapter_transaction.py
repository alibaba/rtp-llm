"""Run actual adapter methods with fake transport to exercise transaction failure semantics."""
import ast, asyncio, contextlib, hashlib, json, os, sys, time, unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace as NS
from unittest.mock import patch
P=Path(__file__).parent/'rtp_verl_backend.py'
tree=ast.parse(P.read_text()); cls=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name=='RTPServerAdapter')
methods=[x for x in cls.body if isinstance(x,ast.AsyncFunctionDef) and x.name in ('update_weights','_update_weights_digest_impl')]
for m in methods:m.decorator_list=[]
tree=ast.fix_missing_locations(ast.Module(body=[ast.ClassDef(name='Adapter',bases=[],keywords=[],body=methods,decorator_list=[])],type_ignores=[]))
class Device:
 type='cuda'
 def __init__(self,*_):pass
 def __eq__(self,other):return isinstance(other,Device)
class Tensor:
 def __init__(self,data):self.data=data;self.device=Device();self.dtype='torch.bfloat16';self.shape=(len(data)//2,)
 def detach(self):return self
 def to(self,**_):return self
 def contiguous(self):return self
 def cpu(self):return self
 def view(self,_):return self
 def numpy(self):return self.data
class Workspace:
 instances=[];fail=False
 def __init__(self,_):self.closed=False;self.instances.append(self)
 def hash(self,v,n,d):
  if self.fail:raise ValueError('hash failure')
  d.update(n.encode()+b'\0'+str(tuple(v.shape)).encode()+b'\0'+str(v.dtype).encode()+b'\0'+v.data)
  return hashlib.sha256(v.data).hexdigest(),len(v.data)
 def close(self):self.closed=True
class Control:
 def __init__(self,owner):self.owner=owner;self.pushed=[];self.committed=False;self.closed=False
 async def status(self):return [NS(phase='running',model_version='v0')]
 async def pause(self):pass
 async def begin_update(self,**_):pass
 async def push_weight(self,n,metadata):
  assert len(self.owner._unacknowledged_ipc)==1
  self.pushed.append(n)
  if self.owner.inject=='ack':raise ValueError('ACK failure')
 async def commit_update(self):
  self.committed=True
  return [NS(model_version='v1',global_step=1,received_weights=2,applied_weights=2,tp_rank=0,tp_size=4,phase='paused')]
 async def close(self):self.closed=True
class Server:
 class Ensure:
  async def remote(self,*_):return dict(restarted=False,native_boot='boot',native_incarnation='inc')
 ensure_native_running=Ensure()
records=[]
async def make_control(_):return current.control
ns=dict(torch=NS(device=Device,cuda=NS(current_device=lambda:0),float32='torch.float32',bfloat16='torch.bfloat16',uint8='torch.uint8'),os=os,contextlib=contextlib,time=time,hashlib=hashlib,json=json,_source=lambda:None,ray=NS(get_actor=lambda _:Server()),_new_control=make_control,_audit_root=lambda:Path('/tmp'),_write_json=lambda p,x:records.append(x),preserve_fp32_source=lambda _:False)
exec(compile(tree,str(P),'exec'),ns);Adapter=ns['Adapter']
ipc=ModuleType('rtp_llm.utils.cuda_ipc');ipc.CudaIpcV2=NS(export_tensor=lambda v:{'ptr':id(v)})
binding=ModuleType('q36_source_binding');binding.verify_source=lambda *args,**kwargs:None
sys.modules['q36_source_binding']=binding
digest=ModuleType('q36_refit_bounded_digest_candidate');digest.CudaDigestWorkspace=Workspace
class TransactionTests(unittest.TestCase):
 def setUp(self):
  global current
  records.clear();Workspace.instances.clear();Workspace.fail=False
  current=Adapter();current.inject=None;current.control=Control(current)
  for k,v in dict(is_leader=True,replica_rank=0,tp_rank=0,tp_size=4,global_rank=0,group_ranks=[0,1,2,3],_last_sync_version='v0',_last_sync_step=None,hf_all_sources={'a','b'},tied_embeddings=False).items():setattr(current,k,v)
  current._required_sources=lambda:{'a','b'};current._local_ipc_binding=lambda _: {'ok':True};current._broadcast=lambda x:x;current._any_global_failure=lambda e:bool(e)
  def gather(item):
   peers=[dict(item) for _ in range(4)] if isinstance(item,dict) else [item]*4
   if current.inject=='mismatch' and isinstance(item,dict) and item.get('sha256'):peers[2]['sha256']='mismatch'
   return peers
  current._gather=gather
  async def terminate():current.retained_on_abort=bool(current._unacknowledged_ipc);current._unacknowledged_ipc=[]
  current._terminate_owned_replica=terminate
  self.mods=patch.dict(sys.modules,{'rtp_llm':ModuleType('rtp_llm'),'rtp_llm.utils':ModuleType('rtp_llm.utils'),'rtp_llm.utils.cuda_ipc':ipc,'q36_refit_bounded_digest_candidate':digest});self.mods.start();self.addCleanup(self.mods.stop)
  self.env=patch.dict(os.environ,{'Q36_REFIT_BOUNDED_DIGEST':'1','Q36_RTP_PROFILE_REFIT':'0'});self.env.start();self.addCleanup(self.env.stop)
 def check(self,inject=None,enabled=True):
  if not enabled:os.environ.pop('Q36_REFIT_BOUNDED_DIGEST',None)
  current.inject=inject;Workspace.fail=inject=='hash';values=[('a',Tensor(b'abcd')),('b',Tensor(b'efgh'))];consumed=[]
  def it():
   for pair in values:consumed.append(pair[0]);yield pair
  if inject:
   with self.assertRaises(RuntimeError):asyncio.run(current.update_weights(it(),global_steps=1))
  else:asyncio.run(current.update_weights(it(),global_steps=1))
  self.assertEqual(consumed,['a','b']);self.assertEqual(len(Workspace.instances),int(enabled));self.assertTrue(current.control.closed)
  if enabled:self.assertTrue(Workspace.instances[0].closed)
  if inject:self.assertFalse(current.control.committed);self.assertTrue(current.retained_on_abort)
  else:
   self.assertTrue(current.control.committed);self.assertEqual(current.control.pushed,['a','b']);self.assertFalse(current._unacknowledged_ipc)
   expected=hashlib.sha256()
   for n,v in values:expected.update(n.encode()+b'\0'+str(tuple(v.shape)).encode()+b'\0'+str(v.dtype).encode()+b'\0'+v.data)
   self.assertEqual(records[-1]['full_source_sha256'],expected.hexdigest());self.assertEqual([x['sha256'] for x in records[-1]['tensors']],[hashlib.sha256(v.data).hexdigest() for _,v in values])
 def test_success(self):self.check()
 def test_default_serial_path(self):self.check(enabled=False)
 def test_hash_failure(self):self.check('hash')
 def test_mismatch(self):self.check('mismatch')
 def test_ack_failure(self):self.check('ack')
if __name__=='__main__':unittest.main()

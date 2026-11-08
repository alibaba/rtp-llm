"""Small independent CPU reference comparison; never initializes CUDA."""
import importlib.util
import os,tempfile,threading,time
import torch
from unittest.mock import patch

torch.set_num_threads(1)
torch.set_num_interop_threads(1)
assert not torch.cuda.is_initialized()

def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);mod=importlib.util.module_from_spec(spec)
    import sys
    sys.modules[name]=mod;spec.loader.exec_module(mod);return mod

baseline=load('baseline',os.environ['Q38_BASELINE_SOURCE'])
import qwen38_bf16_optimizer as candidate
from validate_world72_core_evidence import validate_first_reuse_evidence

def create(mod,reuse):
    params=[torch.nn.Parameter(torch.linspace(-.4,.4,15,dtype=torch.bfloat16).reshape(3,5)),
            torch.nn.Parameter(torch.linspace(-.7,.7,11,dtype=torch.bfloat16))]
    for i,p in enumerate(params):p._qwen36_parameter_name='parameter'+str(i)
    store=mod.GradientStore(retained_bytes=0,retained_device='cuda')
    moments=mod.CpuBf16Moments(cpu_budget_bytes=sum(p.numel()*4 for p in params))
    kwargs=dict(gradients=store,moments=moments,lr=.01,weight_decay=.02,chunk_bytes=16,rounding_seed=771)
    if reuse:kwargs['reuse_cpu_spill']=True
    return params,store,moments,mod.Bf16CpuAdamW(params,**kwargs)

def put(params,store,step):
    for i,p in enumerate(params):
        grad=torch.linspace(-.3,.2,p.numel()).reshape(p.shape)+(step*.003+i*.001)
        store.capture_reduced(p,grad)
    store.clip_grad_norm(.12,expected_global_numel=sum(p.numel() for p in params))

def bits(t):return t.detach().contiguous().view(torch.int16)
def equal_state(a,b):
    for pa,pb in zip(a[0],b[0]):
        assert torch.equal(bits(pa),bits(pb))
        sa,sb=a[3].state[pa],b[3].state[pb]
        assert sa['step']==sb['step']
        for key in ('first','second'):
            assert torch.equal(bits(sa['moments'][key]),bits(sb['moments'][key]))

reference=create(baseline,False);actual=create(candidate,True)
for step in range(1,4):
    put(reference[0],reference[1],step);put(actual[0],actual[1],step)
    if step==1:
        plan=actual[3].prepare_reuse_plan();assert plan['reusable_bytes']==sum(p.numel()*4 for p in actual[0]);actual[3].bind_reuse_plan(plan)
        pointers=[actual[1].slots[p].gradient.untyped_storage().data_ptr() for p in actual[0]]
    reference[3].step();actual[3].step();equal_state(reference,actual)
    if step==1:
        assert actual[2].reused_gradient_bytes==plan['reusable_bytes']
        assert all(s.gradient is None for s in actual[1].slots.values())
        for p,ptr in zip(actual[0],pointers):
            handle=actual[3].state[p]['moments'];first,second=handle['first'],handle['second']
            assert first.untyped_storage().data_ptr()==second.untyped_storage().data_ptr()==ptr
            assert first.untyped_storage().nbytes()==p.numel()*4
            assert first.reshape(-1).stride()==second.reshape(-1).stride()==(2,)
            assert first.storage_offset()==0 and second.storage_offset()==1
        try:actual[3].step();raise AssertionError('consumed gradients replayed')
        except RuntimeError:pass
        try:actual[1].capture_reduced(actual[0][0],torch.ones_like(actual[0][0],dtype=torch.float32));raise AssertionError('capture admitted before world commit')
        except RuntimeError:pass
        actual[3].clear_first_reuse_before_world_vote()
        try:actual[1].capture_reduced(actual[0][0],torch.ones_like(actual[0][0],dtype=torch.float32));raise AssertionError('capture admitted while vote pending')
        except RuntimeError:pass
        actual[3].release_first_reuse_after_world_commit()
    reference[3].zero_grad();actual[3].zero_grad()

# A view into a larger buffer or another slot's storage is never eligible.
bad=create(candidate,True);put(bad[0],bad[1],1)
slot=bad[1].slots[bad[0][0]];slot.gradient=torch.ones(30,dtype=torch.float32)[:15].view(3,5)
try:bad[3].prepare_reuse_plan();raise AssertionError('view into larger storage accepted')
except RuntimeError:pass
bad=create(candidate,True);put(bad[0],bad[1],1)
first=bad[1].slots[bad[0][0]];second=bad[1].slots[bad[0][1]]
second.gradient=first.gradient.view(-1)[:11]
try:bad[3].prepare_reuse_plan();raise AssertionError('shared gradient storage accepted')
except RuntimeError:pass

bad=create(candidate,True);put(bad[0],bad[1],1)
grad=bad[1].slots[bad[0][0]].gradient
alias=torch.empty(0).set_(grad.untyped_storage(),0,grad.shape)
assert alias._base is None
try:bad[3].prepare_reuse_plan();raise AssertionError('separate TensorImpl sharing storage accepted')
except RuntimeError:pass
del alias

bad=create(candidate,True);put(bad[0],bad[1],1)
with tempfile.NamedTemporaryFile() as f:
    f.truncate(11*4)
    mapped=torch.from_file(f.name,shared=False,size=11,dtype=torch.float32)
    slot=bad[1].slots[bad[0][1]];slot.gradient=mapped;storage=mapped.untyped_storage()
    slot._q38_captured_storage=(int(storage._cdata),int(storage.data_ptr()))
    try:candidate._require_private_anonymous(storage.data_ptr(),44,candidate._private_anonymous_mappings());raise AssertionError('file mapping passed VMA check')
    except RuntimeError as error:assert 'private anonymous' in str(error),str(error)
    try:bad[3].prepare_reuse_plan();raise AssertionError('file-backed gradient accepted')
    except RuntimeError:pass

# Direct mutation after binding invalidates the actual version-bound plan.
bad=create(candidate,True);put(bad[0],bad[1],1);plan=bad[3].prepare_reuse_plan();bad[3].bind_reuse_plan(plan)
bad[1].slots[bad[0][0]].gradient.add_(1)
try:bad[3].step();raise AssertionError('changed source after bound admission accepted')
except RuntimeError:pass

# Empty shards may be captured or omitted by FSDP; nonempty coverage must
# remain complete, and a zero slot must still match its real parameter.
for capture_zero in (False,True):
    p=torch.nn.Parameter(torch.ones(5,dtype=torch.bfloat16));z=torch.nn.Parameter(torch.empty(0,dtype=torch.bfloat16))
    p._qwen36_parameter_name='nonempty';z._qwen36_parameter_name='empty'
    store=candidate.GradientStore(retained_bytes=0);moments=candidate.CpuBf16Moments(cpu_budget_bytes=20)
    opt=candidate.Bf16CpuAdamW([p,z],gradients=store,moments=moments,reuse_cpu_spill=True,chunk_bytes=8)
    store.capture_reduced(p,torch.ones(5))
    if capture_zero:store.capture_reduced(z,torch.empty(0))
    store.clip_grad_norm(1,expected_global_numel=5)
    plan=opt.prepare_reuse_plan();empty=next(row for row in plan['slots'] if row['tier']=='empty')
    assert empty['slot_present'] is capture_zero and plan['reusable_bytes']==20
    opt.bind_reuse_plan(plan);opt.step();opt.clear_first_reuse_before_world_vote();opt.release_first_reuse_after_world_commit();opt.zero_grad()
    assert moments.bytes==20 and len(moments.handles)==1

bad=create(candidate,True);put(bad[0],bad[1],1);plan=bad[3].prepare_reuse_plan()
z=torch.nn.Parameter(torch.empty(0,dtype=torch.bfloat16));bad[1].slots[z]=candidate.GradientSlot(z,torch.empty(0),'cpu','unknown')
try:bad[3].prepare_reuse_plan();raise AssertionError('unowned empty slot accepted')
except RuntimeError:pass

# A capture that began before binding but is blocked on the same store lock
# must observe frozen after the atomic bind, not slip into the slot set.
bad=create(candidate,True);put(bad[0],bad[1],1);plan=bad[3].prepare_reuse_plan();started=threading.Event();out=[]
def blocked_capture():
    started.set()
    try:bad[1].capture_reduced(bad[0][0],torch.ones_like(bad[0][0],dtype=torch.float32));out.append('unexpected')
    except RuntimeError:out.append('rejected')
with bad[1]._lock:
    worker=threading.Thread(target=blocked_capture);worker.start();assert started.wait(2)
    time.sleep(.02);bad[3].bind_reuse_plan(plan)
worker.join(3);assert not worker.is_alive() and out==['rejected']

# The offline validator consumes real pre-overwrite metadata and the actual
# post-overwrite optimizer audit; it does not pretend consumed FP32 exists.
checked=create(candidate,True);put(checked[0],checked[1],1)
plan=checked[3].prepare_reuse_plan();checked[3].bind_reuse_plan(plan)
pre=dict(schema='q38-first-adam-pre-update-gradient-v1',run_id='run',rank=4,pid=55,world_size=72,recorded_unix=100.,
    generation=plan['generation'],expected_local_numel=26,gradient_store=checked[1].audit(),slot_plan=plan,
    captured_before_any_optimizer_write=True,tensor_payload_included=False)
checked[3].attach_reuse_evidence('a'*64,'fixture');checked[3].step();opt=checked[3].audit()
assert validate_first_reuse_evidence(pre,'a'*64,opt,rank=4,pid=55,loaded_numel=26,update_time_ns=101_000_000_000,expected_run_id='run')==pre['gradient_store']
assert opt['gradient_store']['gradient_numel']==0 and pre['gradient_store']['gradient_numel']==26

# A thrown rounding computation after the first updated chunk must poison the
# optimizer and the store. No retry or rollback is claimed.
failed=create(candidate,True);put(failed[0],failed[1],1);plan=failed[3].prepare_reuse_plan();failed[3].bind_reuse_plan(plan)
original=candidate.stochastic_bf16;calls=[0]
def fail_later(*args,**kwargs):
    calls[0]+=1
    if calls[0]==4:raise FloatingPointError('injected after completed chunk')
    return original(*args,**kwargs)
with patch.object(candidate,'stochastic_bf16',side_effect=fail_later):
    try:failed[3].step();raise AssertionError('injection ignored')
    except FloatingPointError:pass
assert failed[3]._reuse_failed and failed[1]._reuse_failed
try:failed[3].step();raise AssertionError('poisoned optimizer replayed')
except RuntimeError:pass
try:failed[3].zero_grad();raise AssertionError('poisoned store reused')
except RuntimeError:pass
assert not torch.cuda.is_initialized()
print('FIRST_ADAM_REUSE_CPU_PASS: three exact BF16/Adam steps and partial-failure poison; CUDA uninitialized')

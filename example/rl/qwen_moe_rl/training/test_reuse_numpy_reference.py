"""Compare first-step reuse and two later updates with independent NumPy."""
import importlib.util,os,sys
import numpy as np
import torch
from qwen38_bf16_optimizer import Bf16CpuAdamW,CpuBf16Moments,GradientStore

torch.set_num_threads(1)
spec=importlib.util.spec_from_file_location('numpy_ref',os.environ['Q38_NUMPY_REFERENCE_SOURCE'])
ref=importlib.util.module_from_spec(spec);sys.modules['numpy_ref']=ref;spec.loader.exec_module(ref)
name='cpu_reference.policy';seed=20261106
p=torch.nn.Parameter(torch.linspace(-.5,.7,6001).to(torch.bfloat16));p._qwen36_parameter_name=name
store=GradientStore(retained_bytes=0,retained_device='cpu');moments=CpuBf16Moments(cpu_budget_bytes=p.numel()*4)
opt=Bf16CpuAdamW([p],gradients=store,moments=moments,lr=3e-5,betas=(.9,.999),eps=1e-8,
    weight_decay=.01,chunk_bytes=1028,rounding_seed=seed,reuse_cpu_spill=True)
initial=p.detach().float().numpy().copy();expected=(initial,np.zeros_like(initial),np.zeros_like(initial))
for step in range(1,4):
    grad=torch.sin(torch.arange(p.numel(),dtype=torch.float32)*.01+step)
    store.capture_reduced(p,grad*.3);store.capture_reduced(p,grad*.7)
    source=store.slots[p].gradient.numpy().copy()
    store.clip_grad_norm(100,expected_local_numel=p.numel(),expected_global_numel=p.numel())
    expected=ref.numpy_adam(*expected,source,step=step,lr=3e-5,betas=(.9,.999),eps=1e-8,wd=.01,name=name,seed=seed)
    if step==1:
        plan=opt.prepare_reuse_plan();opt.bind_reuse_plan(plan)
    opt.step()
    h=opt.state[p]['moments'];actual=(p.detach().float().numpy(),h['first'].float().numpy(),h['second'].float().numpy())
    mismatches=[int(np.count_nonzero(x.view(np.uint32)!=y.view(np.uint32))) for x,y in zip(actual,expected)]
    assert mismatches==[0,0,0],mismatches
    if step==1:opt.clear_first_reuse_before_world_vote();opt.release_first_reuse_after_world_commit()
    opt.zero_grad()
assert not torch.cuda.is_initialized()
print('REUSE_VS_NUMPY_CPU_PASS: 3 updates, weight/m/v FP32 representations bit-identical, no CUDA')

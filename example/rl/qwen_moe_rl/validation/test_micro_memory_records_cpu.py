"""Synthetic boundary-corruption tests, not actual CUDA gradient qualification."""
import copy,hashlib,json,tempfile
from pathlib import Path
from validate_world72_core_evidence import Validator,GIB


def fixture():
 records=[];loads={};updates={1:{},2:{}}
 for step in (1,2):
  records.append((f'global-minibatch-step{step}.json',dict(trainer_global_step=step,rows_per_rank=2)))
  for rank in range(72):
   loads[rank]=(None,{'loaded_local_numel':100})
   updates[step][rank]=(None,dict(pid=1000+rank,time_ns=10000,optimizer={'gradient_store':{
    'microbatch_contributions_min':2,'microbatch_contributions_max':2}}))
   for local in range(2):
    micro=(step-1)*2+local+1
    store=dict(stored_gradient_dtype='float32',gradient_numel=0 if local==0 else 100,
     gradient_count=0 if local==0 else 1,microbatch_contributions_min=local,microbatch_contributions_max=local,
     retained_budget_bytes=75*GIB,retained_bytes=0 if local==0 else 400,cpu_spilled_bytes=0)
    for event in ('before-cache-release','enter-forward','exit-forward-before-backward'):
     row=dict(schema='q38-actual-engine-micro-memory-boundary-v1',world_size=72,rank=rank,
      optimizer_steps=[] if step==1 else [1],actual_training_micro_index=micro,event=event,pid=1000+rank,
      time_ns=1000+micro,parameter_shadow_grad_count=0,parameter_shadow_grad_bytes=0,
      gradient_store=copy.deepcopy(store),tensors_loss_rewards_gradients_modified=False,
      CPU_cgroup_limit_or_peak_inferred=False,empty_allocator_cache_released=event=='enter-forward')
     records.append((f'micro-memory-rank{rank}-{event}-{micro}.json',row))
 return records,loads,updates


def check(mutate=None):
 records,loads,updates=fixture()
 if mutate:mutate(records,loads,updates)
 with tempfile.TemporaryDirectory(prefix='q38-micro-evidence-fixture-') as folder:
  root=Path(folder)
  for name,value in records:(root/name).write_text(json.dumps(value))
  v=Validator([root]);v.initial_replay=object();v.micro_memory_boundaries(updates,loads)
  return v.errors,v.details


def row(records,**conditions):
 return next(value for name,value in records if all(value.get(k)==v for k,v in conditions.items()))

errors,details=check();assert not errors,errors
assert [r['expected_original_micro_rows_per_rank'] for r in details['micro_memory_boundaries']]==[2,2]
assert [r['global_micro_indices'] for r in details['micro_memory_boundaries']]==[[1,2],[3,4]]
cases={
 'lost previous gradient':lambda r,l,u:row(r,rank=0,actual_training_micro_index=2,event='enter-forward')['gradient_store'].update(gradient_numel=0),
 'wrong contribution count':lambda r,l,u:row(r,rank=0,actual_training_micro_index=2,event='enter-forward')['gradient_store'].update(microbatch_contributions_max=2),
 'allocator hook changed live gradient metadata':lambda r,l,u:row(r,rank=0,actual_training_micro_index=2,event='enter-forward')['gradient_store'].update(retained_bytes=396),
 'BF16 shadow gradients survive':lambda r,l,u:row(r,rank=0,actual_training_micro_index=1,event='enter-forward').update(parameter_shadow_grad_count=1),
 'different actual optimizer process':lambda r,l,u:row(r,rank=0,actual_training_micro_index=1,event='enter-forward').update(pid=9000),
 'CPU RSS relabelled cgroup peak':lambda r,l,u:row(r,rank=0,actual_training_micro_index=1,event='enter-forward').update(CPU_cgroup_limit_or_peak_inferred=True),
 'wrong cache budget':lambda r,l,u:row(r,rank=0,actual_training_micro_index=1,event='enter-forward')['gradient_store'].update(retained_budget_bytes=80*GIB),
 'historical76 cache cannot be relabelled current75':lambda r,l,u:row(r,rank=0,actual_training_micro_index=1,event='enter-forward')['gradient_store'].update(retained_budget_bytes=76*GIB),
 'failed forward retained in accepted run':lambda r,l,u:row(r,rank=0,actual_training_micro_index=1,event='enter-forward').update(event='forward-exception'),
 'optimizer dropped micro contribution':lambda r,l,u:u[1][0][1]['optimizer']['gradient_store'].update(microbatch_contributions_min=1),
 'missing one observed micro boundary':lambda r,l,u:r.pop(next(i for i,(n,v) in enumerate(r) if v.get('rank')==71 and v.get('event')=='enter-forward' and v.get('actual_training_micro_index')==4)),
}
for label,mutate in cases.items():
 errors,_=check(mutate);assert errors,label
print(json.dumps(dict(passed=True,synthetic_boundary_metadata_only=True,actual_GPU_or_RL_qualified=False,
 original_world_ranks=72,steps=2,micro_indices=[1,2,3,4],rejected_corruptions=list(cases))))

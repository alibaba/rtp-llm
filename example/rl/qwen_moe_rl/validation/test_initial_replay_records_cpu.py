"""Synthetic metadata corruption tests; no real model/GPU/optimizer is tested."""
import copy,hashlib,json,tempfile
from pathlib import Path
from initial_replay_contract import InitialReplay,SOURCE0,SCHEMA,layout_digest,digest_json
from validate_world72_core_evidence import Validator

sha=lambda value:hashlib.sha256(value.encode()).hexdigest()
def write(path,value):
 path.write_text(json.dumps(value));return {'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}

def fixture(folder,mutation=None):
 root=Path(folder);current=root/'current';history=root/'history';current.mkdir();history.mkdir()
 old=[];loads={};proofs=[]
 tensors=[dict(name=f'fixture-{index:04d}',sha256=sha(f'source{index}'),source_bytes=2) for index in range(1590)]
 canonical=write(history/'rtp_sync_rank0_step0.json',dict(global_step=0,received_weights=1590,
  ranks=list(range(72)),error=None,full_source_sha256=SOURCE0,model_version='CPU-only-old:1',tensors=tensors))
 for rank in range(72):
  rows=[dict(name=f'p{rank}',canonical_name=f'p{rank}',checkpoint_file='fixture.safetensors',
   local_numel=100,local_bytes=200,source_begin=0,source_end=100,source_reads=1,
   maximum_source_slice_bytes=200,sample_sha256=sha(f'p{rank}'),source_coordinates_verified=True)]
  previous=dict(rank=rank,world_size=72,loaded_local_numel=100,source_slices=rows)
  oldrow=dict(rank=rank,**write(history/f'checkpoint-load-rank{rank}.json',previous),source_layout_sha256=layout_digest(previous))
  old.append(oldrow)
  current_load=copy.deepcopy(previous);current_load['source_slices'][0].update(full_destination_BF16_bytes_verified=True,full_slice_bytes_sha256=[sha(f'full{rank}')])
  load_path=current/f'checkpoint-load-rank{rank}.json';write(load_path,current_load);loads[rank]=(load_path,current_load)
  no_native=dict(native_never_started=True,native_boot=-1,native_incarnation=None,world_size=72,
   checked_native_ranks=[],remaining_owned_gpu_contexts=[],training_admission_safe=False)
  proof=dict(schema='q38-fresh-initial-policy-replay-worker-v1',rank=rank,world_size=72,pid=1000+rank,time_ns=1000,
   current_run_root='/current/formal8',previous_run_root='/historical/full7',gpu_uuid=f'CPU-fixture-uuid-{rank}',
   previous_checkpoint_load_sha256=oldrow['sha256'],current_checkpoint_load_sha256=hashlib.sha256(load_path.read_bytes()).hexdigest(),
   source_layout_sha256=oldrow['source_layout_sha256'],local_numel=100,global_policy_numel=2419804697984,
   current_full_source_copy_verified=True,bounded_slice_digest=digest_json([[sha(f'full{rank}')]]),
   source_policy_full_sha256=SOURCE0,native_never_started_proof=no_native,native_source_export_never_created=True,
   current_optimizer_steps=[],current_CPU_moment_bytes=0,current_FP32_gradient_bytes=0,old_native_version='CPU-only-old:1',
   old_evidence_is_not_current_native_cleanup=True,current_assigned_canonical_sources=[dict(index=index,**tensors[index],
    read_count=1,maximum_host_window_bytes=2) for index in range(rank,1590,72)])
  proofs.append(proof)
 manifest=dict(schema=SCHEMA,global_step=0,world_size=72,initial_full_source_sha256=SOURCE0,
  source_run_root='/historical/full7',source_rows=list(range(18)),canonical_reference=canonical,
  original_native_model_version='CPU-only-old:1',previous_shards=old,episodes=[{'episode_id':f'CPU-fixture-e{i}'} for i in range(72)])
 if mutation:mutation(manifest,proofs)
 manifest_path=root/'manifest.json';manifest_ref=write(manifest_path,manifest)
 for p in proofs:write(current/f'initial-policy-replay-rank{p["rank"]}.json',p)
 world=dict(schema=SCHEMA,manifest_sha256=manifest_ref['sha256'],first_native_sync_skipped_at_step=0,
  current_run_root='/current/formal8',source_run_root='/historical/full7',current_fresh_worker_proofs=proofs,
  historical_native_pids_used_as_current_proof=False,policy1_and_policy2_native_sync_unmodified=True)
 write(current/'initial-policy-replay-world72.json',world)
 write(current/'original-episodes-used.json',dict(schema=SCHEMA,source_run_root='/historical/full7',source_policy_step=0,
  original_native_version='CPU-only-old:1',episode_ids=[e['episode_id'] for e in manifest['episodes']],correct_episodes=15,
  new_generation_or_tool_execution_performed=False,unchanged_original_training_converter_will_run=True,
  original_ids_logprob_bits_rewards_preserved=True))
 validator=Validator([current]);contract=InitialReplay(validator,manifest_path,manifest_ref['sha256'],[history])
 contract.validate(loads)
 return validator,contract,proofs

def check(mutation=None):
 with tempfile.TemporaryDirectory(prefix='q38-replay-metadata-fixture-') as folder:
  validator,contract,proofs=fixture(folder,mutation)
  if not mutation:
   assert contract.valid and not validator.errors,validator.errors
   for rank in range(72):
    row=dict(pid=1000+rank,time_ns=2000,initial_policy_replay_proof=proofs[rank],
     last_native_cleanup_proof=None,last_refit_source_release=None)
    assert contract.permits_first_update(rank,Path('synthetic-only'),row),validator.errors
   bad=copy.deepcopy(row);bad['last_native_cleanup_proof']={'training_admission_safe':True}
   assert not contract.permits_first_update(71,Path('synthetic-only'),bad)
   assert Validator([]).initial_replay is None
   return True
  assert validator.errors and not contract.valid,'A corrupted fresh replay was accepted'
  return True

check()
cases={
 'old native boot cannot replace current never-started':lambda m,p:p[0]['native_never_started_proof'].update(native_boot=0),
 'invented training-safe native proof':lambda m,p:p[0]['native_never_started_proof'].update(training_admission_safe=True),
 'already-updated worker':lambda m,p:p[0].update(current_optimizer_steps=[1]),
 'nonzero old moments':lambda m,p:p[0].update(current_CPU_moment_bytes=2),
 'nonzero previous gradient':lambda m,p:p[0].update(current_FP32_gradient_bytes=4),
 'existing source exporter':lambda m,p:p[0].update(native_source_export_never_created=False),
 'one missing canonical key':lambda m,p:p[0]['current_assigned_canonical_sources'].pop(),
 'changed canonical source':lambda m,p:p[0]['current_assigned_canonical_sources'][0].update(sha256=sha('wrong')),
 'foreign GPU UUID duplicate':lambda m,p:p[1].update(gpu_uuid=p[0]['gpu_uuid']),
 'wrong current load SHA':lambda m,p:p[0].update(current_checkpoint_load_sha256=sha('wrong')),
 'old run relabelled current':lambda m,p:p[0].update(current_run_root='/historical/full7'),
 'wrong original source layout':lambda m,p:p[0].update(source_layout_sha256=sha('wrong')),
 'partial source-copy proof':lambda m,p:p[0].update(current_full_source_copy_verified=False),
 'wrong original canonical COMMIT':lambda m,p:m.update(initial_full_source_sha256=sha('wrong')),
}
for label,mutation in cases.items():check(mutation)
print(json.dumps(dict(passed=True,synthetic_metadata_only=True,real_RL_or_GPU_qualified=False,
 positive_world_ranks=72,current_canonical_keys=1590,rejected_corruptions=list(cases))))

"""Offline lineage for an explicit first-policy replay, never historical PIDs.

Only the first optimizer admission can use this branch. Current checkpoint,
worker, gradient, update2 and cleanup evidence remain current-run inputs.
"""
from __future__ import annotations
import hashlib,json
from pathlib import Path

WORLD=72
SOURCES=1590
POLICY_NUMEL=2419804697984
SOURCE0='1303c36b3d678821c7e9393c10f5a0754fc081143d7f9c7949ad2ed1b45b744f'
SCHEMA='q38-policy0-original-episode-replay-v1'
ORIGINAL_TRAINING_SOURCE_SHA256={
 'verl.workers.engine_workers':'721f866442475a08768854b58176a4dcd99de4564809e3d067dced7cba1122db',
 'verl.utils.device':'52c5dd8dc3e4afc3e0d67756d3d4995282246b98b767be080c28448353c1ad11',
 'verl.utils.memory_utils':'2c356ff43c37d18f29dd0713cb60cb080209fff2e964f993b8d27f176669c111'}


def validate_original_training_postconditions(need,rank,proof):
 """Validate recorded current actor epilogue facts, not an allocator benchmark."""
 p=proof.get('original_training_postconditions') or {}
 if not need(isinstance(p,dict),f'rank{rank} training epilogue receipt must be an object'):return False
 before=p.get('active_allocator_settings_before');after=p.get('active_allocator_settings_after')
 settings=[part for part in after.replace(' ','').split(',') if part.startswith('expandable_segments:')] if isinstance(after,str) else []
 checks=[
  need(p.get('schema')=='q38-original-sync-training-postconditions-v1'
   and p.get('actual_actor_child_class')=='verl.workers.engine_workers.ActorRolloutRefWorker'
   and p.get('source_sha256')==ORIGINAL_TRAINING_SOURCE_SHA256,
   f'rank{rank} epilogue must bind the actual original actor and three pinned verl source files'),
  need(p.get('original_sync_epilogue_replayed') is True
   and p.get('force_synchronized_aggressive_empty_cache_completed') is True
   and type(p.get('base_sync_done_before')) is bool and p.get('base_sync_done_after') is True,
   f'rank{rank} original synchronized cleanup/base-sync epilogue did not complete'),
  need(isinstance(before,str) and bool(before) and settings==['expandable_segments:True']
   and p.get('active_training_expandable_segments') is True and p.get('allocator_backend')=='native'
   and p.get('IPC_allocator_environment')=='backend:native,expandable_segments:False',
   f'rank{rank} active training allocator must be True while the native IPC environment stays False'),
  need(p.get('native_update_weights_method_unmodified') is True
   and p.get('no_native_resume_or_weight_transaction_performed') is True,
   f'rank{rank} initial epilogue cannot substitute an invented native transaction or a changed sync body')]
 return all(checks)


LAYOUT_FIELDS=('name','canonical_name','checkpoint_file','local_numel','local_bytes','source_begin',
 'source_end','source_reads','maximum_source_slice_bytes','sample_sha256','source_coordinates_verified')


def digest_json(value):
 return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def layout_digest(load):
 return digest_json([{key:row[key] for key in LAYOUT_FIELDS} for row in load['source_slices']])


class InitialReplay:
 def __init__(self,validator,path,sha256,historical_roots):
  self.v=validator;self.path=Path(path);self.sha256=sha256
  self.historical_roots=[Path(p).resolve() for p in historical_roots]
  self.valid=False;self.proofs={};self.manifest={};self.canonical=None;self.current_root=None
  self._reference_index=None

 def need(self,value,message):return self.v.need(value,'initial replay: '+message)

 def reference(self,spec):
  """Find immutable original bytes by SHA inside declared history only.

  Runtime /home/admin paths are provenance labels. Never follow those paths
  on the verification host or mix history into current model/optimizer roots.
  """
  if self._reference_index is None:
   self._reference_index={}
   for root in self.historical_roots:
    for p in root.rglob('*'):
     if not p.is_file() or p.suffix not in ('.json','.jsonl') or p.stat().st_size>64*1024**2:continue
     raw=p.read_bytes();sha=hashlib.sha256(raw).hexdigest()
     self._reference_index.setdefault(sha,[]).append(p)
  matches=self._reference_index.get(spec.get('sha256'),[])
  if not self.need(bool(matches),'original artifact bytes are missing for '+str(spec.get('path'))):return None
  p=matches[0];raw=p.read_bytes()
  self.v.inventory[str(p)]=dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
  return p,json.loads(raw)

 def validate(self,loads):
  before=len(self.v.errors);raw=self.path.read_bytes()
  self.v.inventory[str(self.path.resolve())]=dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
  self.need(hashlib.sha256(raw).hexdigest()==self.sha256,'the opt-in manifest SHA does not match')
  self.manifest=m=json.loads(raw)
  self.need(m.get('schema')==SCHEMA and m.get('global_step')==0 and m.get('world_size')==WORLD
    and m.get('initial_full_source_sha256')==SOURCE0,'only the explicit complete initial-policy lineage is supported')
  self.need(bool(self.historical_roots) and all(root not in self.v.roots for root in self.historical_roots),
    'current evidence and original historical roots must be declared separately')
  self.need(len(m.get('episodes',[]))==72 and len({e.get('episode_id') for e in m.get('episodes',[])})==72,
    'the original72 episode identities are incomplete')
  self.need(len(m.get('source_rows',[]))==18 and len(set(m.get('source_rows',[])))==18,
    'the fixed original18 prompt rows are incomplete')
  previous={r.get('rank'):r for r in m.get('previous_shards',[])}
  self.need(len(m.get('previous_shards',[]))==72 and set(previous)==set(range(72)),
    'original checkpoint shard provenance must cover72 ranks')
  reference=self.reference(m['canonical_reference'])
  if reference is None:return False
  canonical_path,canonical=reference;self.canonical=canonical
  tensors=canonical.get('tensors',[])
  self.need(canonical.get('global_step')==0 and canonical.get('received_weights')==SOURCES
    and canonical.get('ranks')==list(range(WORLD)) and canonical.get('error') is None
    and canonical.get('full_source_sha256')==SOURCE0
    and canonical.get('model_version')==m.get('original_native_model_version'),
    'the original canonical transaction is not policy0 COMMIT')
  self.need(len(tensors)==SOURCES and [r.get('name') for r in tensors]==sorted({r.get('name') for r in tensors}),
    'the original canonical source set/order differs')
  proofs=self.v.by_rank(self.v.records('initial-policy-replay-rank*.json'),'current fresh replay admission')
  worlds=self.v.records('initial-policy-replay-world72.json')
  self.need(len(worlds)==1,'one current world72 replay admission is required')
  world=worlds[0][1] if len(worlds)==1 else {}
  self.current_root=world.get('current_run_root')
  self.need(world.get('schema')==SCHEMA and world.get('manifest_sha256')==self.sha256
    and world.get('first_native_sync_skipped_at_step')==0
    and world.get('source_run_root')==m.get('source_run_root')
    and self.current_root and self.current_root!=m.get('source_run_root')
    and world.get('historical_native_pids_used_as_current_proof') is False
    and world.get('policy1_and_policy2_native_sync_unmodified') is True,
    'history must remain policy0 source provenance, never current native/actor cleanup')
  uuids=set();verified_indices=[]
  for rank,(path,p) in proofs.items():
   if rank not in loads or rank not in previous:continue
   load_path,load=loads[rank];old=previous[rank]
   old_reference=self.reference(old)
   self.need(old_reference is not None,'the original saved load receipt is missing')
   if old_reference is not None:
    self.need(layout_digest(old_reference[1])==old.get('source_layout_sha256'),
      f'rank{rank} original layout digest differs from the pinned manifest')
   self.need(p.get('schema')=='q38-fresh-initial-policy-replay-worker-v1'
    and p.get('rank')==rank and p.get('world_size')==72
    and p.get('current_run_root')==self.current_root and p.get('previous_run_root')==m.get('source_run_root')
    and type(p.get('pid')) is int and p['pid']>0 and type(p.get('time_ns')) is int and p['time_ns']>0,
    f'rank{rank} current process/run/world identity differs')
   uuids.add(p.get('gpu_uuid'))
   self.need(p.get('current_checkpoint_load_sha256')==hashlib.sha256(load_path.read_bytes()).hexdigest()
    and p.get('previous_checkpoint_load_sha256')==old.get('sha256')
    and p.get('source_layout_sha256')==old.get('source_layout_sha256')==layout_digest(load)
    and p.get('local_numel')==load.get('loaded_local_numel')
    and p.get('global_policy_numel')==POLICY_NUMEL
    and p.get('current_full_source_copy_verified') is True,
    f'rank{rank} current full BF16 shard does not bind the original policy0 layout')
   slices=load.get('source_slices',[])
   self.need(bool(slices) and all(row.get('full_destination_BF16_bytes_verified') is True
    and len(row.get('full_slice_bytes_sha256',[]))==row.get('source_reads') for row in slices)
    and p.get('bounded_slice_digest')==digest_json([row.get('full_slice_bytes_sha256') for row in slices]),
    f'rank{rank} complete new source-to-master byte-copy checks are absent')
   self.need(p.get('native_source_export_never_created') is True
    and p.get('current_optimizer_steps')==[]
    and type(p.get('current_CPU_moment_bytes')) is int and p['current_CPU_moment_bytes']==0
    and type(p.get('current_FP32_gradient_bytes')) is int and p['current_FP32_gradient_bytes']==0
    and p.get('old_evidence_is_not_current_native_cleanup') is True
    and p.get('source_policy_full_sha256')==SOURCE0
    and p.get('old_native_version')==m.get('original_native_model_version'),
    f'rank{rank} was updated, exported or populated with old optimizer state before replay')
   no_native=p.get('native_never_started_proof') or {}
   self.need(no_native.get('native_never_started') is True and no_native.get('native_boot')==-1
    and no_native.get('native_incarnation') is None and no_native.get('world_size')==72
    and no_native.get('checked_native_ranks')==[] and no_native.get('remaining_owned_gpu_contexts')==[]
    and no_native.get('training_admission_safe') is False,
    f'rank{rank} must preserve current never-started semantics without fabricated native pause ACKs')
   validate_original_training_postconditions(self.need,rank,p)
   assigned=p.get('current_assigned_canonical_sources',[])
   expected=list(range(rank,SOURCES,WORLD))
   self.need([row.get('index') for row in assigned]==expected,
    f'rank{rank} current CPU source-byte assignment is incomplete')
   for row in assigned:
    index=row.get('index')
    if not self.need(type(index) is int and 0<=index<len(tensors),'invalid canonical source index'):continue
    original=tensors[index];verified_indices.append(index)
    self.need(all(row.get(k)==original.get(k) for k in ('name','sha256','source_bytes'))
     and type(row.get('read_count')) is int and row['read_count']>0
     and type(row.get('maximum_host_window_bytes')) is int and 0<row['maximum_host_window_bytes']<=384*1024**2,
     f'rank{rank} complete bounded canonical bytes differ at index{index}')
  self.need(len(uuids)==72 and None not in uuids,'current fresh proofs require72 distinct allocated UUIDs')
  self.need(sorted(verified_indices)==list(range(SOURCES)),
    'current source hashes must cover all1590 original tensors exactly once')
  echoed=world.get('current_fresh_worker_proofs',[])
  self.need(len(echoed)==72 and all(rank in proofs and p==proofs[rank][1] for rank,p in enumerate(echoed)),
    'the actual CPU world handoff must echo all72 exact current-worker records')
  used=self.v.records('original-episodes-used.json')
  self.need(len(used)==1,'one current first-batch replay receipt is required')
  if len(used)==1:
   u=used[0][1]
   self.need(u.get('schema')==SCHEMA and u.get('source_run_root')==m.get('source_run_root')
    and u.get('source_policy_step')==0 and u.get('original_native_version')==m.get('original_native_model_version')
    and u.get('episode_ids')==[e['episode_id'] for e in m['episodes']]
    and u.get('correct_episodes')==15 and u.get('new_generation_or_tool_execution_performed') is False
    and u.get('unchanged_original_training_converter_will_run') is True
    and u.get('original_ids_logprob_bits_rewards_preserved') is True,
    'current replay must use exactly the original delivered IDs/raw logprobs/rewards without new tool execution')
  self.proofs=proofs;self.valid=len(self.v.errors)==before
  self.v.details['initial_policy_replay']=dict(valid=self.valid,current_fresh_ranks=len(proofs),
   current_run_root=self.current_root,historical_source_run_root=m.get('source_run_root'),
   source_policy_full_sha256=SOURCE0,current_canonical_sources_verified=len(verified_indices),
   historical_native_version=m.get('original_native_model_version'),
   historical_PIDs_and_memory_not_current_evidence=True,first_native_sync_skipped_only_at_step0=True,
   initial_native_cleanup_or_source_release_not_fabricated=True,
   current_original_training_epilogue_required=True,
   native_IPC_environment_false_and_active_training_allocator_true_distinguished=True)
  return self.valid

 def permits_first_update(self,rank,path,row):
  proof=self.proofs.get(rank,(None,None))[1]
  return self.need(self.valid and proof is not None and row.get('initial_policy_replay_proof')==proof
   and row.get('pid')==proof.get('pid') and row.get('time_ns',0)>proof.get('time_ns',0)
   and (row.get('last_native_cleanup_proof') is None or row.get('last_native_cleanup_proof') is False)
   and (row.get('last_refit_source_release') is None or row.get('last_refit_source_release') is False),
   f'{path}: first actual optimizer commit does not bind the independent fresh-run admission')

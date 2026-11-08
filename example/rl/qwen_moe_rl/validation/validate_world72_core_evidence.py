"""Offline, fail-closed validation of actual world72 BF16 training/core proofs.

This checks collected JSON from every node. It does not launch models or
upgrade CPU fixtures, nor does it replace the separate original reward/tool/
trace, physical memory, and owned job-exit checks in ACCEPTANCE-PLAN.json.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import struct
import sys

WORLD=72
POLICY_NUMEL=2419804697984
MAIN_SOURCES=1590
CPU_LIMIT=858993459200
GIB=1024**3


def finite(value):return type(value) in (float,int) and math.isfinite(value)
def digest(value):return isinstance(value,str) and len(value)==64 and all(c in '0123456789abcdef' for c in value)


def validate_first_reuse_evidence(pre,raw_sha,opt,*,rank,pid,loaded_numel,update_time_ns,expected_run_id):
    """Validate pre-overwrite FP32 facts and post-overwrite ownership separately."""
    commit=opt.get('first_adam_reuse_commit') or {};link=commit.get('pre_update_evidence') or {}
    store=pre.get('gradient_store') or {};plan=pre.get('slot_plan') or {};post=opt.get('gradient_store') or {}
    if (not isinstance(expected_run_id,str) or not expected_run_id or pre.get('run_id')!=expected_run_id
            or pre.get('schema')!='q38-first-adam-pre-update-gradient-v1' or pre.get('rank')!=rank
            or pre.get('pid')!=pid or pre.get('world_size')!=WORLD or pre.get('captured_before_any_optimizer_write') is not True
            or pre.get('tensor_payload_included') is not False or link.get('pre_update_sha256')!=raw_sha
            or pre.get('generation')!=plan.get('generation') or commit.get('pre_update_generation')!=pre.get('generation')
            or pre.get('expected_local_numel')!=loaded_numel or store.get('gradient_numel')!=loaded_numel
            or not finite(pre.get('recorded_unix')) or type(update_time_ns) is not int or pre['recorded_unix']>=update_time_ns/1e9
            or store.get('all_gradients_clipped') is not True or commit.get('complete') is not True
            or commit.get('consumed_gradients_still_resident_as_fp32') is not False):
        raise ValueError('Missing, stale or inconsistent pre-overwrite gradient evidence')
    slots=plan.get('slots');consumed=commit.get('consumed_slots')
    if not isinstance(slots,list) or not isinstance(consumed,list):raise ValueError('Missing slot metadata')
    if len({row.get('name') for row in slots})!=len(slots):raise ValueError('Duplicate planned slot names')
    if (any(type(row.get('numel')) is not int or row['numel']<0 or row.get('bytes')!=row['numel']*4 for row in slots)
            or sum(row['numel'] for row in slots)!=loaded_numel or plan.get('planned_moment_bytes')!=loaded_numel*4):
        raise ValueError('Pre-update slot elements do not cover the actual shard')
    present=[row for row in slots if row.get('tier')!='empty' or row.get('slot_present') is True]
    contributions=[row.get('contributions') for row in present]
    if (any(type(n) is not int or n<1 for n in contributions) or len(present)!=store.get('gradient_count')
            or min(contributions,default=0)!=store.get('microbatch_contributions_min')
            or max(contributions,default=0)!=store.get('microbatch_contributions_max')
            or store.get('retained_bytes')+store.get('cpu_spilled_bytes')!=loaded_numel*4):
        raise ValueError('Pre-update contribution or physical gradient bytes differ from slot metadata')
    cpu=[row for row in slots if row.get('tier')=='cpu']
    if any(row.get('numel',0)<=0 or row.get('device')!='cpu' or row.get('storage_nbytes')!=row.get('bytes')
            or row.get('storage_offset')!=0 or row.get('storage_use_count')!=2 or row.get('tensor_use_count')!=1
            for row in cpu):raise ValueError('Unverified or nonexclusive CPU source slot')
    if len(consumed)!=len(cpu):raise ValueError('Consumed slot count differs from plan')
    for old,new in zip(cpu,consumed):
        if (old.get('name')!=new.get('name') or old.get('bytes')!=new.get('gradient_bytes')
                or old.get('storage_id')!=new.get('storage_id') or old.get('storage_pointer')!=new.get('storage_pointer')
                or old.get('contributions')!=new.get('contributions') or new.get('all_chunks_read_before_overwrite') is not True
                or new.get('first_dtype')!='bfloat16' or new.get('second_dtype')!='bfloat16'
                or new.get('first_stride')!=[2] or new.get('second_stride')!=[2]):
            raise ValueError('Consumed BF16 state does not bind its original CPU gradient slot')
    reused=plan.get('reusable_bytes')
    if (type(reused) is not int or reused<0 or reused!=sum(row['gradient_bytes'] for row in consumed)
            or reused!=commit.get('consumed_cpu_gradient_bytes') or reused!=store.get('cpu_spilled_bytes')
            or reused!=opt.get('moments',{}).get('cpu_moment_bytes_reusing_gradient')
            or post.get('cpu_spilled_bytes')!=0 or post.get('consumed_gradient_slots')!=len(consumed)
            or post.get('gradient_numel')+reused//4!=store.get('gradient_numel')):
        raise ValueError('Physical reuse byte accounting differs from pre/post evidence')
    return store


def fp32_scores(values,bits):
    if not isinstance(values,list) or not values or not all(finite(v) for v in values):return False
    try:
        raw=struct.pack(f'<{len(values)}f',*values)
        return raw.hex()==bits and list(struct.unpack(f'<{len(values)}f',raw))==values
    except (OverflowError,TypeError,struct.error):return False


class Validator:
    def __init__(self,roots,*,replay_manifest=None,replay_sha256=None,historical_roots=(),logprob_report_only=False,first_adam_reuse_required=False,production_plan=None,production_plan_sha256=None,expected_run_id=None):
        self.roots=[root.resolve() for root in roots]
        self.logprob_report_only=logprob_report_only
        self.first_adam_reuse_required=first_adam_reuse_required
        self.production_plan_path=production_plan
        self.production_plan_sha256=production_plan_sha256
        self.expected_run_id=expected_run_id
        self.validated_plan_run_id=None
        self.validated_plan_reuse_enabled=None
        self.errors=[];self.inventory={};self.details={}
        self.reuse_pre={}
        self.initial_replay=None
        if replay_manifest is not None:
            from initial_replay_contract import InitialReplay
            self.initial_replay=InitialReplay(self,replay_manifest,replay_sha256,historical_roots)
        elif replay_sha256 or historical_roots:
            self.errors.append('An explicit pinned replay manifest is required to use historical evidence')

    def need(self,value,message):
        if not value:self.errors.append(message)
        return bool(value)

    def reuse_run_identity(self):
        """Use the pinned current plan, and cross-check replay when present."""
        replay=self.initial_replay
        replay_id=None
        if replay is not None:
            root=getattr(replay,'current_root',None)
            path=Path(root) if isinstance(root,str) else None
            if (getattr(replay,'valid',False) is True and path is not None
                    and path.is_absolute() and path.parent.name=='runs' and '..' not in path.parts):
                replay_id=path.name
            else:
                return None
        plan_id=self.validated_plan_run_id
        if replay_id is not None and plan_id is not None and replay_id!=plan_id:
            return None
        return plan_id or replay_id

    def records(self,pattern,roots=None):
        values=[];seen={}
        for root in (self.roots if roots is None else roots):
            for path in sorted(root.rglob(pattern)):
                if not path.is_file() or path.stat().st_size>64*1024**2:continue
                raw=path.read_bytes();sha=hashlib.sha256(raw).hexdigest()
                self.inventory[str(path)]=dict(bytes=len(raw),sha256=sha)
                try:record=json.loads(raw)
                except (UnicodeError,ValueError) as error:
                    self.errors.append(f'{path}: invalid JSON: {error}');continue
                identity=(path.name,record.get('rank'),record.get('time_ns'),record.get('global_step'))
                if identity in seen:
                    self.need(seen[identity]==sha,f'{path}: conflicting copies of one original record')
                    continue
                seen[identity]=sha;values.append((path,record))
        return values

    def by_rank(self,values,label):
        result={}
        for path,row in values:
            rank=row.get('rank')
            if not self.need(type(rank) is int and 0<=rank<WORLD,f'{path}: invalid actual rank'):continue
            self.need(rank not in result,f'{label}: more than one distinct original record for rank{rank}')
            self.need(row.get('world_size',WORLD)==WORLD,f'{path}: world size differs from72')
            result[rank]=(path,row)
        self.need(set(result)==set(range(WORLD)),f'{label}: requires every one of72 actual ranks')
        return result

    def source_loads_and_plans(self):
        loads=self.by_rank(self.records('checkpoint-load-rank*.json'),'initial model load')
        owned=0
        for rank,(path,row) in loads.items():
            self.need(row.get('schema')=='qwen38-flat-sharded-checkpoint-v1'
                and row.get('source_keys')==MAIN_SOURCES and row.get('global_policy_numel')==POLICY_NUMEL,
                f'{path}: full pinned main-policy source coverage differs')
            self.need(row.get('all_master_dtype')==row.get('official_source_dtype')=='bfloat16'
                and row.get('rank0_full_model_materialized') is False
                and row.get('buffer_values_and_aliases_verified') is True,
                f'{path}: unique BF16 shard/buffer initialization was not verified')
            slices=row.get('source_slices',[])
            local=sum(item.get('local_numel',0) for item in slices)
            self.need(local==row.get('loaded_local_numel') and type(local) is int and local>0,
                f'{path}: actual loaded shard coordinates do not sum to its local count')
            owned+=local
        self.need(owned==POLICY_NUMEL,'The72 actual BF16 master shards do not cover exactly one main policy')
        plans=self.records('optimizer-placement-plan-rank*.json')
        ranks=[]
        for path,row in plans:
            plan=row.get('rank_plan',{});ranks.append(plan.get('rank'))
            self.need(row.get('disk_offload_forbidden') is True
                and row.get('node_verified_CPU_limit_bytes')==CPU_LIMIT,
                f'{path}: disk-offload policy or actual800GiB memory limit differs')
            for field in ('node_train_CPU_bytes_with_reserve','node_rollout_CPU_bytes_with_reserve'):
                self.need(type(row.get(field)) is int and 0<row[field]<=CPU_LIMIT,
                    f'{path}: {field} exceeds the actual cgroup limit or is absent')
            self.need(plan.get('planned_disk_moment_bytes')==0 and plan.get('fp32_persistent_moment_bytes')==0,
                f'{path}: unexpected persistent FP32/disk optimizer state')
        self.need(len(ranks)==WORLD and set(ranks)==set(range(WORLD)),
            'Actual bounded CPU/memory placement plans are required from all72 ranks')
        self.details['initial_model_load']=dict(ranks=len(loads),global_BF16_master_numel=owned)
        return loads

    def optimizer_updates(self,loads):
        grouped=defaultdict(dict)
        reuse_pre=self.reuse_pre={}
        for prepath,pre in self.records('pre-adam-reuse-rank*.json'):
            rank=pre.get('rank')
            self.need(rank not in reuse_pre,f'{prepath}: duplicate pre-Adam reuse record')
            reuse_pre[rank]=(prepath,pre)
        world_commits={}
        for commitpath,commit in self.records('reuse-world-commit-rank*.json'):
            key=(commit.get('step'),commit.get('rank'))
            self.need(key not in world_commits,f'{commitpath}: duplicate reuse world commit')
            world_commits[key]=(commitpath,commit)
        for path,row in self.records('optimizer-step-rank*.json'):
            opt=row.get('optimizer',{});steps=opt.get('steps',[])
            if not self.need(len(steps)==1 and type(steps[0]) is int and steps[0] in (1,2),f'{path}: expected actual optimizer step1 or2 only'):continue
            step=steps[0];rank=row.get('rank')
            self.need(rank not in grouped[step],f'{path}: duplicate optimizer update for rank{rank}/step{step}')
            grouped[step][rank]=(path,row)
            enabled=opt.get('first_adam_reuse_enabled') is True
            has_commit=opt.get('first_adam_reuse_commit') is not None
            self.need('first_adam_reuse_enabled' not in opt or type(opt['first_adam_reuse_enabled']) is bool,
                f'{path}: reuse mode must be a boolean')
            self.need(enabled==has_commit and (not self.first_adam_reuse_required or enabled)
                and (step!=1 or rank not in reuse_pre or enabled),
                f'{path}: reuse configuration, commit and pre-update evidence disagree')
            if step==1 and self.validated_plan_reuse_enabled is not None:
                self.need(enabled is self.validated_plan_reuse_enabled,
                    f'{path}: actual reuse mode differs from the pinned production plan')
            if opt.get('first_adam_reuse_enabled') is True:
                expected_run_id=self.reuse_run_identity()
                self.need(expected_run_id is not None,f'{path}: validated current-run plan or replay identity is missing or inconsistent')
                marker=world_commits.get((step,rank),(None,{}))[1];votes=marker.get('post_update_votes',[])
                self.need(marker.get('schema')=='q38-reuse-world-update-vote-v1' and marker.get('rank')==rank
                    and expected_run_id is not None and marker.get('run_id')==expected_run_id
                    and marker.get('pid')==row.get('pid') and marker.get('world_size')==WORLD
                    and marker.get('step')==step and type(marker.get('recorded_ns')) is int
                    and marker['recorded_ns']>row.get('time_ns',marker['recorded_ns'])
                    and len(votes)==WORLD and {vote.get('rank') for vote in votes}==set(range(WORLD))
                    and all(vote.get('phase')=='post-update-audit-and-clear' and vote.get('passed') is True for vote in votes),
                    f'{path}: complete world update/audit vote is missing')
            self.need(row.get('schema')=='qwen38-BF16-state-engine-boundary-v1' and row.get('world_size')==WORLD
                and row.get('phase')=='train' and type(row.get('pid')) is int and row['pid']>0,
                f'{path}: optimizer commit was not observed in the real train phase')
            self.need((opt.get('master_dtype'),opt.get('gradient_dtype'),opt.get('moment_dtype'))==('bfloat16','float32','bfloat16')
                and opt.get('persistent_FP32_master_bytes')==opt.get('persistent_FP32_moment_bytes')==opt.get('disk_offload_bytes')==0,
                f'{path}: persistent state precision/storage differs from the requested BF16 path')
            gradients=opt.get('gradient_store',{})
            if step==1 and opt.get('first_adam_reuse_commit') is not None:
                try:
                    prepath,pre=reuse_pre[rank]
                    gradients=validate_first_reuse_evidence(pre,hashlib.sha256(prepath.read_bytes()).hexdigest(),
                        opt,rank=rank,pid=row['pid'],loaded_numel=loads[rank][1]['loaded_local_numel'],update_time_ns=row['time_ns'],expected_run_id=expected_run_id if enabled else None)
                except (KeyError,TypeError,ValueError) as error:
                    self.need(False,f'{path}: invalid first-Adam reuse evidence: {error}')
            self.need(gradients.get('stored_gradient_dtype')=='float32'
                and gradients.get('independent_gradient_storage') is True
                and gradients.get('captured_before_original_parameter_dtype_cast') is True
                and gradients.get('all_gradients_clipped') is True,
                f'{path}: actual FP32 reduced-gradient capture/clip proof is missing')
            if rank in loads:
                self.need(gradients.get('gradient_numel')==loads[rank][1]['loaded_local_numel'],
                    f'{path}: captured gradients do not cover the actual owned policy shard')
            clip=gradients.get('last_clip') or {}
            self.need(finite(clip.get('global_accumulated_gradient_norm')) and clip['global_accumulated_gradient_norm']>0,
                f'{path}: actual global FP32 gradient norm must be finite and nonzero')
            moments=opt.get('moments',{})
            self.need(moments.get('persistent_moment_dtype')=='bfloat16'
                and moments.get('disk_offload_bytes')==moments.get('persistent_FP32_moment_bytes')==0,
                f'{path}: observed Adam moments are not exclusively CPU BF16 memory')
            if step==1 and self.initial_replay is not None:
                self.initial_replay.permits_first_update(rank,path,row)
            else:
                proof=row.get('last_native_cleanup_proof') or {}
                self.need(proof.get('training_admission_safe') is True and proof.get('all_world_pause_acknowledged') is True
                    and proof.get('all_owned_native_processes_exited') is True
                    and proof.get('remaining_owned_gpu_contexts')==[]
                    and proof.get('checked_native_ranks')==list(range(WORLD)),
                    f'{path}: actual full-world native cleanup did not authorize this update')
                release=row.get('last_refit_source_release') or {}
                self.need(release.get('all_source_views_released') is True and release.get('backing_released') is True,
                    f'{path}: a borrowed CUDA refit source survived training restoration')
                transitions=row.get('phase_transition_records',[])
                train=[item for item in transitions if item.get('phase')=='train']
                rollout=[item for item in transitions if item.get('phase')=='rollout']
                if self.need(bool(train) and bool(rollout),f'{path}: actual offload/restore boundary receipts are absent'):
                    self.need(train[-1].get('refit_source_release_proven') is True
                        and train[-1].get('cuda_ipc_collect_after_native_cleanup') is True,
                        f'{path}: restoration happened without real source/IPC cleanup')
                    before=train[-1].get('cuda_allocated_bytes_before_master_restore')
                    retained=rollout[-1].get('gpu_master_bytes')
                    self.need(type(before) is int and type(retained) is int and before<=retained+4*GIB,
                        f'{path}: unexplained CUDA allocation before restore can conceal a live32GiB refit backing')
        for rank in set(grouped[1]) & set(grouped[2]):
            first=grouped[1][rank][1].get('optimizer',{})
            second=grouped[2][rank][1].get('optimizer',{})
            self.need(first.get('first_adam_reuse_enabled')==second.get('first_adam_reuse_enabled')
                and first.get('first_adam_reuse_commit')==second.get('first_adam_reuse_commit'),
                f'rank{rank}: first-Adam reuse mode or historical commit changed between steps')
        summary=[]
        for step in (1,2):
            ranks=grouped[step]
            self.need(set(ranks)==set(range(WORLD)),f'Update{step}: every one of72 actual optimizer rank commits is required')
            changed=sum(row.get('optimizer',{}).get('changed_weight_elements_last_step',0) for _,row in ranks.values())
            self.need(type(changed) is int and changed>0,f'Update{step}: no actual BF16 policy weight changed')
            summary.append(dict(step=step,ranks=len(ranks),global_changed_BF16_elements=changed))
        self.details['optimizer_updates']=summary
        return grouped

    def policy_loss(self,updates):
        observations=defaultdict(lambda:defaultdict(list))
        for path,row in self.records('policy-loss-rank*.json'):
            rank=row.get('rank');stamp=row.get('time_ns')
            self.need(row.get('schema')=='qwen38-real-policy-loss-v1' and row.get('world_size')==WORLD
                and row.get('cuda_initialized') is True and row.get('loss_requires_grad') is True
                and bool(row.get('loss_grad_fn')),f'{path}: loss is not an observed actual CUDA autograd objective')
            advantage=row.get('advantages') or {};mask=row.get('response_mask') or {}
            try:
                raw=bytes.fromhex(advantage['selected_fp32_bits_hex']);values=list(struct.unpack(f'<{len(raw)//4}f',raw))
                if len(raw)%4:raise ValueError('unaligned FP32 bytes')
            except (KeyError,ValueError,struct.error) as error:
                self.errors.append(f'{path}: observed advantage bits unavailable: {error}');continue
            self.need(hashlib.sha256(raw).hexdigest()==advantage.get('selected_fp32_sha256')
                and len(values)==advantage.get('selected_numel')==mask.get('nonzero')
                and str(advantage.get('device','')).startswith('cuda:') and all(finite(v) for v in values),
                f'{path}: CUDA advantage bytes/mask/checksum disagree')
            later=[(step,pair[1]) for step,ranks in updates.items() if rank in ranks
                   for pair in (ranks[rank],) if type(stamp) is int and pair[1].get('time_ns',0)>stamp]
            if not self.need(bool(later),f'{path}: actual loss does not precede a committed update on the same rank'):continue
            step=min(later,key=lambda item:item[1]['time_ns'])[0]
            observations[step][rank].extend(values)
        for step in (1,2):
            self.need(set(observations[step])==set(range(WORLD)),f'Update{step}: actual policy-loss evidence is missing on one or more ranks')
            values=[value for row in observations[step].values() for value in row]
            self.need(any(v>0 for v in values) and any(v<0 for v in values),
                f'Update{step}: actual GRPO advantages are degenerate')
        self.details['actual_GRPO_advantages']=[dict(step=step,ranks=len(observations[step]),
            positive=sum(v>0 for row in observations[step].values() for v in row),
            negative=sum(v<0 for row in observations[step].values() for v in row)) for step in (1,2)]

    def actual_token_lengths(self):
        records=self.records('training-native-lengths-*.json')
        self.need(len(records)>=2,'Actual original converter length receipts are required for the two real training batches')
        summaries=[]
        for path,record in records:
            self.need(record.get('schema')=='q38-real-native-token-training-lengths-v1'
                and record.get('prompt_bound')==2048 and record.get('total_bound')==4096,
                f'{path}: actual training prompt/total contract differs from2048/4096')
            self.need(record.get('actual_all_native_IDs_preserved') is True
                and record.get('silent_truncation_performed') is False
                and record.get('reward_or_advantage_modified') is False
                and record.get('actual_per_turn_trace_bound_guard_installed') is True
                and digest(record.get('original_converter_sha256')),
                f'{path}: native ID preservation/no-truncation/per-turn observation was not established')
            rows=record.get('rows',[])
            if not self.need(isinstance(rows,list) and bool(rows),f'{path}: no actual converter rows'):continue
            prompt_max=total_max=0
            for index,row in enumerate(rows):
                p=row.get('prompt_tokens');r=row.get('response_tokens_with_observations');t=row.get('total_tokens')
                valid=all(type(value) is int and value>0 for value in (p,r,t))
                if self.need(valid,f'{path}: invalid actual token counts on row{index}'):
                    self.need(p<=2048 and p+r==t and t<=4096,
                        f'{path}: actual native row{index} exceeded the strict prompt/total bounds')
                    prompt_max=max(prompt_max,p);total_max=max(total_max,t)
                self.need(row.get('row')==index and digest(row.get('prompt_sha256')) and digest(row.get('response_sha256')),
                    f'{path}: original row order/native-token digests are absent')
            self.need(record.get('maximum_actual_prompt_tokens')==prompt_max
                and record.get('maximum_actual_total_tokens')==total_max,
                f'{path}: recorded maxima disagree with the actual row counts')
            width=record.get('padded_tensor_width')
            self.need(type(width) is int and width>=total_max
                and record.get('padding_width_is_not_actual_micro_sequence_length') is True,
                f'{path}: padded storage width was confused with actual micro sequence length')
            summaries.append(dict(rows=len(rows),maximum_actual_prompt_tokens=prompt_max,
                maximum_actual_total_tokens=total_max,padded_storage_width=width))
        self.details['actual_training_token_lengths']=dict(conversions=len(records),records=summaries,
            trajectory_count_not_inferred_from_converter_row_count=True,
            policy_step_not_inferred_from_receipt_file_count=True,
            remaining_binding='Validate original72 trajectories/iteration through episode/native traces and actual two optimizer commits separately')

    def global_minibatches(self,updates):
        """Bind the real pre-dispatch metadata to both observed update steps.

        The CPU trainer observes the original backend's dynamic row metadata.
        This receipt does not execute an optimizer, so all72 independent
        optimizer commit receipts remain mandatory for the same explicit step.
        Cross-node wall clocks and the number of converter files are not used
        to infer policy/step identity.
        """
        by_step=defaultdict(list)
        records=self.records('global-minibatch-step*.json')
        for path,row in records:
            step=row.get('trainer_global_step')
            if not self.need(type(step) is int and step in (1,2),
                    f'{path}: pre-dispatch trainer step must be exactly1 or2'):continue
            by_step[step].append((path,row))
            real=row.get('rows_pre_pad');padded=row.get('rows_post_pad')
            valid=type(real) is int and type(padded) is int and 72<=real<=144
            if self.need(valid,f'{path}: actual two-turn72-trajectory row count is outside72..144'):
                self.need(padded==((real+71)//72)*72
                    and row.get('global_minibatch_rows')==padded
                    and row.get('rows_per_rank')==padded//72,
                    f'{path}: neutral dynamic DP padding or one whole global minibatch differs')
            self.need(row.get('schema')=='q38-real-single-global-minibatch-before-dispatch-v1'
                and row.get('rollout_loss_denominator')==row.get('data_parallel_size')==72
                and row.get('ppo_epochs')==row.get('predicted_optimizer_steps_this_iteration')==1
                and row.get('original_backend_metadata_preserved') is True
                and row.get('optimizer_step_executed_by_this_observer') is False,
                f'{path}: original one-minibatch/loss-denominator contract differs')
            losses=row.get('actual_distinct_role_losses')
            self.need(isinstance(losses,list) and len(losses)<=1,
                f'{path}: multiple sequential role-loss updates were scheduled')
            self.need(type(row.get('pid')) is int and row['pid']>0
                and type(row.get('time_ns')) is int and row['time_ns']>0,
                f'{path}: actual CPU trainer process observation is absent')
        summary=[]
        for step in (1,2):
            entries=by_step[step]
            self.need(len(entries)==1,
                f'Global minibatch step{step}: exactly one original pre-dispatch metadata receipt is required')
            actual=updates.get(step,{})
            self.need(set(actual)==set(range(WORLD)),
                f'Global minibatch step{step}: prediction is not backed by all72 actual optimizer commits')
            if len(entries)==1:
                path,row=entries[0]
                summary.append(dict(step=step,rows_before_neutral_padding=row.get('rows_pre_pad'),
                    global_minibatch_rows=row.get('rows_post_pad'),actual_optimizer_rank_commits=len(actual),
                    observer_does_not_execute_an_optimizer=True))
        self.details['global_minibatches']=dict(steps=summary,
            actual_trajectory_count_requires_separate_original_episode_linkage=True,
            policy_step_taken_from_actual_trainer_state_not_receipt_count=True,
            cross_node_wall_clocks_not_used_to_infer_step_order=True)

    def micro_memory_boundaries(self,updates,loads):
        plan_args=(self.production_plan_path,self.production_plan_sha256,self.expected_run_id)
        if self.initial_replay is None and not any(value is not None for value in plan_args):
            return
        budget=75*GIB
        expected_pid=None
        expected_loaded_end=None
        if any(value is not None for value in plan_args):
            # A locally supplied digest is a caller-pinned input, not proof of
            # where the bytes came from. The separate deployment/source-lock
            # acceptance must authenticate that digest for the actual run.
            budget=None
            try:
                if not all(value is not None for value in plan_args) or not digest(self.production_plan_sha256):
                    raise ValueError('An explicit plan, pinned SHA256 and expected run ID are required together')
                path=Path(self.production_plan_path);raw=path.read_bytes()
                if hashlib.sha256(raw).hexdigest()!=self.production_plan_sha256:
                    raise ValueError('Pinned production plan SHA256 mismatch')
                plan=json.loads(raw);run_id=plan['run_id'];root=plan['run_root'];env=plan['environment']
                if not isinstance(env,dict):raise ValueError('Production plan environment must be a mapping')
                b=plan['gradient_cache_bytes'];env_budget=env['Q38_GPU_GRADIENT_CACHE_BYTES']
                root_path=PurePosixPath(root)
                if (not isinstance(run_id,str) or not run_id or '/' in run_id or run_id in ('.','..')
                    or run_id!=self.expected_run_id or not root_path.is_absolute()
                    or '..' in root_path.parts or root_path.name!=run_id or root_path.parent.name!='runs'
                    or env.get('Q38_RUN_ID')!=run_id or env.get('Q38_RUN_ROOT')!=root
                    or type(b) is not int or b<=0 or not isinstance(env_budget,str) or env_budget!=str(b)):
                    raise ValueError('Pinned plan run identity, root or gradient budget is inconsistent')
                if self.initial_replay is not None:
                    if getattr(self.initial_replay,'valid',False) is not True:
                        raise ValueError('Run-specific budget requires validated current-run replay lineage')
                    worlds=self.records('initial-policy-replay-world72.json')
                    if len(worlds)!=1 or worlds[0][1].get('current_run_root')!=root:
                        raise ValueError('Production plan does not bind the current world72 run root')
                    proofs=self.by_rank(self.records('initial-policy-replay-rank*.json'),'budget current-run proof')
                    if (len(proofs)!=WORLD or any(row.get('schema')!='q38-fresh-initial-policy-replay-worker-v1'
                        or row.get('world_size')!=WORLD or row.get('current_run_root')!=root
                        or type(row.get('pid')) is not int or row['pid']<=0 for _,row in proofs.values())):
                        raise ValueError('All72 run-bound replay worker identities are required')
                    expected_pid={rank:row['pid'] for rank,(_,row) in proofs.items()}
                else:
                    if env.get('Q38_INITIAL_POLICY_REPLAY')!='0' or env.get('Q38_REPLAY_VERIFY_FULL_SHARD_BYTES')!='1':
                        raise ValueError('Fresh-run plan must explicitly disable replay and verify every source byte')
                    if set(loads)!=set(range(WORLD)):
                        raise ValueError('Fresh-run budget requires all72 current checkpoint load ranks')
                    rows=[loads[rank][1] for rank in range(WORLD)]
                    if (any(row.get('schema')!='qwen38-flat-sharded-checkpoint-v1'
                            or row.get('rank')!=rank or row.get('world_size')!=WORLD
                            or row.get('run_id')!=run_id or row.get('run_root')!=root
                            or not isinstance(row.get('hostname'),str) or not row['hostname']
                            or type(row.get('pid')) is not int or row['pid']<=0
                            or not finite(row.get('started_unix')) or not finite(row.get('finished_unix'))
                            or row['started_unix']>row['finished_unix']
                            for rank,row in enumerate(rows))
                            or len({(row['hostname'],row['pid']) for row in rows})!=WORLD):
                        raise ValueError('Fresh-run load identities do not cover all72 current rank processes')
                    for row in rows:
                        slices=row.get('source_slices')
                        if (row.get('source_provider')!='platform_mounted' or row.get('runtime_parameters')!=7324
                                or not isinstance(slices,list) or len(slices)!=7324
                                or any(item.get('source_coordinates_verified') is not True
                                    or item.get('full_destination_BF16_bytes_verified') is not True
                                    or type(item.get('local_numel')) is not int or item['local_numel']<0
                                    or item.get('local_bytes')!=item['local_numel']*2
                                    or type(item.get('source_reads')) is not int or item['source_reads']<0
                                    or (item['local_numel']>0 and item['source_reads']==0)
                                    or not isinstance(item.get('full_slice_bytes_sha256'),list)
                                    or len(item['full_slice_bytes_sha256'])!=item['source_reads']
                                    or not all(digest(value) for value in item['full_slice_bytes_sha256'])
                                    for item in slices)):
                            raise ValueError('Fresh-run load is missing complete verified source/destination slice bytes')
                    expected_pid={rank:row['pid'] for rank,row in enumerate(rows)}
                    expected_loaded_end={rank:row['finished_unix'] for rank,row in enumerate(rows)}
                self.inventory[str(path.resolve())]=dict(bytes=len(raw),sha256=self.production_plan_sha256)
                self.details['gradient_cache_budget_source']=dict(kind='caller_pinned_production_plan',
                    run_id=run_id,run_root=root,sha256=self.production_plan_sha256,bytes=b,
                    deployment_lock_must_authenticate_caller_supplied_sha256=True)
                budget=b
            except (OSError,ValueError,TypeError,KeyError,json.JSONDecodeError) as error:
                self.errors.append(f'Invalid run-bound gradient-cache plan: {error}')
        else:
            self.details['gradient_cache_budget_source']=dict(kind='historical_default',bytes=budget)
        configured_budget=budget
        minis=defaultdict(list)
        for path,row in self.records('global-minibatch-step*.json'):
            minis[row.get('trainer_global_step')].append(row)
        expected={}
        for step in (1,2):
            rows=minis[step]
            if self.need(len(rows)==1 and rows[0].get('rows_per_rank') in (1,2),
                    f'Update{step}: actual global minibatch must identify one or two micro rows per rank'):
                expected[step]=rows[0]['rows_per_rank']
        groups=defaultdict(lambda:defaultdict(list));observed=defaultdict(set)
        records=self.records('micro-memory-rank*.json')
        self.need(bool(records),'Dedicated memory profile requires actual per-micro observations')
        for path,row in records:
            steps=row.get('optimizer_steps')
            step=1 if steps==[] else 2 if steps==[1] else None
            rank=row.get('rank');micro=row.get('actual_training_micro_index');event=row.get('event')
            if not self.need(step is not None and type(rank) is int and 0<=rank<WORLD
                    and type(micro) is int and micro>0,
                    f'{path}: actual optimizer/micro/rank boundary identity differs'):continue
            self.need(row.get('schema')=='q38-actual-engine-micro-memory-boundary-v1'
                and row.get('world_size')==72 and row.get('tensors_loss_rewards_gradients_modified') is False
                and row.get('CPU_cgroup_limit_or_peak_inferred') is False,
                f'{path}: live tensor/cgroup boundaries were misrepresented')
            if expected_pid is not None:
                self.need(row.get('pid')==expected_pid.get(rank),
                    f'{path}: micro record does not belong to the plan-bound current rank process')
            if expected_loaded_end is not None:
                self.need(type(row.get('time_ns')) is int and row['time_ns']/1e9>=expected_loaded_end.get(rank,float('inf')),
                    f'{path}: micro record precedes its current rank checkpoint load')
            self.need(event in ('before-cache-release','enter-forward','exit-forward-before-backward'),
                f'{path}: a failed or unknown training-forward boundary remains')
            actual=updates.get(step,{}).get(rank)
            self.need(actual is not None and row.get('pid')==actual[1].get('pid')
                and type(row.get('time_ns')) is int and 0<row['time_ns']<actual[1].get('time_ns',0),
                f'{path}: micro record does not precede the same actual process optimizer commit')
            self.need(row.get('parameter_shadow_grad_count')==row.get('parameter_shadow_grad_bytes')==0,
                f'{path}: a BF16 parameter-shadow gradient survived the original capture hook')
            groups[(step,rank,micro)][event].append((path,row));observed[(step,rank)].add(micro)
        summaries=[]
        offset=0
        for step in (1,2):
            count=expected.get(step)
            if count is None:continue
            step_errors_before=len(self.errors)
            valid_ranks=0;gradient_elements=[];retained=[];spilled=[]
            for rank in range(72):
                micros=observed[(step,rank)]
                if not self.need(micros==set(range(offset+1,offset+count+1)),
                        f'Update{step}/rank{rank}: actual training micro loop is incomplete or duplicated'):continue
                for local_micro,micro in enumerate(sorted(micros)):
                    events=groups[(step,rank,micro)]
                    if not self.need(all(len(events[event])==1 for event in
                            ('before-cache-release','enter-forward','exit-forward-before-backward')),
                            f'Update{step}/rank{rank}/micro{micro}: original forward observations are missing or ambiguous'):continue
                    pre=events['before-cache-release'][0][1];enter=events['enter-forward'][0][1]
                    self.need(pre.get('gradient_store')==enter.get('gradient_store')
                        and pre.get('empty_allocator_cache_released') is False
                        and enter.get('empty_allocator_cache_released') is True,
                        f'Update{step}/rank{rank}/micro{micro}: allocator release changed retained gradient-slot facts')
                    store=enter.get('gradient_store') or {}
                    numel=0 if local_micro==0 else loads.get(rank,(None,{}))[1].get('loaded_local_numel')
                    self.need(store.get('stored_gradient_dtype')=='float32'
                        and store.get('gradient_numel')==numel
                        and store.get('microbatch_contributions_min')==store.get('microbatch_contributions_max')==local_micro,
                        f'Update{step}/rank{rank}/micro{micro}: preceding complete FP32 contribution was dropped or duplicated')
                    budget=store.get('retained_budget_bytes');gpu=store.get('retained_bytes');cpu=store.get('cpu_spilled_bytes')
                    self.need(type(budget) is int and budget==configured_budget
                        and type(gpu) is int and type(cpu) is int and 0<=gpu<=budget and cpu>=0
                        and type(numel) is int and gpu+cpu==numel*4,
                        f'Update{step}/rank{rank}/micro{micro}: FP32 slot bytes or pinned gradient-cache budget differ')
                    if type(numel) is int:gradient_elements.append(numel)
                    if type(gpu) is int:retained.append(gpu)
                    if type(cpu) is int:spilled.append(cpu)
                actual=updates.get(step,{}).get(rank)
                store=actual[1].get('optimizer',{}).get('gradient_store',{}) if actual else {}
                if step==1 and rank in self.reuse_pre and actual and actual[1].get('optimizer',{}).get('first_adam_reuse_commit') is not None:
                    store=self.reuse_pre[rank][1].get('gradient_store',{})
                self.need(store.get('microbatch_contributions_min')==store.get('microbatch_contributions_max')==count,
                    f'Update{step}/rank{rank}: actual optimizer did not consume every original micro contribution exactly once')
                valid_ranks+=1
            summaries.append(dict(step=step,ranks_with_micro_records=valid_ranks,
                expected_original_micro_rows_per_rank=count,global_micro_indices=list(range(offset+1,offset+count+1)),
                enter_gradient_numel_min=min(gradient_elements) if gradient_elements else None,
                enter_gradient_numel_max=max(gradient_elements) if gradient_elements else None,
                maximum_retained_FP32_gradient_bytes=max(retained) if retained else None,
                maximum_CPU_spilled_FP32_gradient_bytes=max(spilled) if spilled else None,
                observed_slot_metadata_and_contribution_counts_preserved_across_empty_cache=(len(self.errors)==step_errors_before),
                raw_gradient_value_bit_hash_not_claimed_by_this_metadata_observer=True))
            offset+=count
        self.details['micro_memory_boundaries']=summaries

    def numerical_policies(self):
        syncs={};probes={};hfs=defaultdict(dict)
        for path,row in self.records('hf-probe-step*-rank*.json'):hfs[row.get('global_step')][row.get('rank')]=(path,row)
        for path,row in self.records('rtp_sync_rank0_step*.json'):syncs[row.get('global_step')]=(path,row)
        for path,row in self.records('rtp_probe_rank0_step*.json'):probes[row.get('global_step')]=(path,row)
        if self.initial_replay is not None and self.initial_replay.valid:
            replay=self.initial_replay;history=replay.historical_roots
            self.need(0 not in syncs and 0 not in probes,
                'Replay policy0 native evidence must remain explicitly historical, not be relabelled current')
            original_syncs=self.records('rtp_sync_rank0_step0.json',roots=history)
            original_probes=self.records('rtp_probe_rank0_step0.json',roots=history)
            original_hfs=self.records('hf-probe-step0-rank*.json',roots=history)
            self.need(len(original_syncs)==len(original_probes)==1,
                'Replay requires the original full7 policy0 transaction and nine-DP report files')
            if len(original_syncs)==1 and len(original_probes)==1:
                old_sync_path,old_sync=original_syncs[0];old_probe_path,old_probe=original_probes[0]
                self.need(hashlib.sha256(old_sync_path.read_bytes()).hexdigest()==replay.manifest['canonical_reference']['sha256']
                    and old_sync['model_version']==replay.manifest['original_native_model_version']
                    and old_probe.get('source_transaction_sha256')==hashlib.sha256(old_sync_path.read_bytes()).hexdigest(),
                    'Historical policy0 raw files do not bind the authenticated replay manifest')
                syncs[0]=original_syncs[0];probes[0]=original_probes[0]
            hfs[0]={row.get('rank'):(path,row) for path,row in original_hfs}
            self.need(len(original_hfs)==len(hfs[0])==72,
                'The original policy0 HF references must cover all72 historical ranks explicitly')
        summary=[];previous=None
        for step in (0,1,2):
            if not self.need(step in syncs and step in probes and set(hfs[step])==set(range(WORLD)),
                    f'Policy{step}: actual full refit,9 native DP probes and72 HF references are required'):continue
            spath,sync=syncs[step];ppath,probe=probes[step]
            self.need(sync.get('received_weights')==MAIN_SOURCES and sync.get('ranks')==list(range(WORLD))
                and sync.get('logical_model_replicas')==1 and digest(sync.get('full_source_sha256')),
                f'{spath}: refit was not a complete one-policy all72 transaction')
            tensors=sync.get('tensors',[])
            self.need(len(tensors)==MAIN_SOURCES and len({row.get('name') for row in tensors})==MAIN_SOURCES
                and all(digest(row.get('sha256')) for row in tensors),f'{spath}: actual canonical per-key source hashes are incomplete')
            self.need(probe.get('model_version')==sync.get('model_version')
                and probe.get('full_source_sha256')==sync.get('full_source_sha256'),
                f'{ppath}: native scores do not bind the actual committed source/version')
            routes=probe.get('probes',[]);statuses=probe.get('statuses',[])
            self.need(len(routes)==9 and {row.get('dp_rank') for row in routes}==set(range(9))
                and len(statuses)==WORLD and {row.get('world_rank') for row in statuses}==set(range(WORLD)),
                f'{ppath}: native probe must cover all9 actual DP endpoints and72 rank ACKs')
            self.need(all(row.get('protocol_version')==3 and row.get('phase')=='running'
                and row.get('global_step')==step and row.get('model_version')==sync.get('model_version')
                and row.get('received_weights')==MAIN_SOURCES and not row.get('update_id') for row in statuses),
                f'{ppath}: actual native statuses disagree about the committed policy')
            self.need(probe.get('HF_comparison_mode')=='report_only'
                and probe.get('HF_error_used_for_acceptance') is False
                and 'HF_reference_tolerance' in probe and probe['HF_reference_tolerance'] is None,
                f'{ppath}: current HF/native report-only profile is not explicit')
            errors=[];native={};reference=hfs[step][0][1].get('raw_log_probs',[])
            target_count=len(probe.get('response_ids',[]))
            for route in routes:
                actual=route.get('raw_log_probs',[])
                if self.need(fp32_scores(actual,route.get('raw_fp32_bits_hex'))
                        and len(actual)==target_count,
                        f'{ppath}: actual native FP32 score values/bits/target lengths differ'):
                    native[route.get('dp_rank')]=route
            for rank,(hpath,hf) in hfs[step].items():
                expected=hf.get('raw_log_probs',[])
                self.need(hf.get('schema')=='q38-real-HF-policy-probe-v1'
                    and hf.get('world_size')==WORLD and hf.get('master_dtype')==hf.get('compute_dtype')=='bfloat16'
                    and hf.get('score_dtype')=='float32'
                    and fp32_scores(expected,hf.get('raw_fp32_bits_hex'))
                    and len(expected)==target_count,
                    f'{hpath}: actual BF16 HF FP32 target score bits differ')
                self.need(hf.get('prompt_ids')==probe.get('prompt_ids') and hf.get('response_ids')==probe.get('response_ids'),
                    f'{hpath}: actual HF/native probe token IDs differ')
                for route in routes:
                    actual=route.get('raw_log_probs',[])
                    if self.need(len(expected)==len(actual),f'{ppath}: target-score lengths differ'):
                        errors.extend(abs(a-b) for a,b in zip(expected,actual) if finite(a) and finite(b))
            maximum=max(errors) if errors else None
            if self.logprob_report_only and not (step==0 and self.initial_replay is not None and self.initial_replay.valid):
                hf_rows=[row for _,row in hfs[step].values()]
                if len(hf_rows)==WORLD and all(fp32_scores(row.get('raw_log_probs',[]),row.get('raw_fp32_bits_hex')) and len(row['raw_log_probs'])==target_count for row in hf_rows):
                    for row in hf_rows:
                        rank_error=max(abs(a-b) for other in hf_rows for a,b in zip(row['raw_log_probs'],other['raw_log_probs']))
                        self.need(row.get('HF_rank_comparison_mode')=='report_only' and row.get('HF_rank_error_used_for_acceptance') is False
                            and row.get('max_HF_rank_error')==rank_error and row.get('all_rank_HF_probe_agreement') is (rank_error<=.05),
                            f'Policy{step}: HF rank error report or explicit profile differs from actual scores')
            self.need(maximum is not None and finite(maximum),
                f'Policy{step}: finite complete HF/native target comparison is absent')
            reports=[];max_dp_error=None;rank0_max=None;rank0_mean=None
            if (set(native)==set(range(9)) and target_count>0
                    and fp32_scores(reference,hfs[step][0][1].get('raw_fp32_bits_hex'))
                    and len(reference)==target_count):
                ordered=[native[dp] for dp in range(9)]
                dp0=ordered[0]['raw_log_probs']
                max_dp_error=max(abs(a-b) for row in ordered for a,b in zip(dp0,row['raw_log_probs']))
                self.need(probe.get('max_dp_error')==max_dp_error,
                    f'Policy{step}: native DP reported maximum differs from actual scores')
                if not self.logprob_report_only:
                    self.need(max_dp_error<=.05, f'Policy{step}: native DP self-agreement exceeds0.05')
                elif not (step==0 and self.initial_replay is not None and self.initial_replay.valid):
                    self.need(probe.get('DP_comparison_mode')=='report_only' and probe.get('DP_error_used_for_acceptance') is False,
                        f'Policy{step}: native DP report-only profile is not explicit')
                rank0_errors=[]
                for route in ordered:
                    absolute=[abs(a-b) for a,b in zip(reference,route['raw_log_probs'])]
                    mean=sum(absolute)/target_count;rank0_errors.extend(absolute)
                    self.need(route.get('absolute_HF_errors')==absolute
                        and finite(route.get('mean_HF_absolute_error')) and route['mean_HF_absolute_error']==mean,
                        f'{ppath}: per-DP reported HF absolute errors or mean differ from actual score values')
                    reports.append(dict(dp_rank=route['dp_rank'],absolute_HF_errors=absolute,
                        mean_HF_absolute_error=mean,HF_raw_log_probs=reference,
                        HF_raw_fp32_bits_hex=hfs[step][0][1]['raw_fp32_bits_hex'],
                        native_raw_log_probs=route['raw_log_probs'],native_raw_fp32_bits_hex=route['raw_fp32_bits_hex']))
                rank0_max=max(rank0_errors);rank0_mean=sum(rank0_errors)/(9*target_count)
                self.need(probe.get('hf_raw_log_probs')==reference
                    and probe.get('hf_raw_fp32_bits_hex')==hfs[step][0][1].get('raw_fp32_bits_hex')
                    and finite(probe.get('max_HF_error')) and probe['max_HF_error']==rank0_max
                    and finite(probe.get('mean_HF_absolute_error')) and probe['mean_HF_absolute_error']==rank0_mean,
                    f'{ppath}: reported HF reference, maximum or mean differs from recomputed complete scores')
                self.need(probe.get('actual_HF_reference_sha256')==hashlib.sha256(hfs[step][0][0].read_bytes()).hexdigest(),
                    f'{ppath}: actual HF reference file digest differs')
            changed=None
            if previous is not None:
                old={row['name']:row['sha256'] for row in previous['tensors']}
                changed=sum(old.get(row['name'])!=row['sha256'] for row in tensors)
                self.need(changed>0 and previous['full_source_sha256']!=sync['full_source_sha256']
                    and previous['model_version']!=sync['model_version'],
                    f'Policy{step}: actual BF16 source bytes and native version did not change after the update')
            summary.append(dict(step=step,evidence_origin=('authenticated_historical_full7_policy0' if step==0 and self.initial_replay is not None and self.initial_replay.valid else 'current_run'),model_version=sync.get('model_version'),HF_ranks=len(hfs[step]),
                native_DP_endpoints=len(routes),recomputed_max_abs_error=maximum,
                recomputed_mean_abs_error=sum(errors)/len(errors) if errors else None,
                HF_comparison_mode='report_only',HF_error_used_for_acceptance=False,
                actual_HF_rank0_recomputed_max_abs_error=rank0_max,
                actual_HF_rank0_recomputed_mean_abs_error=rank0_mean,
                native_DP_self_recomputed_max_abs_error=max_dp_error,DP_error_used_for_acceptance=not self.logprob_report_only,
                per_DP_HF_rank0_comparisons=reports,changed_source_tensors=changed))
            previous=sync
        finals=self.records('rtp_final_policy2_actual_reprobe.json')
        self.need(len(finals)==1 and finals[0][1].get('final_native_GPU_reprobe') is True
            and finals[0][1].get('global_step')==2,'The final actual policy2 GPU re-probe receipt is absent or ambiguous')
        if len(finals)==1 and 2 in probes:
            self.need(all(finals[0][1].get(key)==probes[2][1].get(key) for key in
                ('model_version','full_source_sha256','prompt_ids','response_ids','probes','statuses')),
                'The final actual policy2 re-probe differs from its last recorded full9DP numerical receipt')
        self.details['numerical_policies']=summary

    def validate_reuse_plan_binding(self):
        """Bind the opt-in profile to the caller-pinned, run-specific plan."""
        self.validated_plan_run_id=None
        self.validated_plan_reuse_enabled=None
        args=(self.production_plan_path,self.production_plan_sha256,self.expected_run_id)
        if not any(value is not None for value in args):
            self.need(not self.first_adam_reuse_required,'Required first-Adam reuse needs a pinned current-run plan')
            return
        try:
            if not all(value is not None for value in args) or not digest(self.production_plan_sha256):
                raise ValueError('plan, SHA256 and run ID must be supplied together')
            raw=Path(self.production_plan_path).read_bytes()
            if hashlib.sha256(raw).hexdigest()!=self.production_plan_sha256:
                raise ValueError('pinned plan SHA256 mismatch')
            plan=json.loads(raw);run_id=plan['run_id'];root=plan['run_root'];env=plan['environment']
            if not isinstance(env,dict):raise ValueError('Production plan environment must be a mapping')
            root_path=PurePosixPath(root)
            if (not isinstance(run_id,str) or not run_id or '/' in run_id or run_id in ('.','..')
                    or run_id!=self.expected_run_id or not root_path.is_absolute() or '..' in root_path.parts
                    or root_path.parent.name!='runs' or root_path.name!=run_id
                    or env.get('Q38_RUN_ID')!=run_id or env.get('Q38_RUN_ROOT')!=root):
                raise ValueError('plan identity does not match the current run')
            mode=env.get('Q38_FIRST_ADAM_REUSE_CPU_SPILL','0')
            if mode not in ('0','1') or type(mode) is not str:
                raise ValueError('plan reuse mode must be the string 0 or 1')
            self.validated_plan_run_id=run_id
            required=mode=='1'
            self.validated_plan_reuse_enabled=required
            if required:
                # Enforce evidence mode even when the caller forgot the CLI flag.
                caller_required=self.first_adam_reuse_required
                self.first_adam_reuse_required=True
                self.need(caller_required,'Pinned plan enables first-Adam reuse but the required CLI flag is missing')
            else:
                self.need(not self.first_adam_reuse_required,'Pinned plan disables first-Adam reuse but the CLI requires it')
        except (OSError,ValueError,TypeError,KeyError,json.JSONDecodeError) as error:
            self.need(False,f'Invalid first-Adam reuse plan binding: {error}')

    def run(self):
        self.validate_reuse_plan_binding()
        for label,method in (('load_and_plans',self.source_loads_and_plans),):
            try:loads=method()
            except BaseException as error:self.errors.append(f'{label}: {error!r}');loads={}
        if self.initial_replay is not None:
            try:self.initial_replay.validate(loads)
            except BaseException as error:self.errors.append(f'initial replay: {error!r}')
        try:updates=self.optimizer_updates(loads)
        except BaseException as error:self.errors.append(f'optimizer: {error!r}');updates={}
        for label,method in (('actual_loss',lambda:self.policy_loss(updates)),('actual_token_lengths',self.actual_token_lengths),('global_minibatches',lambda:self.global_minibatches(updates)),('micro_memory_boundaries',lambda:self.micro_memory_boundaries(updates,loads)),('numerical_policies',self.numerical_policies)):
            try:method()
            except BaseException as error:self.errors.append(f'{label}: {error!r}')
        return dict(schema='q38-world72-BF16-core-offline-validation-v5',core_checks_passed=not self.errors,
            full_RL_acceptance=False,errors=self.errors,details=self.details,inputs=self.inventory,
            remaining_mandatory_checks=['Original genuine DeepScaleR correctness rewards and72 trajectories/iteration',
                'Decoded actual Python Tool calls, sandbox replay and exact gateway/native delivered trace linkage',
                'All72 per-GPU peaks and all18 actual host-kernel cgroup800GiB peak/events evidence',
                'Identity-bound final driver/supervisor/Ray/native actual exit and no owned contexts',
                'Pinned deployed native/Python source hashes and actual loaded .so inode identities'])


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence-root',type=Path,action='append',required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--initial-policy-replay-manifest',type=Path)
    parser.add_argument('--initial-policy-replay-sha256')
    parser.add_argument('--historical-evidence-root',type=Path,action='append',default=[])
    parser.add_argument('--logprob-report-only',action='store_true',help='Report native DP score differences without a magnitude gate')
    parser.add_argument('--require-first-adam-reuse',action='store_true',help='Require the explicit first-Adam reuse evidence profile')
    parser.add_argument('--production-plan',type=Path,help='Exact run plan; authenticate its SHA against deployment/source-lock evidence')
    parser.add_argument('--production-plan-sha256',help='Caller-pinned SHA256 of the exact plan bytes')
    parser.add_argument('--expected-run-id',help='Expected current run ID, independently known by the caller')
    args=parser.parse_args();result=Validator(args.evidence_root,replay_manifest=args.initial_policy_replay_manifest,replay_sha256=args.initial_policy_replay_sha256,historical_roots=args.historical_evidence_root,logprob_report_only=args.logprob_report_only,first_adam_reuse_required=args.require_first_adam_reuse,production_plan=args.production_plan,production_plan_sha256=args.production_plan_sha256,expected_run_id=args.expected_run_id).run()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(core_checks_passed=result['core_checks_passed'],full_RL_acceptance=False,
                         error_count=len(result['errors']),output=str(args.output))))
    sys.exit(0 if result['core_checks_passed'] else 2)

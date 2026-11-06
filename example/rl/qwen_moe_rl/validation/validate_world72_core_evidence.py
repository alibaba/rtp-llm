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
from pathlib import Path
import struct
import sys

WORLD=72
POLICY_NUMEL=2419804697984
MAIN_SOURCES=1590
CPU_LIMIT=858993459200
GIB=1024**3


def finite(value):return type(value) in (float,int) and math.isfinite(value)
def digest(value):return isinstance(value,str) and len(value)==64 and all(c in '0123456789abcdef' for c in value)


class Validator:
    def __init__(self,roots):
        self.roots=[root.resolve() for root in roots]
        self.errors=[];self.inventory={};self.details={}

    def need(self,value,message):
        if not value:self.errors.append(message)
        return bool(value)

    def records(self,pattern):
        values=[];seen={}
        for root in self.roots:
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
        for path,row in self.records('optimizer-step-rank*.json'):
            opt=row.get('optimizer',{});steps=opt.get('steps',[])
            if not self.need(len(steps)==1 and steps[0] in (1,2),f'{path}: expected actual optimizer step1 or2 only'):continue
            step=steps[0];rank=row.get('rank')
            self.need(rank not in grouped[step],f'{path}: duplicate optimizer update for rank{rank}/step{step}')
            grouped[step][rank]=(path,row)
            self.need(row.get('schema')=='qwen38-BF16-state-engine-boundary-v1' and row.get('world_size')==WORLD
                and row.get('phase')=='train' and type(row.get('pid')) is int and row['pid']>0,
                f'{path}: optimizer commit was not observed in the real train phase')
            self.need((opt.get('master_dtype'),opt.get('gradient_dtype'),opt.get('moment_dtype'))==('bfloat16','float32','bfloat16')
                and opt.get('persistent_FP32_master_bytes')==opt.get('persistent_FP32_moment_bytes')==opt.get('disk_offload_bytes')==0,
                f'{path}: persistent state precision/storage differs from the requested BF16 path')
            gradients=opt.get('gradient_store',{})
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

    def numerical_policies(self):
        syncs={};probes={};hfs=defaultdict(dict)
        for path,row in self.records('hf-probe-step*-rank*.json'):hfs[row.get('global_step')][row.get('rank')]=(path,row)
        for path,row in self.records('rtp_sync_rank0_step*.json'):syncs[row.get('global_step')]=(path,row)
        for path,row in self.records('rtp_probe_rank0_step*.json'):probes[row.get('global_step')]=(path,row)
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
            errors=[]
            for rank,(hpath,hf) in hfs[step].items():
                expected=hf.get('raw_log_probs',[])
                self.need(hf.get('schema')=='q38-real-HF-policy-probe-v1'
                    and hf.get('world_size')==WORLD and hf.get('master_dtype')==hf.get('compute_dtype')=='bfloat16'
                    and hf.get('score_dtype')=='float32' and expected
                    and all(finite(value) for value in expected)
                    and hf.get('raw_fp32_bits_hex')==struct.pack(f'<{len(expected)}f',*expected).hex(),
                    f'{hpath}: actual BF16 HF FP32 target score bits differ')
                self.need(hf.get('prompt_ids')==probe.get('prompt_ids') and hf.get('response_ids')==probe.get('response_ids'),
                    f'{hpath}: actual HF/native probe token IDs differ')
                for route in routes:
                    actual=route.get('raw_log_probs',[])
                    self.need(all(finite(value) for value in actual)
                        and route.get('raw_fp32_bits_hex')==struct.pack(f'<{len(actual)}f',*actual).hex(),
                        f'{ppath}: actual native FP32 score bits are nonfinite or inconsistent')
                    if self.need(len(expected)==len(actual),f'{ppath}: target-score lengths differ'):
                        errors.extend(abs(a-b) for a,b in zip(expected,actual) if finite(a) and finite(b))
            maximum=max(errors) if errors else None
            self.need(maximum is not None and maximum<=.05,f'Policy{step}: recomputed maximum actual HF/native target error exceeds0.05')
            changed=None
            if previous is not None:
                old={row['name']:row['sha256'] for row in previous['tensors']}
                changed=sum(old.get(row['name'])!=row['sha256'] for row in tensors)
                self.need(changed>0 and previous['full_source_sha256']!=sync['full_source_sha256']
                    and previous['model_version']!=sync['model_version'],
                    f'Policy{step}: actual BF16 source bytes and native version did not change after the update')
            summary.append(dict(step=step,model_version=sync.get('model_version'),HF_ranks=len(hfs[step]),
                native_DP_endpoints=len(routes),recomputed_max_abs_error=maximum,changed_source_tensors=changed))
            previous=sync
        finals=self.records('rtp_final_policy2_actual_reprobe.json')
        self.need(len(finals)==1 and finals[0][1].get('final_native_GPU_reprobe') is True
            and finals[0][1].get('global_step')==2,'The final actual policy2 GPU re-probe receipt is absent or ambiguous')
        if len(finals)==1 and 2 in probes:
            self.need(all(finals[0][1].get(key)==probes[2][1].get(key) for key in
                ('model_version','full_source_sha256','prompt_ids','response_ids','probes','statuses')),
                'The final actual policy2 re-probe differs from its last recorded full9DP numerical receipt')
        self.details['numerical_policies']=summary

    def run(self):
        for label,method in (('load_and_plans',self.source_loads_and_plans),):
            try:loads=method()
            except BaseException as error:self.errors.append(f'{label}: {error!r}');loads={}
        try:updates=self.optimizer_updates(loads)
        except BaseException as error:self.errors.append(f'optimizer: {error!r}');updates={}
        for label,method in (('actual_loss',lambda:self.policy_loss(updates)),('actual_token_lengths',self.actual_token_lengths),('numerical_policies',self.numerical_policies)):
            try:method()
            except BaseException as error:self.errors.append(f'{label}: {error!r}')
        return dict(schema='q38-world72-BF16-core-offline-validation-v2',core_checks_passed=not self.errors,
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
    args=parser.parse_args();result=Validator(args.evidence_root).run()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(core_checks_passed=result['core_checks_passed'],full_RL_acceptance=False,
                         error_count=len(result['errors']),output=str(args.output))))
    sys.exit(0 if result['core_checks_passed'] else 2)

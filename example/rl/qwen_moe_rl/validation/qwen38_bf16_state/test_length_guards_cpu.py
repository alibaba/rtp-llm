"""Actual converter CPU boundary fixtures; no agent, tools, reward or GPU run."""
import argparse,json,os,time
from types import SimpleNamespace
from pathlib import Path
os.environ.update(CUDA_VISIBLE_DEVICES='',Q38_TRAIN_MAX_PROMPT_TOKENS='2048',Q38_TRAIN_MAX_TOTAL_TOKENS='4096')
import torch
from rllm.trainer.verl.dataclass import AccumulatedData,ProcessedStepData
import rllm.trainer.verl.transform as transform
from qwen38_length_guards import install


def fixture(prompt,response):
    data=AccumulatedData()
    item=ProcessedStepData(prompt=prompt,response=response,mask=torch.ones_like(response),
        step_reward=0.,step_id='CPU-boundary-only',multi_modal_inputs={},advantage=None,logprobs=[0.]*response.numel(),routing_matrices=None)
    data.add_step(step_data=item,trajectory_id='CPU-boundary-only',traj_reward=0.,step_num=1,is_last=True,group_role='CPU-fixture')
    data.episode_ids=['CPU-fixture'];data.is_correct=[False];data.termination_reasons=[__import__('rllm.workflows.workflow',fromlist=['TerminationReason']).TerminationReason.UNKNOWN]
    data.metrics=[{}];data.repeat_counts=[1]
    return data


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);args=p.parse_args()
    # Prove the existing underlying primitive would silently remove a token.
    oversize=torch.arange(1,2050)
    old=transform._pad_sequence_batch([oversize],0,2048,left_pad=True)
    if old[0,0].item()!=2 or old.shape!=(1,2048):raise RuntimeError('Expected actual original silent-truncate behavior differs')
    install();passed=[]
    for p_len,r_len in [(2048,2048),(673,3000),(508,1024)]:
        result=transform._batch_tensors_and_build_data_proto(fixture(torch.arange(1,p_len+1),torch.arange(30000,30000+r_len)),0,2048,4096)
        record=result.meta_info['q38_strict_native_token_lengths']
        if record['maximum_actual_total_tokens']!=p_len+r_len or not record['actual_all_native_IDs_preserved']:
            raise RuntimeError('Actual converter did not preserve the valid fixture')
        passed.append(dict(prompt=p_len,response=r_len,actual_total=p_len+r_len,padded_width=record['padded_tensor_width']))
    rejected=[]
    for p_len,r_len in [(2049,1),(2048,2049),(673,3424)]:
        try:transform._batch_tensors_and_build_data_proto(fixture(torch.arange(1,p_len+1),torch.arange(30000,30000+r_len)),0,2048,4096)
        except RuntimeError as error:
            if 'refuses silent token truncation' not in str(error):raise
            rejected.append(dict(prompt=p_len,response=r_len,error=str(error)))
        else:raise RuntimeError('A real oversized row escaped before original truncation')
    # Exercise the actual cumulative multi-turn merge: observations are kept
    # and masked as observations, rather than counted as generated actions.
    prompt=list(range(1,674));a0=list(range(30000,31024));obs=list(range(40000,40150));a1=list(range(50000,51024))
    def step(p,a):return SimpleNamespace(model_output=SimpleNamespace(prompt_ids=p,completion_ids=a,
        logprobs=[-.1]*len(a),multi_modal_inputs={}),routing_matrices=None)
    trajectory=SimpleNamespace(name='CPU-boundary',uid='CPU-boundary',reward=None,
        steps=[step(prompt,a0),step(prompt+a0+obs,a1)])
    accumulated=AccumulatedData()
    emitted=transform._process_trajectory(trajectory,'CPU-fixture',accumulated)
    if emitted!=1:raise RuntimeError('Actual cumulative fixture failed to merge')
    accumulated.episode_ids=['CPU-fixture'];accumulated.is_correct=[False]
    accumulated.termination_reasons=[__import__('rllm.workflows.workflow',fromlist=['TerminationReason']).TerminationReason.UNKNOWN]
    accumulated.metrics=[{}];accumulated.repeat_counts=[1]
    converted=transform._batch_tensors_and_build_data_proto(accumulated,0,2048,4096)
    if converted.batch['response_mask'][0,:2198].tolist()!=[1]*1024+[0]*150+[1]*1024:
        raise RuntimeError('Actual tool observation/action masks changed')
    merged=converted.meta_info['q38_strict_native_token_lengths']
    bad=SimpleNamespace(name='CPU-boundary',uid='CPU-boundary',reward=None,
        steps=[step(list(range(1,2050)),[30000])])
    try:transform._process_trajectory(bad,'CPU-fixture',AccumulatedData())
    except RuntimeError as error:
        if 'oversized actual native trace step' not in str(error):raise
    else:raise RuntimeError('An oversized per-turn prompt was not rejected')
    dropped=fixture(torch.arange(1,9),torch.arange(30000,30010));dropped.repeat_counts=[0,1]
    try:transform._batch_tensors_and_build_data_proto(dropped,0,2048,4096)
    except RuntimeError as error:
        if 'silently dropped episode' not in str(error):raise
    else:raise RuntimeError('A silently dropped episode escaped the guard')
    report=dict(schema='q38-actual-converter-strict-length-CPU-v1',passed=True,
        original_silent_prompt_truncation_reproduced=True,valid_boundaries=passed,oversized_rows_rejected_before_padding=rejected,
        no_original_reward_or_advantage_changes=True,all_data_synthetic_CPU_boundary_fixtures=True,
        actual_tool_calls_or_RL_rewards_generated=False,CUDA_initialized=torch.cuda.is_initialized(),
        actual_cumulative_merge_preserved=True,actual_merged_total=merged['maximum_actual_total_tokens'],
        actual_tool_observation_tokens_preserved=150,per_turn_2049_rejected=True,empty_episode_drop_rejected=True,
        official_model_weights_opened=False,time=time.time())
    Path(args.output).write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)


if __name__=='__main__':main()

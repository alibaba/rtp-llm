"""Reject oversized native-token training rows before rLLM can truncate them.

This observer adds no tokens, tool calls, rewards, advantages, or gradients.
The original conversion/padding and original per-trajectory loss stay intact.
"""
from __future__ import annotations
import functools,hashlib,inspect,json,os,time
from pathlib import Path


def install():
    requested=os.environ.get('Q38_TRAIN_MAX_PROMPT_TOKENS')
    total=os.environ.get('Q38_TRAIN_MAX_TOTAL_TOKENS')
    if requested is None and total is None:return False
    if requested is None or total is None:raise ValueError('Both explicit Q38 training token bounds are required')
    prompt_bound=int(requested);total_bound=int(total)
    if not 0<prompt_bound<total_bound:raise ValueError('Invalid Q38 training token bounds')
    import rllm.trainer.verl.transform as transform
    original=transform._batch_tensors_and_build_data_proto
    if getattr(original,'_q38_no_silent_truncation',False):return True
    if list(inspect.signature(original).parameters)!=['accumulated','pad_token_id','max_prompt_length','max_response_length','processor']:
        raise RuntimeError('Review the actual rLLM conversion signature before installing strict lengths')
    original_sha=hashlib.sha256(inspect.getsource(original).encode()).hexdigest()
    original_trajectory=transform._process_trajectory

    @functools.wraps(original_trajectory)
    def checked_trajectory(trajectory,task_id,accumulated):
        for index,step in enumerate(trajectory.steps):
            value=step.model_output
            if value is None or value.prompt_ids is None or value.completion_ids is None:
                raise RuntimeError('Q38 refuses an unlinked native-token step instead of silently dropping it')
            p=len(value.prompt_ids);r=len(value.completion_ids)
            if p>prompt_bound or p+r>total_bound:
                raise RuntimeError(f'Q38 refuses an oversized actual native trace step: step={index} '
                    f'prompt={p}/{prompt_bound} total={p+r}/{total_bound}')
        return original_trajectory(trajectory,task_id,accumulated)
    checked_trajectory._q38_no_silent_truncation=True

    @functools.wraps(original)
    def checked(accumulated,pad_token_id,max_prompt_length,max_response_length,processor=None):
        # The restored VerlBackend passes max_prompt+max_response as the
        # second pad width. Its name is misleading: validate actual token
        # totals independently, then keep that existing padding untouched.
        if max_prompt_length!=prompt_bound or max_response_length!=total_bound:
            raise RuntimeError('Actual rLLM conversion bounds differ from the explicit Q38 prompt/total contract')
        if not accumulated.prompts or len(accumulated.prompts)!=len(accumulated.responses):
            raise RuntimeError('No complete real training rows reached the strict length boundary')
        if accumulated.repeat_counts and any(n<1 for n in accumulated.repeat_counts):
            raise RuntimeError('Q38 refuses a silently dropped episode at the strict native-length boundary')
        rows=[]
        for index,(prompt,response) in enumerate(zip(accumulated.prompts,accumulated.responses)):
            p=prompt.numel();r=response.numel()
            if prompt.ndim!=1 or response.ndim!=1:
                raise ValueError('Expected original one-dimensional native token rows')
            if p>prompt_bound or p+r>total_bound:
                raise RuntimeError(f'Q38 refuses silent token truncation: row={index} prompt={p}/{prompt_bound} '
                                   f'response_with_tool_observations={r} total={p+r}/{total_bound}')
            rows.append(dict(row=index,prompt_tokens=p,response_tokens_with_observations=r,total_tokens=p+r,
                prompt_sha256=hashlib.sha256(prompt.contiguous().numpy().tobytes()).hexdigest(),
                response_sha256=hashlib.sha256(response.contiguous().numpy().tobytes()).hexdigest()))
        result=original(accumulated,pad_token_id,max_prompt_length,max_response_length,processor)
        # Actual nonpadding token IDs, not rendered text or approximate lengths.
        batch=result.batch
        for index,(prompt,response) in enumerate(zip(accumulated.prompts,accumulated.responses)):
            observed_prompt=batch['prompts'][index]
            observed_response=batch['responses'][index]
            p=prompt.numel();r=response.numel()
            if observed_prompt[-p:].tolist()!=prompt.tolist() or observed_response[:r].tolist()!=response.tolist():
                raise RuntimeError('Actual rLLM conversion changed native prompt/response token IDs')
            actual_valid=int(batch['attention_mask'][index].sum().item())
            if actual_valid!=p+r:
                raise RuntimeError('Actual converted attention mask omitted real native tokens')
        record=dict(schema='q38-real-native-token-training-lengths-v1',pid=os.getpid(),time_ns=time.time_ns(),
            prompt_bound=prompt_bound,total_bound=total_bound,rows=rows,
            maximum_actual_prompt_tokens=max(x['prompt_tokens'] for x in rows),
            maximum_actual_total_tokens=max(x['total_tokens'] for x in rows),
            original_converter_sha256=original_sha,actual_all_native_IDs_preserved=True,
            silent_truncation_performed=False,reward_or_advantage_modified=False,
            actual_per_turn_trace_bound_guard_installed=True,
            padded_tensor_width=int(batch['input_ids'].shape[1]),
            padding_width_is_not_actual_micro_sequence_length=True)
        result.meta_info['q38_strict_native_token_lengths']=record
        directory=os.environ.get('Q38_TRAINING_AUDIT_DIR')
        if directory:
            directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
            with (directory/f'training-native-lengths-{os.getpid()}-{record["time_ns"]}.json').open('x') as stream:
                json.dump(record,stream);stream.write('\n')
        return result
    checked._q38_no_silent_truncation=True
    transform._process_trajectory=checked_trajectory
    transform._batch_tensors_and_build_data_proto=checked
    return True

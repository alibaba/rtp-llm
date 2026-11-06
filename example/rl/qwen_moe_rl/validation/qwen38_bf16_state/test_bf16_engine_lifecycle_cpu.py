"""Exercise actual restored BaseEngineCtx decisions with CPU-only callbacks."""
import json
import os
from pathlib import Path
from types import SimpleNamespace

os.environ.update(CUDA_VISIBLE_DEVICES='',Q38_ENABLE_BF16_STATE_RUNTIME='1')
import qwen38_bf16_training_runtime
import torch
from verl.workers.engine.base import BaseEngineCtx
from verl.workers.engine.fsdp.transformer_impl import FSDPEngine


def main():
    engine=object.__new__(FSDPEngine)
    engine.model_config=SimpleNamespace(hf_config=SimpleNamespace(model_type='qwen3_5_moe_text',
        num_hidden_layers=92,hidden_size=8192,num_experts=512))
    engine._is_offload_param=False;engine._is_offload_optimizer=False
    engine._q38_phases=SimpleNamespace(phase='rollout')
    engine._q38_native_contexts_released=False
    calls=[]
    def phase_to(device,model,optimizer,grad):
        if str(device)=='cuda':
            if engine._q38_native_contexts_released is not True:
                raise RuntimeError('CPU fixture: no real native cleanup proof')
            assert model is True and optimizer is False and grad is False
            engine._q38_phases.phase='train'
        else:
            assert model is False and optimizer is False and grad is False
        calls.append(dict(device=str(device),model=model,optimizer=optimizer,grad=grad))
    engine.to=phase_to
    context=BaseEngineCtx(engine,'eval')
    try:context._context_switch('cuda')
    except RuntimeError:pass
    else:raise AssertionError('Generic offload flags must not bypass the custom restoration proof')
    engine._q38_native_contexts_released=True
    context._context_switch('cuda')
    assert engine._q38_phases.phase=='train' and len(calls)==1
    context._context_switch('cpu')
    assert engine._q38_phases.phase=='train'
    engine._q38_phases.phase='rollout'
    disabled=BaseEngineCtx(engine,'train',disable_auto_offload=True)
    disabled._context_switch('cuda')
    assert engine._q38_phases.phase=='rollout' and len(calls)==2
    assert not torch.cuda.is_initialized()
    result=dict(schema='q38-actual-verl-context-lifecycle-CPU-v1',passed=True,
        generic_offload_flags=[False,False],actual_restored_BaseEngineCtx_used=True,
        missing_native_proof_rejected=True,custom_restore_requested_before_eval=True,
        generic_CPU_exit_does_not_offload_all_weights=True,inner_disable_auto_offload_preserved=True,
        callback_calls=calls,CUDA_initialized=False,actual_native_or_model_task_started=False)
    Path('engine-lifecycle-CPU-result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__=='__main__':main()

"""Control-only test: no tensors/Adam state/model/GPU/Ray allocation."""
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
from qwen38_bf16_training_runtime import _optimizer_step_after_global_clip


def main():
    os.environ['RLLM_REQUIRE_REAL_UPDATE']='1'
    rejects=0
    for norm in (0.,-1.,float('nan'),float('inf'),-float('inf')):
        events=[]
        engine=SimpleNamespace(optimizer=SimpleNamespace(step=lambda:events.append('Adam'),changed_weight_elements=1))
        try:_optimizer_step_after_global_clip(engine,norm,lambda n:events.append('SUM'))
        except RuntimeError:rejects+=1
        else:raise AssertionError('Degenerate gradients must fail before Adam')
        assert events==[]
    events=[]
    engine=SimpleNamespace(optimizer=SimpleNamespace(step=lambda:events.append('Adam'),changed_weight_elements=7))
    def positive_sum(n):events.append(('all72_SUM',n));return 72*n
    _optimizer_step_after_global_clip(engine,1.25,positive_sum)
    assert events==['Adam',('all72_SUM',7)]
    assert engine._q38_last_real_update_gate['global_changed_weight_elements']==504
    events=[];engine.optimizer.changed_weight_elements=0
    try:_optimizer_step_after_global_clip(engine,1.25,positive_sum)
    except RuntimeError:pass
    else:raise AssertionError('No changed BF16 weights must fail before success audit')
    assert events==['Adam',('all72_SUM',0)]
    os.environ['RLLM_REQUIRE_REAL_UPDATE']='0'
    events=[]
    _optimizer_step_after_global_clip(engine,0.,lambda n:events.append('unexpected_SUM'))
    assert events==['Adam'] and engine._q38_last_real_update_gate['enabled'] is False
    assert 'torch' not in sys.modules and 'ray' not in sys.modules
    result=dict(schema='q38-real-update-gate-CPU-control-v1',passed=True,
        pre_Adam_bad_norms_rejected=rejects,zero_norm_did_not_call_Adam=True,
        valid_call_order=['Adam','all72 SUM','success audit permitted'],
        zero_changed_count_refuses_success_audit=True,flag_disabled_preserves_original_call=True,
        optimizer_numerical_algorithm_changed=False,actual_distributed_sum_tested=False,
        Torch_or_Ray_imported=False)
    Path('real-update-gate-CPU-result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__=='__main__':main()

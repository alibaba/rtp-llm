"""A real >0.05 score fixture must report honestly, with structural gates intact."""
import copy
import json
import os
from pathlib import Path
import struct

import qwen38_final_policy as module


def proof_fixture(*, dp_mismatch=False, report_only=True):
    hf = [-1.5, -3.25]
    values = [[-1.625, -3.] for _ in range(9)]
    if dp_mismatch: values[-1] = [-2., -3.]
    errors = [[abs(a-b) for a,b in zip(hf,row)] for row in values]
    return dict(final_native_GPU_reprobe=True,global_step=2,logical_model_replicas=1,
        world_size=72,tp_size=8,dp_size=9,ep_size=72,model_version='test-only-v2',native_incarnation='test-only',
        full_source_sha256='1'*64,prompt_ids=[4,6],response_ids=[7,8],hf_raw_log_probs=hf,
        hf_raw_fp32_bits_hex=struct.pack('<2f',*hf).hex(),max_HF_error=max(v for row in errors for v in row),
        mean_HF_absolute_error=sum(v for row in errors for v in row)/18,
        max_dp_error=max(abs(a-b) for row in values for a,b in zip(values[0],row)),
        HF_comparison_mode='report_only' if report_only else 'strict',HF_error_used_for_acceptance=not report_only,
        statuses=[dict(world_rank=r,global_step=2,model_version='test-only-v2',phase='running',
            received_weights=1590,applied_weights=1,update_id=None) for r in range(72)],
        probes=[dict(dp_rank=dp,generation_world_rank=dp*8,raw_log_probs=row,
            raw_fp32_bits_hex=struct.pack('<2f',*row).hex(),absolute_HF_errors=errors[dp],
            mean_HF_absolute_error=sum(errors[dp])/2) for dp,row in enumerate(values)])


def rejected(proof):
    try: module.validate_final_policy_proof(proof)
    except RuntimeError: return
    raise AssertionError('An incomplete, stale or forged numerical report was accepted')


def run():
    os.environ['Q38_HF_RTP_REPORT_ONLY']='1'
    actual = proof_fixture()
    assert actual['max_HF_error']==.25 and actual['mean_HF_absolute_error']==.1875
    assert module.validate_final_policy_proof(actual) is actual
    mutations = [
        lambda p:p.update(max_HF_error=0.),
        lambda p:p.update(mean_HF_absolute_error=0.),
        lambda p:p.update(HF_error_used_for_acceptance=True),
        lambda p:p.update(HF_comparison_mode='strict'),
        lambda p:p.update(hf_raw_fp32_bits_hex='00'),
        lambda p:p.update(global_step=1),
        lambda p:p.update(final_native_GPU_reprobe=False),
        lambda p:p['statuses'][7].update(received_weights=1589),
        lambda p:p['statuses'][8].update(world_rank=7),
        lambda p:p['probes'][1].update(dp_rank=0),
        lambda p:p['probes'][0].update(raw_fp32_bits_hex='00'),
        lambda p:p['probes'][0].update(absolute_HF_errors=[0.,0.]),
        lambda p:p['probes'][0].update(mean_HF_absolute_error=0.),
        lambda p:p.update(full_source_sha256='bad'),
        lambda p:p.update(hf_raw_log_probs=[float('nan'),-3.25]),
    ]
    for mutate in mutations:
        changed=copy.deepcopy(actual);mutate(changed);rejected(changed)
    # Even an accurately reported large DP discrepancy remains a failure.
    rejected(proof_fixture(dp_mismatch=True))
    os.environ['Q38_LOGPROB_REPORT_ONLY']='1'
    large=proof_fixture(dp_mismatch=True)
    large.update(DP_comparison_mode='report_only',DP_error_used_for_acceptance=False)
    assert module.validate_final_policy_proof(large) is large
    forged=copy.deepcopy(large);forged['max_dp_error']=0.;rejected(forged)
    missing=copy.deepcopy(large);missing.pop('DP_comparison_mode');rejected(missing)
    broken=copy.deepcopy(large);broken['probes'][0]['raw_log_probs'][0]=float('nan');rejected(broken)
    os.environ.pop('Q38_LOGPROB_REPORT_ONLY')
    os.environ.pop('Q38_HF_RTP_REPORT_ONLY')
    rejected(proof_fixture(report_only=False))
    rejected(actual)
    old=dict(proof_fixture(report_only=False))
    old['hf_raw_log_probs']=[-1.625,-3.];old['max_HF_error']=0.
    module.validate_final_policy_proof(old)
    result=dict(passed=True,genuine_large_error_fixture_max=.25,genuine_large_error_fixture_mean=.1875,
        large_HF_difference_is_reported_not_replaced_by_zero=True,invalid_reports_rejected=len(mutations),
        accurately_reported_DP_disagreement_still_rejected=True,default_strict_behavior_preserved=True,
        no_Torch_Ray_GPU_or_model_initialized=True)
    return result


if __name__=='__main__':
    result=run();print('HF_REPORT_ONLY_CPU_PASS',json.dumps(result),flush=True)
    Path('HF-REPORT-ONLY-CPU-FIXTURE.json').write_text(json.dumps(result,indent=2)+'\n')

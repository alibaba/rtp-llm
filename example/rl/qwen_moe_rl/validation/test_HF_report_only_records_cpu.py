"""Synthetic receipt corruption tests only; no model, GPU, Ray or RPC is used."""
from pathlib import Path
import copy
import hashlib
import importlib.util
import json
import struct
import tempfile
import unittest

HERE=Path(__file__).resolve().parent
SPEC=importlib.util.spec_from_file_location('validator',HERE/'validate_world72_core_evidence.py')
MODULE=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(MODULE)
sha=lambda value:hashlib.sha256(value.encode()).hexdigest()
bits=lambda values:struct.pack(f'<{len(values)}f',*values).hex()


def metrics(probe):
    reference=probe['hf_raw_log_probs'];errors=[]
    for route in probe['probes']:
        route['absolute_HF_errors']=[abs(a-b) for a,b in zip(reference,route['raw_log_probs'])]
        route['mean_HF_absolute_error']=sum(route['absolute_HF_errors'])/len(reference)
        errors.extend(route['absolute_HF_errors'])
    probe['max_HF_error']=max(errors)
    probe['mean_HF_absolute_error']=sum(errors)/(9*len(reference))
    dp0=probe['probes'][0]['raw_log_probs']
    probe['max_dp_error']=max(abs(a-b) for row in probe['probes'] for a,b in zip(dp0,row['raw_log_probs']))


def fixture(delta=2.0):
    records={}
    for step in range(3):
        version=f'explicit-CPU-fixture:{step}'
        expected=[-1.0-step,-2.0-step]
        for rank in range(72):
            records[f'hf-probe-step{step}-rank{rank}.json']=dict(
                schema='q38-real-HF-policy-probe-v1',global_step=step,rank=rank,
                world_size=72,master_dtype='bfloat16',compute_dtype='bfloat16',score_dtype='float32',
                raw_log_probs=list(expected),raw_fp32_bits_hex=bits(expected),
                prompt_ids=[10,20],response_ids=[30,40])
        records[f'rtp_sync_rank0_step{step}.json']=dict(global_step=step,received_weights=1590,
            logical_model_replicas=1,ranks=list(range(72)),model_version=version,
            full_source_sha256=sha(f'CPU-fixture-policy{step}'),
            tensors=[dict(name=f'fake-key-{i}',sha256=sha(f'CPU-fixture-policy{step}-key{i}')) for i in range(1590)])
        probe=dict(global_step=step,model_version=version,full_source_sha256=sha(f'CPU-fixture-policy{step}'),
            prompt_ids=[10,20],response_ids=[30,40],hf_raw_log_probs=list(expected),
            hf_raw_fp32_bits_hex=bits(expected),HF_reference_tolerance=None,
            HF_comparison_mode='report_only',HF_error_used_for_acceptance=False,
            statuses=[dict(protocol_version=3,phase='running',world_rank=rank,global_step=step,
                model_version=version,received_weights=1590,applied_weights=1,update_id='') for rank in range(72)],
            probes=[dict(dp_rank=dp,raw_log_probs=[v-delta for v in expected],
                raw_fp32_bits_hex=bits([v-delta for v in expected])) for dp in range(9)])
        metrics(probe)
        records[f'rtp_probe_rank0_step{step}.json']=probe
        records[f'rtp_status_rank0_step{step}.json']=dict(global_step=step,model_version=version,run_id='cpu-fixture-run',
            full_source_sha256=records[f'rtp_sync_rank0_step{step}.json']['full_source_sha256'],
            source_transaction_sha256=hashlib.sha256(json.dumps(records[f'rtp_sync_rank0_step{step}.json']).encode()).hexdigest(),
            statuses=copy.deepcopy(probe['statuses']))
    return records


def check(records,logprob_report_only=False):
    records=copy.deepcopy(records)
    with tempfile.TemporaryDirectory(prefix='q38-HF-report-only-CPU-fixture-') as folder:
        root=Path(folder)
        for step in range(3):
            name=f'hf-probe-step{step}-rank0.json'
            if name in records:
                raw=json.dumps(records[name]).encode()
                (root/name).write_bytes(raw)
                records[f'rtp_probe_rank0_step{step}.json']['actual_HF_reference_sha256']=hashlib.sha256(raw).hexdigest()
        # Preserve the final successful native re-probe equality contract.
        final=copy.deepcopy(records['rtp_probe_rank0_step2.json'])
        final['final_native_GPU_reprobe']=True
        records['rtp_final_policy2_actual_reprobe.json']=final
        for name,value in records.items():
            (root/name).write_text(json.dumps(value))
        validator=MODULE.Validator([root],logprob_report_only=logprob_report_only);validator.numerical_policies()
        return validator.errors,validator.details


class ReportOnlyTests(unittest.TestCase):
    def test_actual_score_disagreements_are_reported_not_gated(self):
        value=fixture();key='rtp_probe_rank0_step1.json'
        row=value[key]['probes'][8];row['raw_log_probs'][0]-=.5
        row['raw_fp32_bits_hex']=bits(row['raw_log_probs']);metrics(value[key])
        for flag in (False,True):
            errors,details=check(value,flag);self.assertEqual(errors,[])
            observations=details['optional_score_report']['steps'][1]
            example=next(x for x in observations['comparisons'] if x['hf_rank']==0 and x['dp_rank']==8)
            self.assertEqual(example['max_abs_error'],2.5)
            self.assertFalse(observations['used_for_acceptance'])

    def test_large_actual_complete_error_is_reported_without_rejection(self):
        errors,details=check(fixture());self.assertEqual(errors,[])
        for row in details['optional_score_report']['steps']:
            self.assertTrue(all(p['max_abs_error']==p['mean_abs_error']==2.0 for p in row['comparisons']))
            self.assertFalse(row['used_for_acceptance'])

    def test_zero_error_current_profile_is_also_valid(self):
        errors,details=check(fixture(0.0));self.assertEqual(errors,[])
        self.assertTrue(all(p['max_abs_error']==0.0 for row in details['optional_score_report']['steps'] for p in row['comparisons']))

    def test_invalid_or_missing_score_never_blocks_valid_sync(self):
        baseline=fixture()
        cases=[
            lambda x:x.pop('hf-probe-step1-rank71.json'),
            lambda x:x['rtp_probe_rank0_step1.json']['probes'].pop(),
            lambda x:x['rtp_probe_rank0_step1.json']['probes'][0].update(raw_log_probs=[float('nan')]),
            lambda x:x['rtp_probe_rank0_step1.json'].update(model_version='wrong-score-version'),
            lambda x:x['rtp_probe_rank0_step1.json'].update(max_HF_error=1.0),
            lambda x:x['rtp_probe_rank0_step1.json'].pop('HF_comparison_mode'),
            lambda x:x['rtp_probe_rank0_step1.json'].update(response_ids=[99,98]),
        ]
        for mutate in cases:
            value=copy.deepcopy(baseline);mutate(value);self.assertEqual(check(value)[0],[])

    def test_bad_refit_source_and_version_still_fail(self):
        for field in ('received_weights','ranks','model_version','full_source_sha256'):
            value=fixture()
            value['rtp_sync_rank0_step1.json'][field]=None
            with self.subTest(field=field):self.assertTrue(check(value)[0])

if __name__=='__main__':
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(ReportOnlyTests)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    record=dict(schema='q38-HF-report-only-offline-CPU-qualification-v1',passed=result.wasSuccessful(),
        unittest_cases=result.testsRun,large_finite_HF_native_error_2_0_is_reported_not_rejected=True,
        native_DP_self_agreement_gate_0_05_retained=False,
        scores_non_gating=True,fixture_only=True,actual_RL_or_GPU_qualification=False,
        Ray_or_Torch_or_RPC_used=False,validator_sha256=hashlib.sha256((HERE/'validate_world72_core_evidence.py').read_bytes()).hexdigest())
    (HERE/'HF-REPORT-ONLY-CPU-QUALIFIED.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record))
    raise SystemExit(0 if result.wasSuccessful() else 1)

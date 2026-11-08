"""Synthetic JSON reporting tests; historical output is not current GPU proof."""
import copy,hashlib,json,tempfile
from pathlib import Path
from types import SimpleNamespace
from validate_world72_core_evidence import Validator
from test_HF_report_only_records_cpu import fixture


def check(mutation=None):
 records=fixture(2.0)
 with tempfile.TemporaryDirectory(prefix='q38-history-reporting-fixture-') as folder:
  current=Path(folder)/'current';history=Path(folder)/'history';current.mkdir();history.mkdir()
  for step in range(3):
   raw=json.dumps(records[f'hf-probe-step{step}-rank0.json']).encode()
   records[f'rtp_probe_rank0_step{step}.json']['actual_HF_reference_sha256']=hashlib.sha256(raw).hexdigest()
  final=copy.deepcopy(records['rtp_probe_rank0_step2.json']);final['final_native_GPU_reprobe']=True
  records['rtp_final_policy2_actual_reprobe.json']=final
  for name,value in records.items():
   destination=history if (value.get('global_step')==0 and name!='rtp_final_policy2_actual_reprobe.json') else current
   (destination/name).write_text(json.dumps(value))
  old=history/'rtp_sync_rank0_step0.json';old_sha=hashlib.sha256(old.read_bytes()).hexdigest()
  p=history/'rtp_probe_rank0_step0.json';probe=json.loads(p.read_bytes());probe['source_transaction_sha256']=old_sha;p.write_text(json.dumps(probe))
  replay=SimpleNamespace(valid=True,historical_roots=[history],manifest={'canonical_reference':{'sha256':old_sha},'original_native_model_version':'explicit-CPU-fixture:0'})
  if mutation:mutation(current,history,replay)
  validator=Validator([current]);validator.initial_replay=replay;validator.numerical_policies()
  return validator.errors,validator.details

errors,details=check();assert not errors,errors
summary=details['numerical_policies'];assert summary[0]['evidence_origin']=='authenticated_historical_full7_policy0'
assert all(row['evidence_origin']=='current_run' for row in summary[1:])
assert all(row['logprob_used_for_acceptance'] is False for row in summary)
assert all(all(comp['max_abs_error']==2.0 for comp in row['comparisons']) for row in details['optional_score_report']['steps'])
cases={
 'history relabelled current':lambda c,h,r:(c/'rtp_sync_rank0_step0.json').write_bytes((h/'rtp_sync_rank0_step0.json').read_bytes()),
 'wrong authenticated history SHA':lambda c,h,r:r.manifest['canonical_reference'].update(sha256='0'*64),
}
for label,mutation in cases.items():assert check(mutation)[0],label
assert not check(lambda c,h,r:(h/'hf-probe-step0-rank71.json').unlink())[0]
assert not check(lambda c,h,r:(c/'rtp_probe_rank0_step1.json').unlink())[0]
print(json.dumps(dict(passed=True,synthetic_reports_only=True,actual_model_or_GPU_qualified=False,
 HF_error_2_0_reported_without_rejection=True,history_remains_policy0_only=True,rejected_corruptions=list(cases))))

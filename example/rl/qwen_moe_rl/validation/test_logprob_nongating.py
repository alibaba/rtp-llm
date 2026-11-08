"""Synthetic fixtures only: policy sync gates are independent of scores."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

HERE=Path(__file__).resolve().parent
REPO=Path('/Users/lilele/Desktop/Codex File/rtp-llm/example/rl/qwen_moe_rl/validation')
sys.path.insert(0,str(REPO));sys.path.insert(0,str(HERE))
from validate_world72_core_evidence import Validator
from test_HF_report_only_records_cpu import fixture


def save(root,records):
    for name,row in records.items():
        (root/name).write_text(json.dumps(row))

def required(records):
    return {k:v for k,v in records.items() if k.startswith(('rtp_sync_','rtp_status_'))}


class NonGatingTests(unittest.TestCase):
    def check(self,records):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);save(root,records)
            v=Validator([root],logprob_report_only=True,expected_run_id='cpu-fixture-run');v.numerical_policies()
            # The final validator writes allow_nan=False; corrupt score
            # material must not make the optional report unserializable.
            json.dumps(v.details,allow_nan=False)
            return v.errors,v.details

    def test_missing_all_scores_does_not_block_valid_syncs(self):
        records=required(fixture())
        errors,details=self.check(records)
        self.assertEqual(errors,[])
        self.assertEqual(len(details['numerical_policies']),3)
        self.assertFalse(details['optional_score_report']['used_for_acceptance'])
        self.assertTrue(all(not x['hf_scores'] and not x['native_scores'] for x in details['optional_score_report']['steps']))

    def test_available_scores_are_reported_and_compared(self):
        errors,details=self.check(fixture(delta=2.0))
        self.assertEqual(errors,[])
        for row in details['optional_score_report']['steps']:
            self.assertEqual(len(row['hf_scores']),72)
            self.assertEqual(len(row['native_scores']),9)
            self.assertEqual(len(row['comparisons']),72*9)
            self.assertTrue(all(x['max_abs_error']==x['mean_abs_error']==2.0 for x in row['comparisons']))
            self.assertEqual(row['hf_scores'][0]['scores'],[-1.0-row['step'],-2.0-row['step']])

    def test_bad_score_values_structure_version_or_missing_ranks_never_gate(self):
        records=fixture()
        records.pop('hf-probe-step1-rank71.json')
        records['hf-probe-step1-rank0.json']['raw_log_probs']=[float('nan'),float('inf')]
        records['hf-probe-step2-rank4.json']['prompt_ids']=[float('nan')]
        records['rtp_probe_rank0_step0.json']['probes']='invalid routes'
        records['rtp_probe_rank0_step1.json']['probes'][0]['raw_log_probs']=[float('inf')]
        records['rtp_probe_rank0_step1.json']['model_version']='wrong-score-version'
        records['rtp_probe_rank0_step2.json']['probes'][1]=None
        records['rtp_probe_rank0_step2.json']['probes'][2]['raw_fp32_bits_hex']='00'*8
        errors,details=self.check(records)
        self.assertEqual(errors,[])
        self.assertTrue(details['optional_score_report']['issues'])
        self.assertEqual(details['optional_score_report']['steps'][1]['comparisons'],[])
        forged=next(x for x in details['optional_score_report']['steps'][2]['native_scores'] if x['dp_rank']==2)
        self.assertFalse(forged['reported_bits_match_values'])

    def test_partial_native_scores_are_preserved_without_gate(self):
        records=fixture();partial=records.pop('rtp_probe_rank0_step1.json')
        partial['probes']=partial['probes'][:3]
        records['rtp_probe_partial_step1_123.json']=partial
        errors,details=self.check(records)
        self.assertEqual(errors,[])
        report=details['optional_score_report']['steps'][1]
        self.assertEqual([r['dp_rank'] for r in report['native_scores']],[0,1,2])
        self.assertEqual(len(report['comparisons']),72*3)
        self.assertTrue(all('rtp_probe_partial_step1_123.json' in r['path'] for r in report['native_scores']))

    def test_invalid_json_score_files_never_gate(self):
        records=required(fixture())
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);save(root,records)
            (root/'hf-probe-step0-rank0.json').write_text('{')
            (root/'rtp_probe_rank0_step1.json').write_text('null')
            (root/'rtp_final_policy2_actual_reprobe.json').write_text('[')
            v=Validator([root]);v.numerical_policies()
            self.assertEqual(v.errors,[]);self.assertEqual(len(v.details['optional_score_report']['issues']),3)

    def test_bad_missing_or_unchanged_sync_never_passes(self):
        for case in ('missing','bad-ranks','bad-count','bad-hash','bad-version','same-version','same-source','missing-key','changed-keys'):
            records=required(fixture())
            key='rtp_sync_rank0_step1.json';sync=records[key]
            if case=='missing':del records[key]
            if case=='bad-ranks':sync['ranks']=list(range(71))
            if case=='bad-count':sync['received_weights']=1589
            if case=='bad-hash':sync['full_source_sha256']='bad'
            if case=='bad-version':sync['model_version']=None
            if case=='same-version':sync['model_version']=records['rtp_sync_rank0_step0.json']['model_version']
            if case=='same-source':sync['full_source_sha256']=records['rtp_sync_rank0_step0.json']['full_source_sha256']
            if case=='missing-key':sync['tensors'].pop()
            if case=='changed-keys':sync['tensors'][0]['name']='not-in-the-other-policy'
            with self.subTest(case=case):self.assertTrue(self.check(records)[0])

    def test_native_world_status_is_required_and_bound(self):
        for case in ('missing','rank','step','version','digest','transaction','run','phase','applied','received','update'):
            records=required(fixture());key='rtp_status_rank0_step1.json';status=records[key]
            if case=='missing':del records[key]
            if case=='rank':status['statuses'].pop()
            if case=='step':status['statuses'][9]['global_step']=0
            if case=='version':status['statuses'][9]['model_version']='wrong'
            if case=='digest':status['full_source_sha256']='0'*64
            if case=='transaction':status['source_transaction_sha256']='0'*64
            if case=='run':status['run_id']='other-run'
            if case=='phase':status['statuses'][9]['phase']='paused'
            if case=='applied':status['statuses'][9]['applied_weights']=0
            if case=='received':status['statuses'][9]['received_weights']=1589
            if case=='update':status['statuses'][9]['update_id']='unfinished'
            with self.subTest(case=case):self.assertTrue(self.check(records)[0])

    def test_replay_binds_historical_sync_without_needing_scores(self):
        records=fixture()
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)/'current';history=Path(directory)/'history';root.mkdir();history.mkdir()
            save(root,{k:v for k,v in required(records).items() if 'step0' not in k})
            save(history,{k:v for k,v in required(records).items() if 'step0' in k})
            old=history/'rtp_sync_rank0_step0.json'
            replay=SimpleNamespace(valid=True,historical_roots=[history],manifest={
                'canonical_reference':{'sha256':hashlib.sha256(old.read_bytes()).hexdigest()},
                'original_native_model_version':records['rtp_sync_rank0_step0.json']['model_version']})
            v=Validator([root]);v.initial_replay=replay;v.numerical_policies();self.assertEqual(v.errors,[])
            self.assertEqual(v.details['numerical_policies'][0]['evidence_origin'],'authenticated_historical_full7_policy0')
            replay.manifest['canonical_reference']['sha256']='0'*64
            v=Validator([root]);v.initial_replay=replay;v.numerical_policies();self.assertTrue(v.errors)

    def test_historical_probe_status_fallback_remains_transaction_bound(self):
        records=fixture()
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)/'current';history=Path(directory)/'history';root.mkdir();history.mkdir()
            save(root,{k:v for k,v in required(records).items() if 'step0' not in k})
            save(history,{'rtp_sync_rank0_step0.json':records['rtp_sync_rank0_step0.json']})
            sync=history/'rtp_sync_rank0_step0.json';digest=hashlib.sha256(sync.read_bytes()).hexdigest()
            probe=records['rtp_probe_rank0_step0.json'];probe['source_transaction_sha256']=digest
            save(history,{'rtp_probe_rank0_step0.json':probe})
            replay=SimpleNamespace(valid=True,historical_roots=[history],manifest={
                'canonical_reference':{'sha256':digest},'original_native_model_version':records['rtp_sync_rank0_step0.json']['model_version']})
            v=Validator([root]);v.initial_replay=replay;v.numerical_policies();self.assertEqual(v.errors,[])
            probe['statuses'].pop();save(history,{'rtp_probe_rank0_step0.json':probe})
            v=Validator([root]);v.initial_replay=replay;v.numerical_policies();self.assertTrue(v.errors)


if __name__=='__main__':unittest.main()

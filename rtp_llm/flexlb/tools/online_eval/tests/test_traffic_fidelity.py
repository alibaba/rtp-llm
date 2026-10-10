"""Production sampling and standalone input-fidelity contracts."""
import hashlib
import json
import random
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest.mock import patch
from traffic.prefix_lineage import encode,decode,expand
from traffic.structure import PrefixIndex,capture_shape,generated_shape
from traffic.derive_synthetic_parameters import derive_parameters
from traffic.realistic import resolve,iter_requests
from traffic.datasets import profile_path,trace_models
from analysis.traffic_fidelity import audit,load_capture,run,compare,grade,ks,thresholds
from analysis.traffic_fidelity_report import render,markdown


def fixture():
    rng=random.Random(17);events=[[i,512*512,-1,0] for i in range(32)]
    for i in range(32,4032):
        blocks=rng.randint(20,500);shared=0 if i%10==0 else int(blocks*.85)
        events.append([i,blocks*512,i%32 if shared else -1,shared])
    raw=encode(events,dict(source_start=0,source_end=4031))
    return raw,derive_parameters(raw,{},None)


class TrafficFidelityTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.raw,cls.profile=fixture()

    def test_prefix_index_matches_naive_observation(self):
        rng=random.Random(19);index=PrefixIndex();seen=[]
        rows=[[1,2,3],[1,2],[1,2,4],[[8,9],1,2],[[8,9],1,3]]
        for _ in range(300):
            parent=rng.choice(rows)
            rows.append(parent[:rng.randrange(len(parent)+1)]+list(range(rng.randrange(1,30),35)))
        for row in rows:
            expected=0
            for old in seen:
                shared=0
                for a,b in zip(old,row):
                    if a!=b: break
                    shared+=1
                expected=max(expected,shared)
            self.assertEqual(expected,index.observe(row),row);seen.append(row)
        events=decode(self.raw)[1]
        self.assertEqual(sorted(Counter(tuple(p[:8]) for _,p in expand(events)).values(),reverse=True),capture_shape(events)['families'])

    def test_joint_uses_depth_and_is_repeatable(self):
        args=dict(seed=42,count=1000,output_tokens=5,sampling='joint')
        p,_=resolve(args,document=self.profile)
        a=list(iter_requests(p,'x'));self.assertEqual(a,list(iter_requests(p,'x')))
        other=dict(p,output_distribution=dict(values=[77],weights=[1]))
        self.assertEqual([r['input_token_blocks'] for r in a],[r['input_token_blocks'] for r in iter_requests(other,'x')])
        legacy=json.loads(json.dumps(self.profile))
        legacy.pop('schema_version');legacy['parameters'].pop('joint_distribution')
        old,_=resolve(dict(args,sampling='independent'),document=legacy)
        upgraded,_=resolve(dict(args,sampling='independent'),document=self.profile)
        self.assertEqual(list(iter_requests(old,'legacy')),list(iter_requests(upgraded,'legacy')))
        high=dict(p,families=1,cold_fraction=0,joint_distribution=dict(warm_pairs=[[20,20]],cold_blocks=[]))
        low=dict(high,joint_distribution=dict(warm_pairs=[[20,2]],cold_blocks=[]))
        self.assertGreater(sum(generated_shape(iter_requests(high,'x'))['shared']),sum(generated_shape(iter_requests(low,'x'))['shared']))
        with tempfile.TemporaryDirectory() as directory:
            from traffic.traffic_source import materialize
            root=Path(directory);profile=root/'profile.json';profile.write_text(json.dumps(self.profile))
            with patch('traffic.realistic.profile_path',return_value=profile):
                for name in ('a','b'):
                    materialize(root/(name+'.jsonl'),dict(kind='synthetic',model='realistic',version='1',parameters=args),'x',directory)
                self.assertEqual((root/'a.jsonl').read_bytes(),(root/'b.jsonl').read_bytes())
                self.assertFalse((root/'fidelity.html').exists())

    def test_joint_improves_actual_structure(self):
        real=capture_shape(decode(self.raw)[1]);metrics={}
        for method in ('joint','independent'):
            p,_=resolve(dict(seed=42,count=10000,output_tokens=1,sampling=method),document=self.profile)
            metrics[method]=compare(real,generated_shape(iter_requests(p,'test')))
        self.assertLess(metrics['joint']['length_ks'],.05)
        self.assertLess(metrics['joint']['depth_ks'],metrics['independent']['depth_ks']*.6)
        self.assertLess(metrics['joint']['joint_tv'],metrics['independent']['joint_tv']*.6)

    def test_invalid_overrides_and_cold_only(self):
        for fields in (dict(input_distribution=dict(values=[512],weights=[1])),dict(shared_blocks=1),dict(sampling='unknown')):
            args=dict(seed=1,count=2,output_tokens=1,sampling='joint');args.update(fields)
            with self.assertRaises(ValueError): resolve(args,document=self.profile)
        prof=derive_parameters(encode([[0,512,-1,0],[1,1024,-1,0]]),{},None)
        p,_=resolve(dict(seed=1,count=20,output_tokens=1,sampling='joint'),document=prof)
        self.assertEqual(0,sum(generated_shape(iter_requests(p,'x'))['shared']))

    def test_metrics_grades_audit_and_stable_report(self):
        self.assertEqual(.5,ks([1,1],[1,2]));self.assertEqual(0,ks([1,2],[2,1]))
        self.assertEqual('WARN',grade(dict(length_ks=0,depth_ks=0,joint_tv=0))['cache'])
        self.assertEqual('FAIL',grade(dict(length_ks=.3,depth_ks=.4,joint_tv=.5))['cache'])
        self.assertEqual('UNASSESSED',grade({},valid=False)['length'])
        with self.assertRaises(ValueError): thresholds(dict(length_ks=[.9,.1]))
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);source=root/'renamed.xz';source.write_bytes(self.raw)
            profile=root/'profile.json';profile.write_text(json.dumps(self.profile))
            a=run(profile,[source],count=1000);b=run(profile,[source],count=1000)
            self.assertEqual(a,b);self.assertEqual(render(a),render(b));self.assertEqual(markdown(a),markdown(b))
            self.assertEqual('OK',a['identity']['status']);self.assertEqual('renamed',a['fit_source'])
            self.assertNotIn('<script src=',render(a))
            a['limitations'].append('</script><script>alert(1)</script>')
            self.assertNotIn('</script><script>alert(1)',render(a))
            absent=run(profile,[],count=10)
            self.assertEqual('UNAVAILABLE',absent['identity']['status']);self.assertIn('UNASSESSED',render(absent))
            cap=load_capture(source)
            for key in ('families','prefix_blocks','cold_fraction'):
                changed=json.loads(json.dumps(self.profile));changed['parameters'][key]+=1
                self.assertEqual('MISMATCH',audit(changed,cap)['status'])
            changed=json.loads(json.dumps(self.profile));changed['parameters']['families']+=1
            profile.write_text(json.dumps(changed))
            from scripts.commands.compare_traffic import main
            from contextlib import redirect_stdout
            from io import StringIO
            with redirect_stdout(StringIO()):
                code=main(['--profile',str(profile),'--captures',str(source),'--out',str(root/'report'),
                           '--count','20','--check'])
            self.assertEqual(2,code)

    def test_bundled_profile_audit(self):
        # Part of normal regression discovery; manually defined shapes make no empirical claim.
        paths=list(trace_models().values())
        for path in profile_path().parent.glob('*.profile.json'):
            prof=json.loads(path.read_text());sha=prof.get('calibration',{}).get('model_sha256')
            if not sha: continue
            source=next((p for p in paths if hashlib.sha256(p.read_bytes()).hexdigest()==sha),None)
            self.assertIsNotNone(source,path)
            result=audit(prof,load_capture(source));self.assertEqual('OK',result['status'],result)


if __name__=='__main__':unittest.main()

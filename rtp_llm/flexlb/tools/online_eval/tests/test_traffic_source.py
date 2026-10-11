import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from traffic.traffic_source import materialize,validate_plan,SOURCES
from traffic.prefix_lineage import encode,decode,expand
from traffic.playback_config import normalize


class TrafficSourceTest(unittest.TestCase):
    def test_registered_source_versions(self):
        self.assertEqual(set(SOURCES),{('synthetic','realistic','1'),('trace','prefix_lineage','2'),('trace','prefix_lineage','3')})

    def test_lineage_true_timestamps_and_old_label_allocation(self):
        events=[[0,1537,-1,0],[51,2050,0,2],[999,1025,1,1],[1000,100,-1,0]]
        raw=encode(events,{'missing_pod_indices':[7]})
        meta,decoded=decode(raw)
        paths=list(expand(decoded))
        self.assertEqual([t for t,_ in paths],[0,51,999,1000])
        self.assertEqual(paths[0][1],[1,2,3])
        self.assertEqual(paths[1][1],[1,2,5,6])
        self.assertEqual(paths[2][1],[1,8])
        self.assertEqual(meta['token_adjustment']['short_requests'],1)
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);model=root/'model.xz';model.write_bytes(raw)
            spec=dict(kind='trace',model='prefix_lineage',version='2',parameters=dict(
                path=str(model),sha256=hashlib.sha256(raw).hexdigest(),count=4,output_tokens=8,priority=50))
            target=materialize(root/'trace.jsonl',spec,'r',root)
            rows=[json.loads(l) for l in target.read_text().splitlines()]
            self.assertEqual([r['ts'] for r in rows],[0,51,999,1000])
            self.assertEqual([r['il'] for r in rows],[1536,2048,1024,512])
            manifest=json.loads(target.with_suffix('.manifest.json').read_text())
            self.assertEqual(manifest['provenance']['missing_pod_indices'],[7])
            for field in ('realism','arrival','tail'):self.assertIn(field,manifest)
            spec['parameters']['qps']=240
            with self.assertRaises(ValueError):materialize(root/'bad.jsonl',spec,'r',root)
            self.assertFalse((root/'bad.jsonl').exists())

    def test_lineage_rejects_bad_references_and_corruption(self):
        for events in ([[0,512,0,0]],[[1,512,-1,0],[0,512,0,1]],[[0,512,-1,1]]):
            with self.assertRaises(ValueError):encode(events)
        raw=encode([[0,512,-1,0]])
        with self.assertRaises(Exception):decode(raw[:-4])

    def test_compact_pinned_and_partial_tail_validation(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'trace.jsonl'
            row=dict(rid='r:1',ts=0,il=1031,ol=8,priority=50,cache_key_block_size=512,
                input_token_blocks=[[1,2]*256,8,9])
            path.write_text(json.dumps(row)+'\n');self.assertEqual(validate_plan(path,'r'),1)
            for invalid in (dict(row,il=1024),dict(row,input_ids=[1]),dict(row,input_token_blocks=[-1,8,9]),dict(row,request_id_int=7)):
                path.write_text(json.dumps(invalid)+'\n')
                with self.assertRaises(ValueError):validate_plan(path,'r')

    def test_long_statistical_inputs_and_independent_outputs(self):
        with tempfile.TemporaryDirectory() as d:
            spec=dict(kind='synthetic',model='realistic',version='1',parameters=dict(
                seed=5,count=30,families=2,prefix_blocks=2,zipf_alpha=1,cold_fraction=0.2,
                input_distribution=dict(values=[600000],weights=[1]),
                output_distribution=dict(values=[8,64],weights=[1,1])))
            one=materialize(Path(d)/'a.jsonl',spec,'r',d)
            rows=[json.loads(l) for l in one.read_text().splitlines()]
            self.assertTrue(all(len(r['input_token_blocks'])>1024 for r in rows))
            self.assertEqual({r['ol'] for r in rows},{8,64})
            spec['parameters']['output_distribution']=dict(values=[1],weights=[1])
            two=materialize(Path(d)/'b.jsonl',spec,'r',d)
            second=[json.loads(l) for l in two.read_text().splitlines()]
            self.assertEqual([r['input_token_blocks'] for r in rows],[r['input_token_blocks'] for r in second])

    def test_playback_explicit_laps_and_no_duplicate_controls(self):
        env,p=normalize(dict(DURATION_S='60',playback=dict(mode='burst',qps=20,burst_factor=5,max_laps=0)))
        self.assertEqual(env['LOOP'],'true');self.assertEqual(p['identity'],'structural-relabel')
        with self.assertRaises(ValueError):normalize(dict(playback=dict(mode='uniform',qps=1,max_laps=0)))
        with self.assertRaises(ValueError):normalize(dict(SEND_MODE='uniform',playback=dict(mode='uniform',qps=1)))
        with self.assertRaises(ValueError):normalize(dict(playback=dict(mode='true-ts',qps=1)))

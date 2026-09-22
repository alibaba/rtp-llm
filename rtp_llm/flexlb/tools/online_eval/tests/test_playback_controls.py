import hashlib
import json
import math
import tempfile
import unittest
from pathlib import Path

from traffic.playback import normalize, comparison_notice
from traffic.playback_controls import integral, inverse, poisson_count
from traffic.output_sampling import output_sampler, event_uniform
from traffic import prefix_lineage, prefix_lineage_v3


class PlaybackControlsTest(unittest.TestCase):
    def test_yaml_env_roundtrip_and_ab_mismatch(self):
        p = dict(mode='uniform', qps=100, seed=42, arrival='poisson',
                 rate_curve=[[0, 1], [10, 3], [20, 1]], identity='partial',
                 retain_schedule=dict(kind='linear', start=0, end=1, laps=5))
        env, a = normalize(dict(playback=p))
        _, b = normalize(env)
        for key in p:
            self.assertEqual(a[key], b[key])
        self.assertNotIn('LAP_RETAIN_PROBABILITY', env)
        for key, value in [('rate_curve', [[0, 2]]), ('arrival', 'deterministic'),
                           ('seed', 43), ('retain_schedule', dict(kind='sequence', values=[0, 1]))]:
            self.assertIn('DIFFERENT', comparison_notice(dict(playback=a), dict(playback=dict(a, **{key:value}))))
        self.assertIn('DIFFERENT', comparison_notice(dict(output_distribution={'values':[1]}),
                                                   dict(output_distribution={'values':[2]})))

    def test_invalid_combinations_fail_before_launch(self):
        base = dict(mode='uniform', qps=10)
        for extra in [dict(arrival='possion'), dict(arrival='poisson'),
                      dict(arrival='poisson', seed=True), dict(arrival='poisson', seed=2**63),
                      dict(arrival='poisson', seed=1, ramp_up_seconds=1),
                      dict(rate_curve=[[0, 0]]), dict(rate_curve=[[1, 1]]),
                      dict(rate_curve=[[0, 1], [0, 2]]), dict(rate_curve=[[0, float('inf')]]),
                      dict(rate_curve=[[0, 1]], burst_factor=2), dict(rate_curv=[[0, 1]]),
                      dict(retain_schedule=dict(kind='sequence',values=[0,1]), seed=1),
                      dict(identity='partial', seed=1, retain_probability=0,
                           retain_schedule=dict(kind='sequence',values=[0,1])),
                      dict(identity='partial',seed=1,retain_schedule=dict(kind='linear',start=0,end=1,laps=1)),
                      dict(identity='partial',seed=1,retain_schedule=dict(kind='sequence',values=[0,1,.5])),
                      dict(identity='partial',seed=1,retain_schedule=dict(kind='sequence',values=[0,1],typo=1))]:
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                normalize(dict(playback=dict(base, **extra)))
        with self.assertRaises(ValueError):
            normalize(dict(playback=dict(base), RATE_CURVE='[[0,1]]'))
        with self.assertRaises(ValueError):
            normalize(dict(SEND_MODE='uniform', ARRIVAL_PROCESS='poisson'))
        with self.assertRaises(ValueError):
            normalize(dict(playback=dict(mode='true-ts', arrival='poisson',seed=1)))
        normalize(dict(playback=dict(mode='true-ts',rate_curve=[[0,.5],[10,2]])))

    def test_integrated_curve_counts_and_seeded_poisson_budget(self):
        curve = [[0, .5], [10, 3], [20, 1]]
        for i in range(1001):
            t = i/20
            self.assertAlmostEqual(inverse(curve, integral(curve, t)), t, places=10)
        # Deterministic inversion has <= one event window quantization error.
        due = [inverse(curve, i/100) for i in range(6000)]
        for end in [1, 5, 10, 20, 30]:
            expected=100*integral(curve,end)
            self.assertLessEqual(abs(sum(t<end for t in due)-expected), 1.000001)
        expected = 100*integral(curve,30)
        self.assertLess(abs(poisson_count(expected,42)-expected), 5*math.sqrt(expected))
        self.assertEqual(poisson_count(100,42), poisson_count(100,42))
        with self.assertRaises(ValueError): poisson_count(1e9,42,limit=3)
        # Golden cross-language arrival intensity for seed 42.
        self.assertAlmostEqual(-math.log1p(-event_uniform(0,42 ^ 0xd1b54a32d192ed03)), 1.5354135069822445, places=8)

    def test_discrete_lineage_preserves_snapshot_filter_and_original_index(self):
        for version, module in [(2,prefix_lineage),(3,prefix_lineage_v3)]:
            events = ([[i, 1024, -1, 0] for i in range(100)] if version==2
                      else [[i, 32769 if i%3==0 else 513, -1, 0] for i in range(100)])
            # Both encoders consume original four-column source events.
            raw = module.encode(events)
            with tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp); model=root/'model.xz'; model.write_bytes(raw)
                params=dict(path='model.xz',sha256=hashlib.sha256(raw).hexdigest(),count=100,
                    output_tokens=64,priority=50,output_distribution=dict(kind='discrete',values=[1,16,64],weights=[0,1,3],seed=7))
                module.write_trace(root/'a',params,'A',root)
                module.write_trace(root/'b',params,'B',root)
                a=[json.loads(x) for x in (root/'a').read_text().splitlines()]
                b=[json.loads(x) for x in (root/'b').read_text().splitlines()]
                self.assertEqual([r['ol'] for r in a],[r['ol'] for r in b])
                self.assertEqual(set(r['ol'] for r in a),{16,64})
                if version==3:
                    module.write_trace(root/'c',dict(params,max_input_tokens=32768),'C',root,max_requests=20)
                    c=[json.loads(x) for x in (root/'c').read_text().splitlines()]
                    for row in c: self.assertEqual(row['ol'],a[int(row['rid'].split(':')[-1])]['ol'])
                self.assertEqual(model.read_bytes(),raw)

    def test_output_invalid_modes_and_weights(self):
        policy=dict(kind='discrete',values=[1,4],weights=[1,2],seed=4)
        for bad in [dict(policy,mean_tokens=2),dict(policy,values=[1,5]),dict(policy,weights=[0,0]),
                    dict(policy,weights=[1,float('nan')]),dict(policy,weights=[1,-1]),
                    dict(policy,weights=[1]),dict(policy,seed=True),dict(policy,kind='unknown')]:
            with self.assertRaises(ValueError): output_sampler(dict(output_tokens=4,output_distribution=bad))
        sample=output_sampler(dict(output_tokens=4,output_distribution=policy))
        self.assertAlmostEqual(sum(sample(i)==4 for i in range(10000))/10000,2/3,delta=.02)

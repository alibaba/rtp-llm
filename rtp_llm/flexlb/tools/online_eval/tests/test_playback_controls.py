from traffic.lineage_transforms import LineageProjector
from traffic.codecs import codec_for_version
import json
import tempfile
import unittest
from pathlib import Path

from adapters.load_client import client_environment
from traffic.playback.configuration import PlaybackConfigurationCodec
from traffic.output_sampling import output_sampler
from traffic import prefix_lineage, prefix_lineage_v3


class PlaybackControlsTest(unittest.TestCase):
    def test_bounded_multi_lap_environment_roundtrip(self):
        p = dict(mode='uniform', duration_s=10, qps=100, max_laps=3, identity='structural-relabel', ramp_up_seconds=2)
        env, a = client_environment(dict(playback=p))
        b = PlaybackConfigurationCodec.from_environment(env).to_dict()
        self.assertEqual(a, b)
        self.assertEqual(env['MAX_LAPS'], '3')
        self.assertEqual(env['LOOP'], 'true')
        with self.assertRaises(ValueError):
            client_environment(dict(playback=dict(mode='uniform', qps=10, max_laps=0)))

    def test_unknown_fields_and_invalid_values_are_rejected(self):
        for key in ('arrival', 'rate_curve', 'retain_schedule', 'retain_probability',
                    'seed', 'burst_factor', 'diurnal_amplitude'):
            with self.subTest(key=key), self.assertRaises(ValueError):
                client_environment(dict(playback=dict(mode='uniform', qps=10, **{key: 1})))
        for mode in ('burst', 'gradient'):
            with self.assertRaises(ValueError):
                client_environment(dict(playback=dict(mode=mode, qps=10)))
        with self.assertRaises(ValueError):
            client_environment(dict(playback=dict(mode='uniform', qps=10, identity='partial')))
        with self.assertRaises(ValueError):
            client_environment(dict(SEND_MODE='uniform', SEND_MODE_QPS='10', MAX_LAPS='3', LOOP='false'))

    def test_discrete_lineage_preserves_snapshot_filter_and_original_index(self):
        for version, module in [(2,prefix_lineage),(3,prefix_lineage_v3)]:
            events = ([[i, 1024, -1, 0] for i in range(100)] if version==2
                      else [[i, 32769 if i%3==0 else 513, -1, 0] for i in range(100)])
            # Both encoders consume original four-column source events.
            raw = module.encode(events)
            with tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp); model=root/'model.xz'; model.write_bytes(raw)
                params=dict(path='model.xz',count=100,
                    output_tokens=64,priority=50,output_distribution=dict(kind='discrete',values=[1,16,64],weights=[0,1,3],seed=7))
                LineageProjector(codec_for_version(version).decode).write_trace(root/'a',params,'A',root)
                LineageProjector(codec_for_version(version).decode).write_trace(root/'b',params,'B',root)
                a=[json.loads(x) for x in (root/'a').read_text().splitlines()]
                b=[json.loads(x) for x in (root/'b').read_text().splitlines()]
                self.assertEqual([r['ol'] for r in a],[r['ol'] for r in b])
                self.assertEqual(set(r['ol'] for r in a),{16,64})
                if version==3:
                    LineageProjector(codec_for_version(version).decode).write_trace(root/'c',dict(params,max_input_tokens=32768),'C',root,max_requests=20)
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

"""Fetch verification must remain fail-closed after removing RPC exposition."""
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import observe


class ObserveTest(unittest.TestCase):
    def read(self, engines):
        prom = b'decode_unused 0\nmock_engine_completed_total{role="decode"} 4\nrtp_llm_generate_tps{role="decode"} 8\n'
        with patch.object(observe.urllib.request, 'urlopen', side_effect=[
            io.BytesIO(prom), io.BytesIO(json.dumps({'engines': engines}).encode())
        ]) as get:
            result = observe.read_metrics('http://mock/')
        self.assertEqual(get.call_args_list[1].args[0], 'http://mock/snapshot')
        return result

    def test_fetch_counts_sum_by_role_without_rpc_metric(self):
        rows = [{'role': role, 'rpc_counts': {'fetch_response': n}}
                for role, n in [('prefill', 0), ('decode', 1), ('decode', 2)]]
        result = self.read(rows)
        self.assertEqual(result['prefill.fetch_response'], 0)
        self.assertEqual(result['decode.fetch_response'], 3)
        self.assertEqual(result['decode.mock_engine_completed_total'], 4)

    def test_missing_fetch_counter_is_not_zero(self):
        with self.assertRaises(KeyError):
            self.read([{'role': 'prefill', 'rpc_counts': {}}])

    def test_missing_role_is_rejected_by_observer(self):
        values = self.read([{'role': 'prefill', 'rpc_counts': {'fetch_response': 0}}])
        with patch.object(observe, 'read_metrics', return_value=values), tempfile.TemporaryDirectory() as td:
            with self.assertRaisesRegex(ValueError, 'missing'):
                observe.observe('http://mock', 1, 1, Path(td) / 'report.json')

    def test_nonzero_fetch_cannot_pass_zero_fetch_acceptance(self):
        values = {'prefill.fetch_response': 0, 'decode.fetch_response': 1,
                  'decode.mock_engine_completed_total': 4, 'decode.rtp_llm_generate_tps': 8}
        with patch.object(observe, 'read_metrics', return_value=values), \
             patch.object(observe.time, 'sleep'), \
             patch.object(observe.time, 'monotonic', side_effect=[0, 1]), tempfile.TemporaryDirectory() as td:
            report = observe.observe('http://mock', 1, 1, Path(td) / 'report.json')
        self.assertFalse(report['zero_fetch_rpc'])


if __name__ == '__main__':
    unittest.main()

import argparse
import contextlib
import io
import json
import tempfile
import unittest
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import kimi_k3_perf as perf


class PrefillWorkingSetTest(unittest.TestCase):
    def test_decode_bounds_unused_prefill_workspace_without_shortening_kv(self):
        from rtp_llm.models_py.modules.kimi_k3.utils import (
            collective_gemm_workspace_global_tokens,
            prefill_chunk_tokens,
        )

        for configured, expected in ((None, 65536), ("32768", 32768), ("0", 960519)):
            with self.subTest(chunk=configured), tempfile.TemporaryDirectory() as directory:
                args = argparse.Namespace(
                    mode="decode", input_len=960000, batch_size=1,
                    workset_size=1, rounds=1, decode_test_length=512,
                    result_dir=Path(directory), profile=False,
                    profile_trace_name="test", dp_size=1,
                )
                env = {} if configured is None else {"KIMI_K3_PREFILL_CHUNK_TOKENS": configured}
                remaining = ["--role_type", "PDFUSION", "--sp_type", "mtp"]

                def check_geometry(*, max_seq_len, max_concurrency):
                    self.assertEqual((max_seq_len, max_concurrency), (960519, 1))
                    self.assertEqual(
                        collective_gemm_workspace_global_tokens(
                            max_seq_len, 1, prefill_chunk_tokens()
                        ),
                        expected,
                    )

                with patch.dict(perf.os.environ, env, clear=True), \
                     patch.object(perf, "parse_args", return_value=(args, remaining)), \
                     patch.object(perf, "create_unique_queries", return_value=["A"]), \
                     patch.object(perf, "EngineServer") as server, \
                     patch.object(perf, "BatchPerfImpl") as batch, \
                     contextlib.redirect_stdout(io.StringIO()):
                    server.return_value.start.side_effect = check_geometry
                    batch.return_value.run.return_value = SimpleNamespace(avg_decode_time=1.0)
                    self.assertEqual(perf.main(), 0)
                    server.return_value.start.assert_called_once()
                    self.assertEqual(server.call_args.args[1], remaining)

    def test_single_role_reuse_survives_metric_aggregation(self):
        from dataclasses import asdict
        from rtp_llm.utils.base_model_datatypes import AuxInfo
        from rtp_llm.test.perf_test.dataclass import ResponseInfo, analyze_results

        responses = [
            ResponseInfo({"aux_info": asdict(AuxInfo(
                input_len=256, output_len=1, reuse_len=local,
                local_reuse_len=local, memory_reuse_len=memory,
                pd_sep=False,
            ))})
            for local, memory in ((64, 0), (128, 96))
        ]
        metrics = analyze_results(responses)
        self.assertEqual(metrics.avg_reuse_len, 96)
        self.assertEqual(metrics.avg_local_reuse_len, 96)
        self.assertEqual(metrics.avg_memory_reuse_len, 48)
        self.assertEqual(metrics.avg_prefill_memory_reuse_len, 0)

    def test_evicted_prefixes_are_not_warmed_before_measurement(self):
        cache = OrderedDict()
        visits = []

        class FakeBatch:
            def __init__(self, port, dp, bs, queries, **kwargs):
                self.queries, self.options = queries, kwargs

            def run(self):
                result = SimpleNamespace(hit=False, avg_prefill_time=30.0, avg_wait_time=20.0)
                for phase, count in (
                    ('warmup', self.options.get('warmup_runs', 1)),
                    ('measure', self.options.get('measure_runs', 1)),
                    ('profile', self.options.get('profile_runs', 0)),
                ):
                    for _ in range(count):
                        for query in self.queries:
                            hit = query in cache
                            cache[query] = True
                            cache.move_to_end(query)
                            if len(cache) > 2:
                                cache.popitem(last=False)
                            visits.append((phase, query, hit))
                            result = SimpleNamespace(hit=hit, avg_prefill_time=30.0, avg_wait_time=20.0)
                return result

        with tempfile.TemporaryDirectory() as directory:
            args = argparse.Namespace(
                mode='prefill', input_len=960000, batch_size=1,
                workset_size=3, rounds=2, decode_test_length=512,
                result_dir=Path(directory), profile=True,
                profile_trace_name='test', dp_size=1,
            )
            with patch.object(perf, 'parse_args', return_value=(args, ['--sp_type', 'mtp', '--tp_size', '8'])), \
                 patch.dict(perf.os.environ, {'SP_TYPE': 'eagle3'}), \
                 patch.object(perf, 'create_unique_queries', return_value=['A', 'B', 'C']), \
                 patch.object(perf, 'EngineServer') as server, \
                 patch.object(perf, 'BatchPerfImpl', FakeBatch), \
                 contextlib.redirect_stdout(io.StringIO()):
                perf.os.environ.pop('KIMI_K3_SHARED_EXPERT_WEIGHT_SHARD', None)
                self.assertEqual(perf.main(), 0)
                self.assertEqual(perf.os.environ['SP_TYPE'], 'mtp')
                self.assertEqual(perf.os.environ['KIMI_K3_SHARED_EXPERT_WEIGHT_SHARD'], '1')
            server.return_value.stop.assert_called_once()
            server.return_value.start.assert_called_once_with(
                max_seq_len=960008, max_concurrency=1,
            )
            self.assertEqual(server.call_args.args[1][-6:], [
                '--use_batch_decode_scheduler', '0', '--role_type', 'PREFILL', '--reuse_cache', '1',
            ])
            result = json.loads((Path(directory) / 'k3_perf.json').read_text())
        self.assertEqual([q for _, q, _ in visits], list('ABCABCABCABCA'))
        self.assertEqual([phase for phase, _, _ in visits], ['measure'] * 12 + ['profile'])
        self.assertEqual(len(result['metrics']), 2)
        self.assertTrue(all(not row['hit'] for round_ in result['metrics'] for row in round_['metrics']))
        self.assertTrue(all(row['avg_ttft_ms'] == 50.0 for round_ in result['metrics'] for row in round_['metrics']))


if __name__ == '__main__':
    unittest.main()

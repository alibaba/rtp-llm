"""Regression guard for the collector's explicit CUDA API contract.

This source check supplements GPU integration profiling; it cannot detect
synchronization hidden inside an external library or prove runtime performance.
"""
import re
from pathlib import Path
import unittest

ROOT = Path(__file__).parents[2]


def body(source, signature):
    start = source.index('{', source.index(signature))
    depth = 1
    end = start + 1
    while depth:
        depth += (source[end] == '{') - (source[end] == '}')
        end += 1
    text = source[start:end]
    return re.sub(r'//[^\n]*|/\*.*?\*/', '', text, flags=re.S)


class NoSyncContractTest(unittest.TestCase):
    def test_snapshot_has_no_wait_allocation_or_host_transfer(self):
        source = (ROOT / 'cpp/utils/ForwardTrace.cc').read_text()
        hot = body(source, 'void ForwardTraceSession::snapshot(')
        for forbidden in ('synchronize(', '.cpu(', '.item', '.to(', 'torch::empty', 'cudaMalloc',
                          'cudaFree', 'cudaMemcpyDeviceToHost', 'cudaStreamWaitEvent', '.block(',
                          'torch::Event', '.copy_(', '.contiguous('):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, hot)
        self.assertIn('cudaMemcpyDeviceToDevice', hot)
        self.assertIn('used_ + out.count <= arena_.numel()', hot)

    def test_only_exporter_materializes_values(self):
        paths = [ROOT / 'cpp/utils/ForwardTrace.cc', ROOT / 'cpp/utils/ForwardTraceExport.cc',
                 ROOT / 'cpp/models/PyWrappedModel.cc', ROOT / 'cpp/cuda_graph/cuda_graph_runner.cc']
        calls = [(p.name, re.findall(r'(?<!::)\bmaterialize\(\)', p.read_text())) for p in paths]
        self.assertEqual([(name, len(found)) for name, found in calls if found], [('ForwardTraceExport.cc', 1)])


if __name__ == '__main__':
    unittest.main()

"""Compare actual old/new controller method AST with CPU-only fixtures.

No model, Torch, Ray, RPC, process group, or native runtime is constructed.
"""
import argparse
import ast
import asyncio
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import tempfile
from types import SimpleNamespace


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--old', required=True)
    p.add_argument('--new', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    reports = []

    async def run(source, kind, native_values):
        tree = ast.parse(source)
        method = next(n for c in tree.body if isinstance(c, ast.ClassDef)
                      for n in c.body if isinstance(n, ast.AsyncFunctionDef)
                      and n.name == '_wake_and_probe_locked')
        helper = [n for n in tree.body if isinstance(n, ast.FunctionDef)
                  and n.name == '_probe_attempt_payload']
        with tempfile.TemporaryDirectory(prefix='q38-probe-audit-cpu-') as directory:
            root = Path(directory)
            os.environ['Q38_TRAINING_AUDIT_DIR'] = directory
            hf = dict(prompt_ids=[11, 12], response_ids=[13, 14],
                      raw_log_probs=[-1., -2.],
                      raw_fp32_bits_hex=struct.pack('<2f', -1., -2.).hex())
            (root / 'hf-probe-step0-rank0.json').write_text(json.dumps(hf))
            (root / 'rtp_sync_rank0_step0.json').write_text(json.dumps(dict(full_source_sha256='a' * 64)))
            statuses = [SimpleNamespace(model_version='inc:1', global_step=0, update_id='',
                         received_weights=1590, applied_weights=1590, phase='running') for _ in range(72)]

            class Control:
                async def status(self): return statuses
                async def close(self): pass

            async def control(_topology): return Control()

            def require(value, message):
                if not value: raise ValueError(message)

            def write(path, value):
                path.write_text(json.dumps(value, allow_nan=False, sort_keys=True))

            request_ids = []
            counter = iter(range(100, 200))
            clients = []
            for value in native_values:
                class Client:
                    expected_step = expected_version = None
                    def __init__(self, fixture):
                        self.fixture = fixture
                        self.admission = False
                    def open_admission(self): self.admission = True
                    def close_admission(self, _reason): self.admission = False
                    async def score_tokens(self, request_id, prompt, target, *, expected_version):
                        assert prompt == [11, 12] and target == [13, 14] and expected_version == 'inc:1'
                        request_ids.append(request_id)
                        if isinstance(self.fixture, BaseException): raise self.fixture
                        return self.fixture
                clients.append(Client(value))

            async def close_admission(_reason): pass
            scope = dict(asyncio=asyncio, hashlib=hashlib, json=json, math=math,
                         os=os, Path=Path, struct=struct,
                         time=SimpleNamespace(time=lambda: 1700., time_ns=lambda: 1700000000000),
                         uuid=SimpleNamespace(uuid4=lambda: SimpleNamespace(int=next(counter))),
                         require=require, _control=control, _write_json=write,
                         _status_rows=lambda rows: [dict(global_step=s.global_step) for s in rows],
                         validate_policy_probe_sources=lambda sync, value, **kwargs: value)
            isolated = ast.fix_missing_locations(ast.Module(body=helper + [method], type_ignores=[]))
            exec(compile(isolated, kind + '-actual-controller-method-AST', 'exec'), scope)
            server = SimpleNamespace(_started=True, poisoned=False, _incarnation_failed=False,
                _generation_ready=False, _close_admission=close_admission, main_audit_root=root,
                topology=SimpleNamespace(generation_addresses=[f'fixture-dp-{i}' for i in range(9)]),
                boot=0, incarnation='inc',
                tokenizer=SimpleNamespace(encode=lambda text, **kwargs: [11, 12] if text.startswith('The value') else [13, 14]),
                clients=clients, _admission=asyncio.Lock())
            outcome = None
            try:
                result = await scope['_wake_and_probe_locked'](server, 0, 'inc:1')
                for key in ('actual_HF_reference_path', 'source_transaction_path'):
                    result[key] = '<CPU-fixture>/' + Path(result[key]).name
                outcome = dict(passed=True, record=result)
            except ValueError as error:
                outcome = dict(passed=False, error=str(error))
            assert request_ids == list(range(100, 109))
            outcome.update(generation_ready=server._generation_ready,
                           incarnation_failed=server._incarnation_failed,
                           clients_admitted=[client.admission for client in clients],
                           accepted_file_written=(root / 'rtp_probe_rank0_step0.json').exists())
            attempts = [json.loads(path.read_text()) for path in root.glob('rtp_probe_attempt_*.json')]
            if kind == 'new':
                assert len(attempts) == 1
                audit = attempts[0]
                assert audit['acceptance_proven_here'] is False
                assert audit['generation_admission_granted_here'] is False
                assert audit['numeric_gates_not_yet_evaluated'] is True
                assert audit['tolerance'] == .05 and len(audit['probes']) == 9
                assert audit['hf_raw_fp32_bits_hex'] == hf['raw_fp32_bits_hex']
                for value, row in zip(native_values, audit['probes']):
                    if isinstance(value, BaseException):
                        assert row['returned'] is False and repr(value) == row['error']
                    else:
                        assert row['parsed_fp32_bits_hex'] == struct.pack(f'<{len(value)}f', *value).hex()
                if any(not isinstance(v, BaseException) and any(not math.isfinite(x) for x in v) for v in native_values):
                    assert audit['all_nine_scores_finite_and_complete'] is False
            else: assert attempts == []
            return outcome

    cases = [
        ('success', [[-1.03125, -2.] for _ in range(9)]),
        ('HF_mismatch', [[-1.125, -2.] for _ in range(9)]),
        ('DP_mismatch', [[-1., -2.] for _ in range(8)] + [[-1.125, -2.]]),
        ('transport_error', [[-1., -2.] for _ in range(8)] + [RuntimeError('CPU transport fixture')]),
        ('nonfinite', [[-1., -2.] for _ in range(8)] + [[float('nan'), -0.]]),
        ('incomplete', [[-1., -2.] for _ in range(8)] + [[-1.]]),
    ]
    old = Path(a.old).read_text()
    new = Path(a.new).read_text()
    for name, values in cases:
        left = asyncio.run(run(old, 'old', values))
        right = asyncio.run(run(new, 'new', values))
        assert left == right, (name, left, right)
        reports.append(dict(case=name, original_behavior_identical=True, passed=right['passed'],
                            accepted_file_written=right['accepted_file_written'],
                            generation_ready=right['generation_ready']))
    report = dict(schema='q38-actual-controller-probe-attempt-CPU-qualification-v1', passed=True,
                  cases=reports, old_sha256=hashlib.sha256(old.encode()).hexdigest(),
                  new_sha256=hashlib.sha256(new.encode()).hexdigest(),
                  native_numerical_root_cause_fixed=False, actual_RL_accepted=False,
                  Torch_Ray_RPC_GPU_or_distributed_initialized=False)
    Path(a.output).write_text(json.dumps(report, indent=2) + '\n')
    print('PROBE_ATTEMPT_CPU_PASS', json.dumps(report), flush=True)


if __name__ == '__main__': main()
